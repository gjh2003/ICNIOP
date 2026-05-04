"""
Asymmetric Knowledge Distillation Finetune (AKDF).

Motivation:
    Post-hoc fusion (TBLF) caps at +0.16% Macro F1 because frozen CE has
    over-confident head logits that the additive tail boost cannot overcome.
    To go further, we need to BAKE the tail expert's knowledge into the
    backbone itself and SIMULTANEOUSLY suppress head over-confidence.

Method:
    Continue training the CE checkpoint with an asymmetric per-sample loss:

      • Tail samples (true label ∈ T):
            L = (1-β) · CE(student_logits, y)
                + β · KL( student_softmax_over_tail || TFE_softmax )

        This distils the TFE teacher's tail-class distribution into the
        student's tail logits. β controls the trade-off; β→0 = pure CE,
        β→1 = pure mimicry. The KL is restricted to tail classes (the
        teacher does not predict head classes).

      • Head samples (true label ∉ T):
            L = CE_LS(student_logits, y, ε=ε_head)

        Label smoothing pushes a small probability mass onto non-target
        classes — including tail classes — making head logits less
        over-confident. This widens the margin TBLF can later exploit.

    Game-theoretic reading:
        Each class plays a different role. Tail classes "outsource" their
        logit boundary to the TFE teacher (KD); head classes pay an
        ε-LS tax that cedes a sliver of probability mass to the tail.
        Asymmetric Nash bidding — head pays, tail receives.

Training:
    Initialise from CE checkpoint (mll_swin_t_ce_best.pth). Continue
    end-to-end finetune with low backbone lr, monitor val Macro F1,
    early-stop. After best model, also evaluate TBLF on top of AKDF
    logits to see if TBLF stacks.

Usage:
    python -m experiments.run_akdf
    python -m experiments.run_akdf --kd_beta 0.7 --ls_eps 0.1 --max_epochs 8
"""

import os
import sys
import copy
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR, print_paths
from common.train_utils import (
    setup_logger, EarlyStopping, save_resume_checkpoint, load_resume_checkpoint,
    History, compute_lt_metrics, set_seed,
)
from common.data import build_dataset
from common.models import build_backbone, ClassifierHead

from experiments.run_tfe import TFEHead, build_tail_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--primary_run", default="mll_swin_t_ce")
    p.add_argument("--tfe_run", default="mll_swin_t_tfe_thr500")
    p.add_argument("--tail_threshold", type=int, default=500)

    # AKDF hyperparameters
    p.add_argument("--kd_beta", type=float, default=0.3,
                   help="Weight on KD term for tail samples (0=pure CE, 1=pure mimicry)")
    p.add_argument("--kd_temperature", type=float, default=2.0,
                   help="Temperature for KD softmax (T>1 softens distributions)")
    p.add_argument("--kd_type", default="mse", choices=["mse", "kl"],
                   help="KD form: mse on tail logits (scale-preserving) or kl "
                        "on tail-subspace softmax (scale-invariant, can collapse)")
    p.add_argument("--ls_eps", type=float, default=0.1,
                   help="Label smoothing epsilon for head samples")

    # Training
    p.add_argument("--max_epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone_lr", type=float, default=5e-6,
                   help="Backbone lr (small since we start from a strong CE checkpoint)")
    p.add_argument("--head_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # TBLF on top of AKDF
    p.add_argument("--tblf_alphas", type=str, default="0.0,0.1,0.2,0.3,0.5",
                   help="Fixed-α TBLF sweep on the finetuned model")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.dataset, args.backbone, "akdf",
             f"b{args.kd_beta}", f"e{args.ls_eps}", f"T{args.kd_temperature}"]
    return "_".join(parts)


def load_ce_checkpoint(name, device, num_classes, backbone_type):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = build_backbone(backbone_type).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    return backbone, head


def load_frozen_tfe(name, device, feat_dim, n_tail):
    tfe_head = TFEHead(feat_dim, n_tail, hidden_dim=0).to(device)
    tfe_path = os.path.join(CHECKPOINT_DIR, f"{name}_head.pth")
    tfe_head.load_state_dict(torch.load(tfe_path, map_location=device, weights_only=True))
    tfe_head.eval()
    for p in tfe_head.parameters():
        p.requires_grad = False
    return tfe_head


# ===================== Asymmetric loss =====================
def akdf_loss(student_logits, labels, teacher_tfe_logits,
              tail_indices_tensor, is_tail_mask,
              kd_beta, kd_temp, ls_eps, kd_type="mse"):
    """
    student_logits: (B, K)
    labels: (B,)
    teacher_tfe_logits: (B, n_tail) — only meaningful where is_tail_mask
    tail_indices_tensor: (n_tail,) long
    is_tail_mask: (B,) bool — true if labels[i] is a tail class
    """
    device = student_logits.device

    # ---- Head branch: label-smoothed CE ----
    head_mask = ~is_tail_mask
    n_head = head_mask.sum().item()
    if n_head > 0:
        h_logits = student_logits[head_mask]
        h_labels = labels[head_mask]
        h_loss = F.cross_entropy(h_logits, h_labels, label_smoothing=ls_eps)
    else:
        h_loss = torch.tensor(0.0, device=device)

    # ---- Tail branch: (1-β)·CE + β·KL on tail subspace ----
    n_tail = is_tail_mask.sum().item()
    if n_tail > 0:
        t_logits = student_logits[is_tail_mask]            # (n_tail_b, K)
        t_labels = labels[is_tail_mask]
        t_teacher = teacher_tfe_logits[is_tail_mask]       # (n_tail_b, n_tail)

        ce_t = F.cross_entropy(t_logits, t_labels)

        # Student tail logits at the positions corresponding to tail classes
        student_tail_logits = t_logits.index_select(1, tail_indices_tensor)  # (n_tail_b, n_tail)
        if kd_type == "mse":
            # Scale-preserving: forces absolute logit values to match teacher's.
            # Teacher TFE logits live in their own scale; we normalise both by
            # subtracting per-sample mean so we match the *shape* not absolute level.
            s_centered = student_tail_logits - student_tail_logits.mean(dim=1, keepdim=True)
            t_centered = t_teacher - t_teacher.mean(dim=1, keepdim=True)
            kd = F.mse_loss(s_centered, t_centered)
        else:  # "kl" — original (collapses)
            student_log_p = F.log_softmax(student_tail_logits / kd_temp, dim=1)
            teacher_p = F.softmax(t_teacher / kd_temp, dim=1)
            kd = F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (kd_temp ** 2)

        t_loss = (1 - kd_beta) * ce_t + kd_beta * kd
    else:
        t_loss = torch.tensor(0.0, device=device)

    # Weighted average by sample counts in batch
    total = is_tail_mask.size(0)
    loss = (h_loss * n_head + t_loss * n_tail) / max(total, 1)
    return loss, h_loss.item() if isinstance(h_loss, torch.Tensor) else 0.0, \
           t_loss.item() if isinstance(t_loss, torch.Tensor) else 0.0


# ===================== Eval =====================
@torch.no_grad()
def evaluate(backbone, head, loader, class_names, priors, device, tag=""):
    backbone.eval(); head.eval()
    preds_all, trues_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = head(backbone(imgs))
        preds_all.append(logits.argmax(1).cpu())
        trues_all.append(labels)
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    return compute_lt_metrics(preds, trues, class_names, priors, tag=tag)


@torch.no_grad()
def collect_logits(backbone, head, tfe_head, loader, device):
    backbone.eval(); head.eval(); tfe_head.eval()
    all_ce, all_tfe, all_y = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        all_ce.append(head(feats).cpu())
        all_tfe.append(tfe_head(feats).cpu())
        all_y.append(labels)
    return torch.cat(all_ce), torch.cat(all_tfe), torch.cat(all_y)


def tblf_eval_fixed(test_ce, test_tfe, test_y, tail_idx_tensor, alpha,
                    class_names, priors):
    combined = test_ce.clone()
    combined[:, tail_idx_tensor] = combined[:, tail_idx_tensor] + alpha * test_tfe
    preds = combined.argmax(dim=1).numpy()
    trues = test_y.numpy() if torch.is_tensor(test_y) else test_y
    return compute_lt_metrics(preds, trues, class_names, priors, tag="")


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  AKDF (Asymmetric Knowledge Distillation Finetune) - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    priors = data["priors"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    # Tail metadata
    logger.info("[2] Determining tail class indices...")
    td = build_tail_dataset(args.dataset, args.tail_threshold,
                             args.batch_size, args.seed, args.num_workers)
    tail_orig_indices = td["tail_orig_indices"]
    tail_class_names = td["tail_class_names"]
    n_tail = len(tail_orig_indices)
    logger.info(f"  Tail classes ({n_tail}): {tail_class_names}")
    logger.info(f"  Tail orig indices: {tail_orig_indices}")
    tail_idx_tensor = torch.tensor(tail_orig_indices, dtype=torch.long, device=device)
    tail_idx_cpu = torch.tensor(tail_orig_indices, dtype=torch.long)

    # Lookup table for is_tail
    is_tail_lookup = torch.zeros(num_classes, dtype=torch.bool)
    for i in tail_orig_indices:
        is_tail_lookup[i] = True
    is_tail_lookup = is_tail_lookup.to(device)

    # Load CE checkpoint
    logger.info(f"[3] Loading CE checkpoint: {args.primary_run}")
    backbone, head = load_ce_checkpoint(args.primary_run, device, num_classes, args.backbone)

    # Frozen TFE teacher
    logger.info(f"[4] Loading frozen TFE teacher: {args.tfe_run}")
    tfe_head = load_frozen_tfe(args.tfe_run, device, backbone.feat_dim, n_tail)

    # Sanity: evaluate starting CE on val/test
    logger.info("[5] Sanity: starting CE checkpoint evaluation")
    val0 = evaluate(backbone, head, val_loader, class_names, priors, device, tag="val0")
    test0 = evaluate(backbone, head, test_loader, class_names, priors, device, tag="test0")
    logger.info(f"  CE start: val Macro F1={val0['macro_f1']:.4f}, "
                f"test Macro F1={test0['macro_f1']:.4f}, test Tail F1={test0['tail_f1']:.4f}")

    # Optimiser / scheduler
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    # Resume
    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1
    # Register starting CE as fallback
    early.step(val0["macro_f1"], 0,
               state={"backbone": copy.deepcopy(backbone.state_dict()),
                      "head": copy.deepcopy(head.state_dict())})

    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            backbone.load_state_dict(ckpt["backbone"])
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [RESUME] Loaded epoch {ckpt['epoch']}, best={early.best_value:.4f}")

    # Training loop
    logger.info(f"[6] AKDF finetune (β={args.kd_beta}, T={args.kd_temperature}, "
                f"ε_LS={args.ls_eps})...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        backbone.train(); head.train()
        train_loss = 0.0; train_correct = 0; train_total = 0
        head_loss_sum = 0.0; tail_loss_sum = 0.0
        n_head_b = 0; n_tail_b = 0

        pbar = tqdm(train_loader, desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)

            feats = backbone(imgs)
            s_logits = head(feats)
            with torch.no_grad():
                t_logits = tfe_head(feats.detach())

            is_tail_mask = is_tail_lookup[labels]
            loss, hl, tl = akdf_loss(s_logits, labels, t_logits,
                                      tail_idx_tensor, is_tail_mask,
                                      args.kd_beta, args.kd_temperature,
                                      args.ls_eps, args.kd_type)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (s_logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
            head_loss_sum += hl * (~is_tail_mask).sum().item()
            tail_loss_sum += tl * is_tail_mask.sum().item()
            n_head_b += (~is_tail_mask).sum().item()
            n_tail_b += is_tail_mask.sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                              tail=f"{n_tail_b}",
                              acc=f"{100*train_correct/train_total:.1f}%")

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total
        avg_h = head_loss_sum / max(n_head_b, 1)
        avg_t = tail_loss_sum / max(n_tail_b, 1)

        # Validate
        val_metrics = evaluate(backbone, head, val_loader, class_names, priors,
                               device, tag=f"E{epoch} val")
        history.append(epoch,
                       train_loss=train_loss, train_acc=train_acc,
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"],
                       lr=optimizer.param_groups[0]["lr"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={"backbone": copy.deepcopy(backbone.state_dict()),
                                    "head": copy.deepcopy(head.state_dict())})
        marker = "*best*" if is_best else f"(p {early.counter}/{args.patience})"
        logger.info(f"  E{epoch} {marker} loss={train_loss:.4f} (h={avg_h:.4f} t={avg_t:.4f}) "
                    f"train_acc={train_acc:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                    f"val_tail_f1={val_metrics['tail_f1']:.4f} val_head_f1={val_metrics['head_f1']:.4f}")

        save_resume_checkpoint(
            resume_path, epoch=epoch,
            backbone=backbone.state_dict(), head=head.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history.records,
            best_f1=early.best_value, best_epoch=early.best_epoch,
            patience_counter=early.counter, best_state=early.best_state,
        )

        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch} (best E{early.best_epoch})")
            break

    elapsed = (time.time() - t_start) / 60

    # Restore best
    if early.best_state is not None:
        backbone.load_state_dict(early.best_state["backbone"])
        head.load_state_dict(early.best_state["head"])

    # ============= Final evaluation =============
    logger.info(f"\n[7] Final evaluation with best AKDF model "
                f"(val Macro F1={early.best_value:.4f}, E{early.best_epoch})...")
    test_metrics = evaluate(backbone, head, test_loader, class_names, priors,
                            device, tag="Test")
    delta_ce = test_metrics["macro_f1"] - test0["macro_f1"]
    delta_tail = test_metrics["tail_f1"] - test0["tail_f1"]
    marker = "✓" if delta_ce > 0 else "✗"
    logger.info(f"  AKDF test: Macro F1={test_metrics['macro_f1']:.4f} "
                f"(Δ CE={delta_ce:+.4f}) Tail F1={test_metrics['tail_f1']:.4f} "
                f"(Δ={delta_tail:+.4f}) {marker}")

    # ============= TBLF on top of AKDF =============
    logger.info(f"\n[8] TBLF on top of AKDF (compounded gain)...")
    test_ce_new, test_tfe_new, test_y = collect_logits(
        backbone, head, tfe_head, test_loader, device)
    tblf_results = {}
    for a_str in args.tblf_alphas.split(","):
        a = float(a_str)
        m = tblf_eval_fixed(test_ce_new, test_tfe_new, test_y, tail_idx_cpu, a,
                             class_names, priors)
        d = m["macro_f1"] - test0["macro_f1"]
        marker = "✓" if d > 0 else "✗"
        logger.info(f"  AKDF + TBLF α={a:>4.2f}: Macro F1={m['macro_f1']:.4f} "
                    f"(Δ vs CE={d:+.4f}) Tail F1={m['tail_f1']:.4f} {marker}")
        tblf_results[f"akdf_tblf_alpha{a}"] = {k: v for k, v in m.items() if k != "report"}

    # ============= Save =============
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    try:
        history.plot_curves(os.path.join(save_dir, "training_curves.png"))
    except Exception:
        pass

    summary = {
        "run_name": run_name, "args": vars(args),
        "ce_baseline": {k: v for k, v in test0.items() if k != "report"},
        "akdf_test": {k: v for k, v in test_metrics.items() if k != "report"},
        "tblf_on_akdf": tblf_results,
        "best_val_macro_f1": early.best_value,
        "best_epoch": early.best_epoch,
        "training_min": elapsed,
        "tail_class_names": tail_class_names,
        "tail_orig_indices": tail_orig_indices,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n")
        f.write(f"Best epoch: {early.best_epoch}, val_macro_f1: {early.best_value:.4f}\n")
        f.write(f"Training time: {elapsed:.1f} min\n\n")
        f.write(f"CE baseline:  Macro F1={test0['macro_f1']:.4f} Tail F1={test0['tail_f1']:.4f}\n")
        f.write(f"AKDF only:    Macro F1={test_metrics['macro_f1']:.4f} "
                f"(Δ={test_metrics['macro_f1']-test0['macro_f1']:+.4f}) "
                f"Tail F1={test_metrics['tail_f1']:.4f}\n\n")
        for k, v in tblf_results.items():
            f.write(f"{k}: Macro F1={v['macro_f1']:.4f} "
                    f"(Δ={v['macro_f1']-test0['macro_f1']:+.4f}) "
                    f"Tail F1={v['tail_f1']:.4f}\n")
        f.write("\n" + test_metrics["report"])

    torch.save({
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "args": vars(args),
        "best_epoch": early.best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"},
    }, os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))

    logger.info(f"\n[9] DONE. Results in {save_dir}")
    logger.info(f"  CE → AKDF: Macro F1 {test0['macro_f1']:.4f} → "
                f"{test_metrics['macro_f1']:.4f} (Δ={delta_ce:+.4f})")
    best_tblf = max(tblf_results.values(), key=lambda v: v["macro_f1"])
    logger.info(f"  CE → AKDF+TBLF best: Macro F1 {test0['macro_f1']:.4f} → "
                f"{best_tblf['macro_f1']:.4f} "
                f"(Δ={best_tblf['macro_f1']-test0['macro_f1']:+.4f})")


if __name__ == "__main__":
    main()

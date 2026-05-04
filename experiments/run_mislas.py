"""
MiSLAS-Nash: MixUp pretraining + Label-Aware Smoothing as a Nash bidding tax.

Stage 1 (representation):
    Fresh Swin-T backbone + classifier head, trained from scratch with:
        - Standard cross-entropy
        - MixUp (Zhang et al. 2018) with α=0.2
        - Natural sampling
    For 25-30 epochs.

Stage 2 (classifier rebalancing, our Nash twist):
    Freeze backbone. Re-initialise classifier (cRT-style). Train with:
        - Class-balanced sampler
        - Asymmetric Label-Aware Smoothing:
            ε_k = ε_max · (1 - (count_k - count_min) / (count_max - count_min))
          → head classes get ε_k near 0 (tight target)
          → wait, MiSLAS does the OPPOSITE: head class gets HIGHER smoothing
            (because it's already over-confident; smoothing concedes mass).
          → ε_k = ε_max · ((count_k - count_min) / (count_max - count_min))
            head: ε_k ≈ ε_max  (smooth, concede mass to all classes)
            tail: ε_k ≈ 0      (peaked target, full mass on truth)

    Game-theoretic reading:
        Each class k pays a "label-mass tax" ε_k proportional to its frequency.
        Frequent classes pay more (their over-confidence is the imbalance toll).
        Rare classes are exempt. This is an asymmetric Nash equilibrium where
        the cost is set by frequency, not learned per-class.

    For 15 epochs, AdamW.
"""

import os
import sys
import copy
import time
import json
import argparse

import numpy as np
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t")

    # Stage 1
    p.add_argument("--s1_epochs", type=int, default=25)
    p.add_argument("--s1_patience", type=int, default=8)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Stage 2
    p.add_argument("--s2_epochs", type=int, default=15)
    p.add_argument("--s2_patience", type=int, default=5)
    p.add_argument("--s2_lr", type=float, default=5e-4)
    p.add_argument("--ls_eps_max", type=float, default=0.1,
                   help="ε for the most frequent (head) class. ε_min=0 for tail.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    return f"{args.dataset}_{args.backbone}_mislas"


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def label_aware_smoothing_loss(logits, labels, eps_per_class):
    """ε_k depends on the TRUE class.
    eps_per_class: (K,) tensor.
    For target y: target_dist = (1-ε_y) * onehot(y) + ε_y/K * 1.
    """
    K = logits.size(1)
    eps_y = eps_per_class[labels]                 # (B,)
    log_p = F.log_softmax(logits, dim=1)
    onehot_term = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
    uniform_term = log_p.mean(dim=1)              # (B,)
    loss = -((1 - eps_y) * onehot_term + eps_y * uniform_term)
    return loss.mean()


@torch.no_grad()
def evaluate(backbone, head, loader, class_names, priors, device, tag=""):
    backbone.eval(); head.eval()
    preds, trues = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = head(backbone(imgs))
        preds.append(logits.argmax(1).cpu())
        trues.append(labels)
    preds = torch.cat(preds).numpy(); trues = torch.cat(trues).numpy()
    return compute_lt_metrics(preds, trues, class_names, priors, tag=tag)


def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  MiSLAS-Nash - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)
    natural_loader = data["train_loader"]
    balanced_loader = data["balanced_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    priors = data["priors"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    # Models
    logger.info("[2] Building model...")
    backbone = build_backbone(args.backbone).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.s1_epochs, eta_min=1e-7)

    s1_resume = os.path.join(CHECKPOINT_DIR, f"{run_name}_s1_resume.pth")
    history = History()
    early = EarlyStopping(patience=args.s1_patience, mode="max")
    start_epoch = 1

    if args.resume:
        ckpt = load_resume_checkpoint(s1_resume)
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
            logger.info(f"  [RESUME] from E{ckpt['epoch']}, best={early.best_value:.4f}")

    # Stage 1
    logger.info(f"[3] Stage 1: CE + MixUp(α={args.mixup_alpha}), {args.s1_epochs} epochs...")
    t_start = time.time()
    for epoch in range(start_epoch, args.s1_epochs + 1):
        backbone.train(); head.train()
        train_loss = 0; n_seen = 0
        pbar = tqdm(natural_loader, desc=f"  S1 E{epoch}/{args.s1_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            x_mix, y_a, y_b, lam = mixup_batch(imgs, labels, args.mixup_alpha)
            logits = head(backbone(x_mix))
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            bs = imgs.size(0); train_loss += loss.item() * bs; n_seen += bs
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        train_loss /= n_seen

        val_m = evaluate(backbone, head, val_loader, class_names, priors, device,
                          tag=f"S1 E{epoch} val")
        history.append(epoch, train_loss=train_loss,
                       val_macro_f1=val_m["macro_f1"],
                       val_tail_f1=val_m["tail_f1"], val_head_f1=val_m["head_f1"])
        is_best = early.step(val_m["macro_f1"], epoch,
                             state={"backbone": copy.deepcopy(backbone.state_dict()),
                                    "head": copy.deepcopy(head.state_dict())})
        marker = "*best*" if is_best else f"(p {early.counter}/{args.s1_patience})"
        logger.info(f"  S1 E{epoch} {marker} loss={train_loss:.4f} "
                    f"val_macro_f1={val_m['macro_f1']:.4f} "
                    f"val_tail_f1={val_m['tail_f1']:.4f} val_head_f1={val_m['head_f1']:.4f}")

        save_resume_checkpoint(
            s1_resume, epoch=epoch,
            backbone=backbone.state_dict(), head=head.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history.records,
            best_f1=early.best_value, best_epoch=early.best_epoch,
            patience_counter=early.counter, best_state=early.best_state,
        )
        if early.should_stop:
            logger.info(f"  S1 early stop E{epoch} (best E{early.best_epoch})")
            break

    elapsed1 = (time.time() - t_start) / 60
    if early.best_state is not None:
        backbone.load_state_dict(early.best_state["backbone"])
        head.load_state_dict(early.best_state["head"])

    # Eval Stage 1 on test
    logger.info(f"\n[4] Stage 1 best on test...")
    s1_test = evaluate(backbone, head, test_loader, class_names, priors, device, tag="S1 test")
    logger.info(f"  S1 Test: Macro F1={s1_test['macro_f1']:.4f}, "
                f"Tail F1={s1_test['tail_f1']:.4f}")

    # ============= Stage 2: cRT + label-aware smoothing =============
    logger.info(f"\n[5] Stage 2: Freeze backbone, retrain classifier with "
                f"asymmetric LS (ε_max={args.ls_eps_max})")
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # Compute eps_k = ε_max · (count_k - count_min) / (count_max - count_min)
    counts = np.bincount([s[1] for s in natural_loader.dataset.samples], minlength=num_classes)
    cmin, cmax = counts.min(), counts.max()
    eps_per_class = args.ls_eps_max * (counts - cmin) / max(cmax - cmin, 1)
    eps_per_class_t = torch.tensor(eps_per_class, dtype=torch.float32, device=device)
    logger.info(f"  Per-class ε:")
    for i, name in enumerate(class_names):
        logger.info(f"    {name:8s} count={counts[i]:>6d} ε={eps_per_class[i]:.4f}")

    # Re-initialise classifier
    head_s2 = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    s2_opt = torch.optim.AdamW(head_s2.parameters(), lr=args.s2_lr, weight_decay=1e-4)
    s2_sched = CosineAnnealingLR(s2_opt, T_max=args.s2_epochs, eta_min=1e-6)
    s2_early = EarlyStopping(patience=args.s2_patience, mode="max")

    t_start2 = time.time()
    for epoch in range(1, args.s2_epochs + 1):
        head_s2.train()
        loss_sum = 0; n_seen = 0
        for imgs, labels in tqdm(balanced_loader, desc=f"  S2 E{epoch}/{args.s2_epochs}",
                                  ncols=100):
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                feat = backbone(imgs)
            logits = head_s2(feat)
            loss = label_aware_smoothing_loss(logits, labels, eps_per_class_t)
            s2_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head_s2.parameters(), 1.0)
            s2_opt.step()
            bs = imgs.size(0); loss_sum += loss.item() * bs; n_seen += bs
        s2_sched.step()

        val_m = evaluate(backbone, head_s2, val_loader, class_names, priors, device,
                          tag=f"S2 E{epoch} val")
        is_best = s2_early.step(val_m["macro_f1"], epoch,
                                state=copy.deepcopy(head_s2.state_dict()))
        marker = "*best*" if is_best else f"(p {s2_early.counter}/{args.s2_patience})"
        logger.info(f"  S2 E{epoch} {marker} loss={loss_sum/n_seen:.4f} "
                    f"val_macro_f1={val_m['macro_f1']:.4f} "
                    f"val_tail_f1={val_m['tail_f1']:.4f}")
        if s2_early.should_stop:
            break

    if s2_early.best_state is not None:
        head_s2.load_state_dict(s2_early.best_state)

    elapsed2 = (time.time() - t_start2) / 60
    s2_test = evaluate(backbone, head_s2, test_loader, class_names, priors, device, tag="S2 test")
    logger.info(f"\n[6] Stage 2 Test: Macro F1={s2_test['macro_f1']:.4f}, "
                f"Tail F1={s2_test['tail_f1']:.4f}")

    # Save
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    summary = {
        "run_name": run_name, "args": vars(args),
        "s1_test": {k: v for k, v in s1_test.items() if k != "report"},
        "s2_test": {k: v for k, v in s2_test.items() if k != "report"},
        "s1_min": elapsed1, "s2_min": elapsed2,
        "best_s1_epoch": early.best_epoch, "best_s1_val_macro": early.best_value,
        "best_s2_val_macro": s2_early.best_value,
        "eps_per_class": eps_per_class.tolist(),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, "test_report.txt"), "w") as f:
        f.write(f"Run: {run_name}\n\n")
        f.write(f"S1 Test: Macro F1={s1_test['macro_f1']:.4f} Tail F1={s1_test['tail_f1']:.4f}\n")
        f.write(f"S2 Test: Macro F1={s2_test['macro_f1']:.4f} Tail F1={s2_test['tail_f1']:.4f}\n\n")
        f.write(s2_test["report"])

    torch.save({
        "backbone": backbone.state_dict(),
        "head_s1": head.state_dict(),
        "head_s2": head_s2.state_dict(),
        "args": vars(args),
        "test_metrics": {k: v for k, v in s2_test.items() if k != "report"},
    }, os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))

    logger.info(f"\n[7] DONE. S1 {elapsed1:.1f} min + S2 {elapsed2:.1f} min")
    logger.info(f"  S1 Macro F1={s1_test['macro_f1']:.4f}, "
                f"S2 Macro F1={s2_test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

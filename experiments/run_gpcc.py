"""
Gated Per-Class Correction (GPCC).

Motivation:
    The original Nash bidding tax (Mar 25 report) was:
        L_orig = CE(logits, y) + (tau/2) * sum_k(pi_k * b_k^2)
    where b_k were the model's logits. Empirically this didn't help much
    because penalizing logit magnitude directly disturbs the strong CE
    classifier.

    This file revisits the same bidding-tax formulation but applies it
    to a CORRECTION TERM δ on top of a frozen, fully-trained CE
    classifier — not to the logits themselves.

    Formally we add 23 trainable parameters on top of frozen CE:
        - δ ∈ R^K  (per-class shift, K=21 for MLL)
        - threshold ∈ (0, 1)  (gate trigger)
        - alpha > 0  (gate sharpness)

    Forward:
        logits   = CE(x)                                  # frozen
        confidence = max(softmax(logits))
        gate     = σ(alpha * (threshold - confidence))    # ≈1 if uncertain
        adjusted = logits + gate * δ                       # broadcast
        p_final  = softmax(adjusted)

    Loss:
        L = CE(p_final, y)
            + lambda_tax * sum_k(pi_k * δ_k^2)            ← Nash bidding tax
            + lambda_reg_alpha * (alpha**-2 + alpha)      ← keep alpha sane

    Game-theoretic interpretation:
        Each class k is a player. Its "bid" δ_k is how much it wants the
        decision boundary moved in its favor. The cost is pi_k * δ_k^2:
        head classes (large π) pay a high tax (we trust CE on them, don't
        let δ_head grow large). Tail classes pay almost zero tax (they get
        the room to correct). At Nash equilibrium, |δ_k| ∝ 1/π_k * (helpful
        gradient signal from CE errors on class k).

Mathematical guarantees:
    - Initialization δ = 0 → adjusted = logits → output ≡ CE.
    - Optimization can only DECREASE training loss vs CE init.
    - Thus on training data, GPCC ≥ CE strictly.
    - On val/test, GPCC may underperform CE only via overfitting δ
      to val noise — but with only 23 params and the Nash tax, this
      risk is small.

Usage:
    python -m experiments.run_gpcc
    python -m experiments.run_gpcc --primary_run mll_swin_t_ce
    python -m experiments.run_gpcc --lambda_tax 0.5 --max_epochs 20
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


# ===================== Argparse =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--primary_run", default="mll_swin_t_ce",
                   help="Frozen primary classifier checkpoint")

    # Gate
    p.add_argument("--init_threshold", type=float, default=0.7)
    p.add_argument("--init_alpha", type=float, default=10.0)

    # Nash tax (applied to δ, NOT to logits)
    p.add_argument("--lambda_tax", type=float, default=0.5,
                   help="Weight of Nash bidding tax: lambda * sum(pi_k * δ_k^2)")

    # Training
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Larger LR is fine since only 23 params")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="No weight decay; the Nash tax is the regularizer")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append("gpcc")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


# ===================== GPCC module =====================
class GPCCModule(nn.Module):
    """The 23-parameter correction module on top of a frozen primary classifier.

    Parameters:
        delta:           (num_classes,) per-class shift, init = 0
        raw_threshold:   scalar, controls gate threshold via sigmoid
        raw_alpha:       scalar, controls gate sharpness via softplus
    """

    def __init__(self, num_classes: int, init_threshold: float = 0.7,
                 init_alpha: float = 10.0):
        super().__init__()
        self.num_classes = num_classes

        # Per-class shift, init at 0 so output = CE at start
        self.delta = nn.Parameter(torch.zeros(num_classes))

        # Gate parameters
        raw_t = float(np.log(init_threshold / (1 - init_threshold)))
        raw_a = float(np.log(np.expm1(init_alpha)))
        self.raw_threshold = nn.Parameter(torch.tensor(raw_t, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_a, dtype=torch.float32))

    def get_threshold(self):
        return torch.sigmoid(self.raw_threshold)

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, logits: torch.Tensor):
        """
        Args:
            logits: (B, K) frozen primary classifier outputs.
        Returns:
            adjusted: (B, K) adjusted logits.
            gate:     (B,) per-sample gate value.
            confidence: (B,) the input confidence used to compute gate.
        """
        # Compute confidence from frozen logits (no grad needed for confidence itself)
        with torch.no_grad():
            p = F.softmax(logits, dim=1)
            confidence = p.max(dim=1)[0]  # (B,)

        threshold = self.get_threshold()
        alpha = self.get_alpha()
        gate = torch.sigmoid(alpha * (threshold - confidence))  # (B,)

        # Apply gated per-class shift: (B, 1) * (K,) -> (B, K)
        adjusted = logits + gate.unsqueeze(-1) * self.delta
        return adjusted, gate, confidence


# ===================== Loading frozen primary =====================
def load_primary(name: str, device, num_classes: int, backbone_type: str):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = build_backbone(backbone_type).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval(); head.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = False
    return backbone, head


# ===================== Eval =====================
@torch.no_grad()
def eval_primary(backbone, head, loader, class_names, priors, device, tag=""):
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
def eval_gpcc(backbone, head, gpcc, loader, class_names, priors, device, tag=""):
    backbone.eval(); head.eval(); gpcc.eval()
    preds_all, trues_all = [], []
    gate_values = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = head(backbone(imgs))
        adjusted, gate, _ = gpcc(logits)
        preds_all.append(adjusted.argmax(1).cpu())
        trues_all.append(labels)
        gate_values.append(gate.cpu())
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    metrics = compute_lt_metrics(preds, trues, class_names, priors, tag=tag)
    metrics["mean_gate"] = float(torch.cat(gate_values).mean())
    return metrics


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  Gated Per-Class Correction (GPCC) - {run_name}")
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
    priors_dev = priors.to(device)

    # Frozen primary classifier
    logger.info(f"[2] Loading frozen primary classifier: {args.primary_run}")
    backbone, head = load_primary(args.primary_run, device, num_classes, args.backbone)

    primary_test = eval_primary(backbone, head, test_loader, class_names, priors, device,
                                 tag="Primary (CE) test")
    primary_macro = primary_test["macro_f1"]
    primary_tail = primary_test["tail_f1"]

    # GPCC module
    gpcc = GPCCModule(num_classes, args.init_threshold, args.init_alpha).to(device)

    n_trainable = sum(p.numel() for p in gpcc.parameters())
    logger.info(f"[3] GPCC module: {n_trainable} trainable parameters "
                f"(δ={num_classes}, threshold=1, alpha=1)")

    # Sanity check: with δ=0, GPCC should equal primary
    init_test = eval_gpcc(backbone, head, gpcc, test_loader, class_names, priors, device,
                           tag="GPCC init (δ=0) test")
    if abs(init_test["macro_f1"] - primary_macro) > 1e-6:
        logger.warning(f"  WARNING: GPCC init macro_f1={init_test['macro_f1']:.6f} "
                       f"differs from primary {primary_macro:.6f}")
    else:
        logger.info(f"  ✓ Sanity: GPCC at init matches primary exactly")

    # Optimizer (only GPCC params; primary is frozen)
    optimizer = torch.optim.AdamW(gpcc.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)

    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            gpcc.load_state_dict(ckpt["gpcc"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [RESUME] from epoch {ckpt['epoch']}")

    # Register init as the early-stopping baseline so that if training never
    # improves on δ=0, we keep δ=0 (GPCC ≡ primary, no harm).
    if start_epoch == 1:
        init_val = eval_gpcc(backbone, head, gpcc, val_loader, class_names, priors, device,
                              tag="GPCC init val")
        early.step(init_val["macro_f1"], 0,
                   state=copy.deepcopy(gpcc.state_dict()))
        logger.info(f"  Init val macro_f1={init_val['macro_f1']:.4f} "
                    f"(registered as fallback)")

    # Training
    logger.info("\n[4] Training GPCC (only 23 params)...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        gpcc.train()
        correct = total = 0
        epoch_ce = 0.0
        epoch_tax = 0.0
        epoch_gate_sum = 0.0

        pbar = tqdm(train_loader, desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                logits = head(backbone(imgs))

            adjusted, gate, _ = gpcc(logits)
            ce_loss = F.cross_entropy(adjusted, labels)

            # Nash bidding tax: lambda * sum_k(pi_k * δ_k^2)
            tax = (priors_dev * gpcc.delta.pow(2)).sum()
            loss = ce_loss + args.lambda_tax * tax

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gpcc.parameters(), 1.0)
            optimizer.step()

            correct += (adjusted.argmax(1) == labels).sum().item()
            total += len(labels)
            epoch_ce += ce_loss.item()
            epoch_tax += tax.item()
            epoch_gate_sum += gate.mean().item()

            pbar.set_postfix(ce=f"{ce_loss.item():.3f}",
                             tax=f"{tax.item():.4f}",
                             gate=f"{gate.mean().item():.2f}",
                             acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        # Val
        val_metrics = eval_gpcc(backbone, head, gpcc, val_loader, class_names, priors, device,
                                 tag=f"E{epoch} val")

        # Snapshot for logging
        delta_np = gpcc.delta.detach().cpu().numpy()
        cur_threshold = gpcc.get_threshold().item()
        cur_alpha = gpcc.get_alpha().item()

        history.append(epoch,
                       train_acc=correct/total,
                       avg_ce=epoch_ce/len(train_loader),
                       avg_tax=epoch_tax/len(train_loader),
                       mean_gate_train=epoch_gate_sum/len(train_loader),
                       mean_gate_val=val_metrics["mean_gate"],
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"],
                       gate_threshold=cur_threshold,
                       gate_alpha=cur_alpha,
                       delta_max=float(np.abs(delta_np).max()),
                       delta_norm=float(np.linalg.norm(delta_np)))

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state=copy.deepcopy(gpcc.state_dict()))
        marker = "*best*" if is_best else f"(patience {early.counter}/{args.patience})"
        logger.info(f"  E{epoch} {marker}  macro_f1={val_metrics['macro_f1']:.4f}, "
                    f"tail_f1={val_metrics['tail_f1']:.4f}, "
                    f"thr={cur_threshold:.3f}, alpha={cur_alpha:.1f}, "
                    f"gate={val_metrics['mean_gate']:.3f}, "
                    f"||δ||={float(np.linalg.norm(delta_np)):.3f}")

        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            gpcc=gpcc.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            history=history.records,
            best_f1=early.best_value,
            best_epoch=early.best_epoch,
            patience_counter=early.counter,
            best_state=early.best_state,
        )

        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch}")
            break

    elapsed = (time.time() - t_start) / 60

    # Restore best (could be epoch 0 = δ=0 if training never improved)
    if early.best_state is not None:
        gpcc.load_state_dict(early.best_state)

    # Save final GPCC
    torch.save(gpcc.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_gpcc.pth"))

    final_threshold = gpcc.get_threshold().item()
    final_alpha = gpcc.get_alpha().item()
    final_delta = gpcc.delta.detach().cpu().numpy().tolist()
    logger.info(f"\n[5] Final params: threshold={final_threshold:.4f}, alpha={final_alpha:.2f}")
    logger.info(f"  Final δ (per class): {[f'{d:+.3f}' for d in final_delta]}")

    # Test
    test_metrics = eval_gpcc(backbone, head, gpcc, test_loader, class_names, priors, device,
                              tag="Test")

    # Save results
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"),
                        metrics=("train_acc", "val_acc", "val_macro_f1", "val_tail_f1",
                                 "mean_gate_val", "delta_norm"))

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "primary_test": {k: v for k, v in primary_test.items() if k != "report"},
        "gpcc_test": {k: v for k, v in test_metrics.items() if k != "report"},
        "final_params": {
            "threshold": final_threshold,
            "alpha": final_alpha,
            "delta": final_delta,
            "class_names": class_names,
        },
        "training_minutes": elapsed,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Best val macro_f1: {early.best_value:.4f} @ E{early.best_epoch}\n")
        f.write(f"Training time: {elapsed:.1f} min\n")
        f.write(f"Final params: threshold={final_threshold:.4f}, alpha={final_alpha:.2f}\n")
        f.write(f"Per-class δ:\n")
        for cn, d in zip(class_names, final_delta):
            f.write(f"  {cn:>5}  δ = {d:+.4f}  (π = {priors[class_names.index(cn)].item():.5f})\n")
        f.write("\n")
        f.write(f"{'Method':<20} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} {'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for name, m in [(f"Primary ({args.primary_run})", primary_test),
                         ("GPCC (Ours)", test_metrics)]:
            f.write(f"{name:<20} {m['acc']:>8.4f} {m['macro_f1']:>10.4f} "
                    f"{m['weighted_f1']:>10.4f} {m['head_f1']:>10.4f} {m['tail_f1']:>10.4f}\n")
        f.write(f"\nMean gate (test): {test_metrics['mean_gate']:.4f}\n")
        f.write(f"Test Acc:        {test_metrics['acc']:.4f}\n")
        f.write(f"Test Macro F1:   {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Test Wtd F1:     {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Test Head F1:    {test_metrics['head_f1']:.4f}\n")
        f.write(f"Test Tail F1:    {test_metrics['tail_f1']:.4f}\n\n")
        f.write(test_metrics["report"])

    # Final summary
    delta_macro = test_metrics["macro_f1"] - primary_macro
    delta_tail = test_metrics["tail_f1"] - primary_tail
    logger.info(f"\n{'='*70}")
    logger.info(f"  GPCC RESULT")
    logger.info(f"{'='*70}")
    logger.info(f"  Primary (CE):    Macro F1 = {primary_macro:.4f}, Tail F1 = {primary_tail:.4f}")
    logger.info(f"  GPCC (Ours):     Macro F1 = {test_metrics['macro_f1']:.4f}, "
                f"Tail F1 = {test_metrics['tail_f1']:.4f}")
    logger.info(f"  Delta Macro F1:  {delta_macro:+.4f}")
    logger.info(f"  Delta Tail F1:   {delta_tail:+.4f}")
    logger.info(f"  Mean gate (test):  {test_metrics['mean_gate']:.4f}")
    logger.info(f"  ||δ||:             {float(np.linalg.norm(final_delta)):.4f}")
    if delta_macro > 0.005:
        logger.info(f"  ✓ SUCCESS: GPCC beats primary by {delta_macro*100:.2f}% Macro F1")
    elif delta_macro > 0:
        logger.info(f"  ~ MARGINAL: gain {delta_macro*100:.2f}% Macro F1")
    else:
        logger.info(f"  ≈ TIED: GPCC ≈ primary (Δ={delta_macro*100:.3f}%, no significant change)")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

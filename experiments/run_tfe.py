"""
Tail-Focused Expert (TFE).

Motivation:
    Previous attempts at "tail specialists" (Weighted CE, LDAM-DRW, cRT,
    Logit Adjustment) all underperformed plain CE on tail classes.
    Reason: they shared the SAME 21-class output space with the primary
    classifier, so head-class gradients dominated training and the tail
    "specialty" never materialized.

    This script trains a TRUE tail specialist:
        - Output dimension = num_tail_classes (e.g., 7), NOT 21
        - Training data = ONLY tail-class samples (~900 images)
        - 7-way classification is much easier than 21-way
        - Frozen CE backbone provides strong features
        - Only a Linear(feat_dim, num_tail_classes) is trained

    If TFE on its 7 tail classes beats CE's per-class F1 on those same 7
    classes, we have a genuinely complementary expert. Then we can route:
        if CE predicts a head class with high confidence → use CE
        if CE predicts a tail class OR is uncertain     → use TFE

Workflow:
    Step 1: Train TFE alone, isolated.
    Step 2: Sanity-check: TFE per-tail-class F1 vs CE per-tail-class F1.
            If TFE doesn't win on most tail classes, the route is dead-end.
    Step 3: Build router (head_confidence-gated mixture).

This script handles Step 1 and Step 2. Step 3 is in run_tfe_route.py.

Usage:
    python -m experiments.run_tfe
    python -m experiments.run_tfe --tail_threshold 500
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR, print_paths
from common.train_utils import (
    setup_logger, EarlyStopping, save_resume_checkpoint, load_resume_checkpoint,
    History, set_seed,
)
from common.data import build_dataset, SubsetDataset, get_transforms, safe_loader
from common.models import build_backbone, ClassifierHead

# We need raw access to dataset for filtering; reuse internals
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--primary_run", default="mll_swin_t_ce")

    # What counts as a "tail class"?
    p.add_argument("--tail_threshold", type=int, default=500,
                   help="Classes with training count <= this are tail classes")

    # Training
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Smaller batch since dataset is tiny")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--head_hidden", type=int, default=0,
                   help="Hidden dim of TFE head (0 = pure linear)")
    p.add_argument("--use_class_weight", action="store_true",
                   help="Use inverse-frequency class weights within tail subset")

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
    parts.append(f"tfe_thr{args.tail_threshold}")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


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


def build_tail_dataset(dataset_name: str, tail_threshold: int, batch_size: int,
                       seed: int = 42, num_workers: int = 4):
    """Build train/val/test loaders FILTERED to only tail-class samples.

    Same split logic as common.data.build_dataset (70/15/15 stratified, same seed)
    but then filtered to tail classes only.

    Returns:
        train_loader, val_loader, test_loader (only tail samples)
        tail_class_indices: original 21-class label → 7-way label mapping
        original_class_names: full 21 class names
        tail_class_names: 7 tail class names
        tail_priors: (7,) priors of tail classes (within tail subset)
        n_per_tail_class: dict of training counts
    """
    from common.paths import MLL_DIR, PBC_DIR
    if dataset_name == "mll":
        data_dir = MLL_DIR
    else:
        data_dir = PBC_DIR

    train_tf, val_tf = get_transforms()

    base_ds = ImageFolder(root=data_dir, loader=safe_loader)
    full_class_names = base_ds.classes
    all_samples = base_ds.samples
    all_labels = [s[1] for s in all_samples]

    # Same train/val/test split as common.data
    train_s, temp_s, train_l, temp_l = train_test_split(
        all_samples, all_labels, test_size=0.3, stratify=all_labels, random_state=seed)
    val_s, test_s = train_test_split(
        temp_s, test_size=0.5, stratify=temp_l, random_state=seed)

    # Determine tail classes by training count
    train_label_arr = np.array([s[1] for s in train_s])
    counts = np.bincount(train_label_arr, minlength=len(full_class_names))
    tail_orig_indices = sorted([i for i, c in enumerate(counts) if c <= tail_threshold])
    tail_class_names = [full_class_names[i] for i in tail_orig_indices]
    n_tail = len(tail_orig_indices)
    orig_to_tail = {orig: i for i, orig in enumerate(tail_orig_indices)}

    # Filter splits and remap labels
    def filter_remap(samples):
        return [(p, orig_to_tail[lbl]) for (p, lbl) in samples if lbl in orig_to_tail]

    train_s_tail = filter_remap(train_s)
    val_s_tail = filter_remap(val_s)
    test_s_tail = filter_remap(test_s)

    train_ds = SubsetDataset(train_s_tail, train_tf)
    val_ds = SubsetDataset(val_s_tail, val_tf)
    test_ds = SubsetDataset(test_s_tail, val_tf)

    # Per-class training counts after filtering
    tail_train_labels = np.array([s[1] for s in train_s_tail])
    tail_counts = np.bincount(tail_train_labels, minlength=n_tail)
    tail_priors = torch.tensor(
        (tail_counts.astype(np.float64) + 1) / (len(train_s_tail) + n_tail),
        dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tail_orig_indices": tail_orig_indices,
        "orig_to_tail": orig_to_tail,
        "full_class_names": full_class_names,
        "tail_class_names": tail_class_names,
        "tail_priors": tail_priors,
        "tail_train_counts": dict(zip(tail_class_names, tail_counts.tolist())),
        "n_train": len(train_s_tail),
        "n_val": len(val_s_tail),
        "n_test": len(test_s_tail),
    }


# ===================== TFE Head =====================
class TFEHead(nn.Module):
    """The trainable head for the tail expert. Either pure Linear or small MLP."""

    def __init__(self, in_dim: int, n_tail: int, hidden_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, n_tail)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_tail),
            )

    def forward(self, x):
        return self.net(x)


# ===================== Eval =====================
@torch.no_grad()
def eval_tfe(backbone, tfe_head, loader, tail_class_names, device, tag=""):
    """Evaluate TFE on tail-only test loader."""
    backbone.eval(); tfe_head.eval()
    preds_all, trues_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = tfe_head(feats)
        preds_all.append(logits.argmax(1).cpu())
        trues_all.append(labels)
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()

    acc = (preds == trues).mean() if len(trues) else 0.0
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    per_f1 = f1_score(trues, preds, average=None,
                      labels=list(range(len(tail_class_names))),
                      zero_division=0)
    report = classification_report(trues, preds, target_names=tail_class_names,
                                    labels=list(range(len(tail_class_names))),
                                    zero_division=0)
    print(f"  [{tag}] (tail-only) Acc={acc:.4f}, Macro F1={macro_f1:.4f}, Wtd F1={weighted_f1:.4f}")
    return {
        "acc": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "per_f1": per_f1.tolist(), "preds": preds, "trues": trues, "report": report,
    }


@torch.no_grad()
def eval_ce_on_tail(backbone, head, loader, tail_orig_indices, full_class_names, device, tag=""):
    """Evaluate CE on tail-only loader, filtering its 21-way prediction to compare.

    For samples whose true label is a tail class, look at CE's argmax over 21 classes.
    Calculate per-tail-class F1 by treating "predicted as some non-tail class" as wrong.
    """
    backbone.eval(); head.eval()
    n_tail = len(tail_orig_indices)
    orig_to_tail = {orig: i for i, orig in enumerate(tail_orig_indices)}

    preds_in_tail_space = []
    trues_in_tail_space = []
    for imgs, labels_in_tail in loader:
        imgs = imgs.to(device)
        logits = head(backbone(imgs))
        pred_orig = logits.argmax(1).cpu().numpy()
        # Map predictions back to tail-space; non-tail predictions become "wrong"
        # We use n_tail (out-of-vocab) to mark non-tail predictions
        for p in pred_orig:
            if p in orig_to_tail:
                preds_in_tail_space.append(orig_to_tail[p])
            else:
                preds_in_tail_space.append(n_tail)  # sentinel for "head-class prediction"
        trues_in_tail_space.extend(labels_in_tail.numpy().tolist())

    preds_arr = np.array(preds_in_tail_space)
    trues_arr = np.array(trues_in_tail_space)

    # F1 per tail class. For the sentinel (=n_tail), we don't include in macro avg.
    tail_class_names = [full_class_names[i] for i in tail_orig_indices]
    per_f1 = f1_score(trues_arr, preds_arr,
                      labels=list(range(n_tail)),
                      average=None, zero_division=0)
    macro_f1_on_tail = float(np.mean(per_f1))
    acc_on_tail = float((preds_arr == trues_arr).mean())

    # Detailed report
    report = classification_report(trues_arr, preds_arr,
                                    labels=list(range(n_tail)),
                                    target_names=tail_class_names,
                                    zero_division=0)
    print(f"  [{tag}] (CE on tail samples) Acc={acc_on_tail:.4f}, "
          f"Macro F1={macro_f1_on_tail:.4f}")
    return {
        "acc": acc_on_tail,
        "macro_f1": macro_f1_on_tail,
        "per_f1": per_f1.tolist(),
        "preds": preds_arr, "trues": trues_arr,
        "report": report,
    }


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  Tail-Focused Expert (TFE) - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Build TAIL-ONLY data
    logger.info(f"[1] Building tail-only dataset (threshold={args.tail_threshold})...")
    td = build_tail_dataset(args.dataset, args.tail_threshold,
                             args.batch_size, args.seed, args.num_workers)

    n_tail = len(td["tail_orig_indices"])
    logger.info(f"  Tail classes ({n_tail}): {td['tail_class_names']}")
    logger.info(f"  Per-class training counts: {td['tail_train_counts']}")
    logger.info(f"  Splits: train={td['n_train']}, val={td['n_val']}, test={td['n_test']}")

    # Frozen primary
    logger.info(f"[2] Loading frozen primary: {args.primary_run}")
    backbone, ce_head = load_primary(args.primary_run, device,
                                      num_classes=len(td["full_class_names"]),
                                      backbone_type=args.backbone)
    feat_dim = backbone.feat_dim

    # Sanity: how does CE do on tail samples?
    logger.info("[3] Sanity: CE on tail-only test set...")
    ce_on_tail = eval_ce_on_tail(backbone, ce_head, td["test_loader"],
                                  td["tail_orig_indices"], td["full_class_names"],
                                  device, tag="CE-on-tail")
    for cn, f1 in zip(td["tail_class_names"], ce_on_tail["per_f1"]):
        logger.info(f"    CE  {cn:<5}: F1 = {f1:.4f}")

    # Build TFE head
    tfe_head = TFEHead(feat_dim, n_tail, hidden_dim=args.head_hidden).to(device)
    n_trainable = sum(p.numel() for p in tfe_head.parameters())
    logger.info(f"[4] TFE head: hidden={args.head_hidden} (0=Linear), "
                f"{n_trainable} trainable params")

    # Optional class weights (within tail subset)
    if args.use_class_weight:
        w = 1.0 / td["tail_priors"]
        w = w / w.sum() * n_tail
        loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
        logger.info(f"  Using class weights: {w.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(tfe_head.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            tfe_head.load_state_dict(ckpt["tfe_head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [RESUME] from epoch {ckpt['epoch']}")

    # Train
    logger.info("\n[5] Training TFE on tail-only data...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        tfe_head.train()
        correct = total = 0
        loss_sum = 0.0
        pbar = tqdm(td["train_loader"], desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = tfe_head(feats)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tfe_head.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = eval_tfe(backbone, tfe_head, td["val_loader"],
                                td["tail_class_names"], device, tag=f"E{epoch} val")

        history.append(epoch,
                       train_loss=loss_sum/max(total, 1),
                       train_acc=correct/max(total, 1),
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_weighted_f1=val_metrics["weighted_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state=copy.deepcopy(tfe_head.state_dict()))
        marker = "*best*" if is_best else f"(patience {early.counter}/{args.patience})"
        logger.info(f"  E{epoch} {marker}  val_macro_f1={val_metrics['macro_f1']:.4f}, "
                    f"val_acc={val_metrics['acc']:.4f}")

        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            tfe_head=tfe_head.state_dict(),
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

    # Restore best
    if early.best_state is not None:
        tfe_head.load_state_dict(early.best_state)

    # Save
    torch.save(tfe_head.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_head.pth"))

    # Test
    tfe_test = eval_tfe(backbone, tfe_head, td["test_loader"],
                         td["tail_class_names"], device, tag="TFE test")

    # Comparison: TFE vs CE on tail samples
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)

    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"))

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "tail_class_names": td["tail_class_names"],
        "tail_orig_indices": td["tail_orig_indices"],
        "tail_train_counts": td["tail_train_counts"],
        "ce_on_tail_test": {
            "macro_f1": ce_on_tail["macro_f1"],
            "acc": ce_on_tail["acc"],
            "per_f1": ce_on_tail["per_f1"],
        },
        "tfe_test": {
            "macro_f1": tfe_test["macro_f1"],
            "acc": tfe_test["acc"],
            "weighted_f1": tfe_test["weighted_f1"],
            "per_f1": tfe_test["per_f1"],
        },
        "tail_priors": td["tail_priors"].tolist(),
        "training_minutes": elapsed,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Tail classes ({n_tail}): {td['tail_class_names']}\n")
        f.write(f"Tail train counts: {td['tail_train_counts']}\n")
        f.write(f"Test set sizes: train={td['n_train']}, val={td['n_val']}, test={td['n_test']}\n")
        f.write(f"Best val macro_f1: {early.best_value:.4f} @ E{early.best_epoch}\n")
        f.write(f"Training time: {elapsed:.1f} min\n\n")

        f.write("=== Per-tail-class F1 comparison (test set) ===\n")
        f.write(f"{'Class':<8} {'CE F1':>10} {'TFE F1':>10} {'Diff':>10}\n")
        f.write("-" * 45 + "\n")
        for i, cn in enumerate(td["tail_class_names"]):
            ce_f1 = ce_on_tail["per_f1"][i]
            tfe_f1 = tfe_test["per_f1"][i]
            f.write(f"{cn:<8} {ce_f1:>10.4f} {tfe_f1:>10.4f} {tfe_f1-ce_f1:>+10.4f}\n")
        f.write(f"{'macro':<8} {ce_on_tail['macro_f1']:>10.4f} {tfe_test['macro_f1']:>10.4f} "
                f"{tfe_test['macro_f1']-ce_on_tail['macro_f1']:>+10.4f}\n\n")

        f.write("=== TFE classification report ===\n")
        f.write(tfe_test["report"])
        f.write("\n=== CE-on-tail classification report ===\n")
        f.write(ce_on_tail["report"])

    # Final summary
    delta = tfe_test["macro_f1"] - ce_on_tail["macro_f1"]
    logger.info(f"\n{'='*70}")
    logger.info(f"  TFE RESULT (only on tail-class samples)")
    logger.info(f"{'='*70}")
    logger.info(f"  CE on tail samples:    Macro F1 = {ce_on_tail['macro_f1']:.4f}")
    logger.info(f"  TFE on tail samples:   Macro F1 = {tfe_test['macro_f1']:.4f}")
    logger.info(f"  Delta (TFE - CE):      {delta:+.4f}")
    logger.info(f"  ----------------------------------------------------------------")
    logger.info(f"  Per-tail-class F1:")
    for i, cn in enumerate(td["tail_class_names"]):
        ce_f1 = ce_on_tail["per_f1"][i]
        tfe_f1 = tfe_test["per_f1"][i]
        diff = tfe_f1 - ce_f1
        marker = "✓" if diff > 0 else ("=" if abs(diff) < 1e-4 else "✗")
        logger.info(f"  {marker} {cn:<5}  CE={ce_f1:.4f}  TFE={tfe_f1:.4f}  Δ={diff:+.4f}")
    logger.info(f"")
    if delta > 0.02:
        logger.info(f"  ✓ TFE WINS on tail: gain {delta*100:.2f}% Macro F1 over CE.")
        logger.info(f"     Next step: train router (run_tfe_route.py).")
    elif delta > 0:
        logger.info(f"  ~ Marginal: TFE gains {delta*100:.2f}% over CE. Try run_tfe_route.")
    else:
        logger.info(f"  ✗ TFE LOSES: gain {delta*100:.2f}% over CE on tail.")
        logger.info(f"     The bottleneck is data scarcity, not architecture.")
        logger.info(f"     Recommendation: switch to benchmark/interpretability narrative.")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

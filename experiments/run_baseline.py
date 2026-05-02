"""
Unified end-to-end baseline trainer.

Supported methods:
  - ce             Standard Cross-Entropy
  - weighted_ce    Weighted CE (weights = 1/pi_k)
  - focal          Focal Loss (gamma=2)
  - ldam           LDAM (no DRW)
  - ldam_drw       LDAM + Deferred Re-Weighting
  - logit_adjustment   Logit Adjustment (tau=1.0)
  - gala           Gradient-Aware Logit Adjustment

All methods use: same backbone (Swin-T or ResNet50), same training schedule,
early stopping, resume support, full per-epoch logging.

Usage:
    python -m experiments.run_baseline --method ce --dataset mll
    python -m experiments.run_baseline --method ldam_drw --dataset mll --backbone resnet50
    python -m experiments.run_baseline --method logit_adjustment --dataset pbc --variant lt --imb_factor 100
"""

import os
import sys
import copy
import time
import json
import argparse

import torch
import torch.nn as nn
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
from common.losses import build_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["ce", "weighted_ce", "focal", "ldam", "ldam_drw",
                            "logit_adjustment", "gala"])
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if exists")

    # Method-specific hyperparameters
    p.add_argument("--la_tau", type=float, default=1.0, help="Logit adjustment tau")
    p.add_argument("--ldam_max_m", type=float, default=0.5)
    p.add_argument("--ldam_s", type=float, default=30.0)
    p.add_argument("--drw_start", type=int, default=20, help="LDAM-DRW start epoch")
    p.add_argument("--gala_eta", type=float, default=0.5)
    return p.parse_args()


def make_run_name(args):
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append(args.method)
    if args.method == "logit_adjustment":
        parts.append(f"tau{args.la_tau}")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


def evaluate(backbone, head, loader, class_names, priors, device, tag=""):
    backbone.eval(); head.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = head(backbone(imgs))
            preds_all.append(logits.argmax(1).cpu())
            trues_all.append(labels)
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    return compute_lt_metrics(preds, trues, class_names, priors, tag=tag)


def main():
    args = parse_args()
    run_name = make_run_name(args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  Run: {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    logger.info(f"  Device: {device}")
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

    # Model
    logger.info("[2] Building model...")
    backbone = build_backbone(args.backbone).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)

    # Optimizer (differential LR for backbone vs head)
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    # Loss (some methods need to be re-built each epoch)
    def build_current_loss(epoch):
        kwargs = dict(num_classes=num_classes, epoch=epoch,
                      tau=args.la_tau, max_m=args.ldam_max_m,
                      s=args.ldam_s, drw_start=args.drw_start, eta=args.gala_eta)
        return build_loss(args.method, priors, device, **kwargs)

    # Resume
    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1

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
            logger.info(f"  [RESUME] Loaded from epoch {ckpt['epoch']}, best val_macro_f1={early.best_value:.4f}")

    # Training loop
    logger.info("[3] Training...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        criterion = build_current_loss(epoch)
        backbone.train(); head.train()
        train_loss = 0.0; train_correct = 0; train_total = 0

        pbar = tqdm(train_loader, desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = head(backbone(imgs))
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*train_correct/train_total:.1f}%")

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        val_metrics = evaluate(backbone, head, val_loader, class_names, priors, device,
                               tag=f"E{epoch} val")
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
        if is_best:
            logger.info(f"  E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  E{epoch} (patience {early.counter}/{args.patience})")

        # Save resume checkpoint every epoch
        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            backbone=backbone.state_dict(),
            head=head.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            history=history.records,
            best_f1=early.best_value,
            best_epoch=early.best_epoch,
            patience_counter=early.counter,
            best_state=early.best_state,
        )

        if early.should_stop:
            logger.info(f"  Early stopping at epoch {epoch} (best={early.best_value:.4f} @ E{early.best_epoch})")
            break

    elapsed = (time.time() - t_start) / 60

    # Restore best model and evaluate on test
    if early.best_state is not None:
        backbone.load_state_dict(early.best_state["backbone"])
        head.load_state_dict(early.best_state["head"])

    logger.info(f"\n[4] Test evaluation (best model, val_macro_f1={early.best_value:.4f})")
    test_metrics = evaluate(backbone, head, test_loader, class_names, priors, device, tag="Test")

    # Save results
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)

    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"))

    with open(os.path.join(save_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in test_metrics.items() if k != "report"}, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Args: {vars(args)}\n")
        f.write(f"Best epoch (val): {early.best_epoch}, val_macro_f1: {early.best_value:.4f}\n")
        f.write(f"Training time: {elapsed:.1f} min\n\n")
        f.write(f"Test Acc:        {test_metrics['acc']:.4f}\n")
        f.write(f"Test Macro F1:   {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Test Wtd F1:     {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Test Head F1:    {test_metrics['head_f1']:.4f}\n")
        f.write(f"Test Tail F1:    {test_metrics['tail_f1']:.4f}\n\n")
        f.write(test_metrics["report"])

    # Save final best model checkpoint
    torch.save({
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "args": vars(args),
        "best_epoch": early.best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"},
    }, os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))

    logger.info(f"\n[5] DONE. Results saved to {save_dir}")
    logger.info(f"     Total time: {elapsed:.1f} min")
    logger.info(f"     Test Macro F1: {test_metrics['macro_f1']:.4f}, Tail F1: {test_metrics['tail_f1']:.4f}")


if __name__ == "__main__":
    main()

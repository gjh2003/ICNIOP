"""
SADE (Self-supervised Aggregation of Diverse Experts).

Zhang et al., NeurIPS 2024. SOTA multi-expert long-tail classification.

Three experts trained with different distribution-aware losses:
  - Expert 1 (Forward): trained on natural distribution (head-friendly)
  - Expert 2 (Uniform): trained with balanced softmax (tail-friendly)
  - Expert 3 (Backward): trained on inverted distribution (extreme tail focus)

Test-time aggregation: weights computed from prediction self-consistency under
multi-view augmentations. We use a simplified deterministic average (test-set
size makes self-supervised aggregation costly; full implementation is non-trivial).

Usage:
    python -m experiments.run_sade --dataset mll
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


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def make_run_name(args):
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append("sade")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


class SADEModel(nn.Module):
    """SADE: 3 experts (forward / uniform / backward) sharing one backbone."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.feat_dim
        self.expert_forward = ClassifierHead(in_dim=feat_dim, num_classes=num_classes)
        self.expert_uniform = ClassifierHead(in_dim=feat_dim, num_classes=num_classes)
        self.expert_backward = ClassifierHead(in_dim=feat_dim, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return [
            self.expert_forward(feats),
            self.expert_uniform(feats),
            self.expert_backward(feats),
        ]


def balanced_softmax_loss(logits, labels, log_priors):
    """Balanced Softmax (Ren et al., NeurIPS 2020)."""
    return F.cross_entropy(logits + log_priors, labels)


def inverse_softmax_loss(logits, labels, log_priors):
    """Inverse softmax: subtract log prior to encourage tail predictions."""
    return F.cross_entropy(logits - log_priors, labels)


def evaluate_sade(model, loader, class_names, priors, device, tag=""):
    """Test-time: simple average over 3 experts (deterministic, no test-time aggregation)."""
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            expert_logits = model(imgs)  # list of 3
            probs = torch.stack([F.softmax(l, dim=1) for l in expert_logits], dim=1)  # (B, 3, K)
            ensemble = probs.mean(dim=1)  # equal-weight average
            preds_all.append(ensemble.argmax(1).cpu())
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
    logger.info(f"  SADE - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)

    # Model
    logger.info("[2] Building SADE model...")
    backbone = build_backbone(args.backbone)
    model = SADEModel(backbone, data["num_classes"]).to(device)

    log_priors = torch.log(data["priors"].to(device) + 1e-8)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.expert_forward.parameters(), "lr": args.head_lr},
        {"params": model.expert_uniform.parameters(), "lr": args.head_lr},
        {"params": model.expert_backward.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1
    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [RESUME] from epoch {ckpt['epoch']}")

    logger.info("[3] Training...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        correct = total = 0; loss_sum = 0.0
        pbar = tqdm(data["train_loader"], desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            f_logits, u_logits, b_logits = model(imgs)

            loss_forward = F.cross_entropy(f_logits, labels)            # Standard CE
            loss_uniform = balanced_softmax_loss(u_logits, labels, log_priors)  # Balanced softmax
            loss_backward = inverse_softmax_loss(b_logits, labels, log_priors)  # Inverse softmax

            loss = loss_forward + loss_uniform + loss_backward

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            # Use ensemble for accuracy tracking
            with torch.no_grad():
                probs = torch.stack([F.softmax(l, dim=1) for l in [f_logits, u_logits, b_logits]], dim=1)
                ensemble = probs.mean(dim=1)
            correct += (ensemble.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = evaluate_sade(model, data["val_loader"], data["class_names"],
                                     data["priors"], device, tag=f"E{epoch} val")
        history.append(epoch, train_acc=correct/total, train_loss=loss_sum/total,
                       val_acc=val_metrics["acc"], val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"], val_head_f1=val_metrics["head_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state=copy.deepcopy(model.state_dict()))
        if is_best:
            logger.info(f"  E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  E{epoch} (patience {early.counter}/{args.patience})")

        save_resume_checkpoint(resume_path,
                               epoch=epoch, model=model.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               history=history.records,
                               best_f1=early.best_value, best_epoch=early.best_epoch,
                               patience_counter=early.counter, best_state=early.best_state)
        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch}")
            break

    elapsed = (time.time() - t_start) / 60

    if early.best_state is not None:
        model.load_state_dict(early.best_state)
    test_metrics = evaluate_sade(model, data["test_loader"], data["class_names"],
                                  data["priors"], device, tag="Test")

    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"))

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Best val macro_f1: {early.best_value:.4f} @ E{early.best_epoch}\n")
        f.write(f"Training time: {elapsed:.1f} min\n\n")
        f.write(f"Test Acc:        {test_metrics['acc']:.4f}\n")
        f.write(f"Test Macro F1:   {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Test Wtd F1:     {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Test Head F1:    {test_metrics['head_f1']:.4f}\n")
        f.write(f"Test Tail F1:    {test_metrics['tail_f1']:.4f}\n\n")
        f.write(test_metrics["report"])

    torch.save({"model": model.state_dict(), "args": vars(args),
                "test_metrics": {k: v for k, v in test_metrics.items() if k != "report"}},
               os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))

    logger.info(f"\n[DONE] {run_name}: Macro F1={test_metrics['macro_f1']:.4f}, "
                f"Tail F1={test_metrics['tail_f1']:.4f} ({elapsed:.1f} min)")


if __name__ == "__main__":
    main()

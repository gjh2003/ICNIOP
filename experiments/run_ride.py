"""
RIDE (Routing Diverse Distribution-Aware Experts).

Wang et al., ICLR 2021. https://arxiv.org/abs/2010.01809

Key components:
  - 3 experts (each is a classifier head)
  - Diversity loss: forces experts to disagree on hard samples
  - Routing: learned gating for expert selection
  - Distribution-aware: Each expert specializes via different loss weighting

Simplified for our paper: 3 experts trained jointly with diversity loss + standard CE
+ a small routing network. End-to-end training.

Usage:
    python -m experiments.run_ride --dataset mll --backbone swin_t
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
    p.add_argument("--num_experts", type=int, default=3)
    p.add_argument("--diversity_lambda", type=float, default=0.5,
                   help="Weight of diversity loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def make_run_name(args):
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append("ride")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


class RIDEModel(nn.Module):
    """RIDE: 3 experts (classifier heads) + 1 router."""

    def __init__(self, backbone, num_classes, num_experts=3):
        super().__init__()
        self.backbone = backbone
        self.num_experts = num_experts
        feat_dim = backbone.feat_dim
        self.experts = nn.ModuleList([
            ClassifierHead(in_dim=feat_dim, num_classes=num_classes) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_experts),
        )

    def forward(self, x):
        feats = self.backbone(x)
        expert_logits = torch.stack([e(feats) for e in self.experts], dim=1)  # (B, E, K)
        gate = F.softmax(self.router(feats), dim=1)  # (B, E)
        ensemble = (gate.unsqueeze(-1) * F.softmax(expert_logits, dim=2)).sum(dim=1)
        return expert_logits, ensemble  # raw logits per expert + ensemble probs


def diversity_loss(expert_logits):
    """RIDE diversity loss: KL divergence between expert prediction distributions."""
    probs = F.softmax(expert_logits, dim=2)  # (B, E, K)
    E = probs.size(1)
    if E == 1:
        return torch.tensor(0.0, device=probs.device)

    # Average KL divergence between all pairs of experts (negative -> we want them to disagree)
    div = 0.0
    count = 0
    for i in range(E):
        for j in range(i + 1, E):
            # KL(p_i || p_j) + KL(p_j || p_i)
            kl_ij = (probs[:, i] * (torch.log(probs[:, i] + 1e-8) - torch.log(probs[:, j] + 1e-8))).sum(-1).mean()
            kl_ji = (probs[:, j] * (torch.log(probs[:, j] + 1e-8) - torch.log(probs[:, i] + 1e-8))).sum(-1).mean()
            div += (kl_ij + kl_ji) / 2
            count += 1
    return -div / count  # negative because we minimize loss but want max diversity


def evaluate(model, loader, class_names, priors, device, tag=""):
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, ensemble = model(imgs)
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
    logger.info(f"  RIDE - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)

    # Model
    logger.info("[2] Building RIDE model...")
    backbone = build_backbone(args.backbone)
    model = RIDEModel(backbone, data["num_classes"], args.num_experts).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.experts.parameters(), "lr": args.head_lr},
        {"params": model.router.parameters(), "lr": args.head_lr},
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

    # Training
    logger.info("[3] Training...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        correct = total = 0; ce_sum = div_sum = 0.0
        pbar = tqdm(data["train_loader"], desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            expert_logits, ensemble = model(imgs)

            # CE on each expert
            ce = sum(F.cross_entropy(expert_logits[:, i], labels) for i in range(args.num_experts)) / args.num_experts
            # Diversity loss
            div = diversity_loss(expert_logits)
            loss = ce + args.diversity_lambda * div

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ce_sum += ce.item() * imgs.size(0)
            div_sum += div.item() * imgs.size(0)
            correct += (ensemble.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(ce=f"{ce.item():.3f}", div=f"{div.item():.3f}",
                             acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = evaluate(model, data["val_loader"], data["class_names"],
                               data["priors"], device, tag=f"E{epoch} val")
        history.append(epoch, train_acc=correct/total,
                       train_ce=ce_sum/total, train_div=div_sum/total,
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

    # Test
    if early.best_state is not None:
        model.load_state_dict(early.best_state)
    test_metrics = evaluate(model, data["test_loader"], data["class_names"],
                            data["priors"], device, tag="Test")

    # Save
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

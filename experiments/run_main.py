"""
Main method: Competitive Expert Bidding Game (Ours).

Three phases:
  Phase 1: Train backbone + head with CE (early stopping)
  Phase 2: Freeze backbone, pre-train 3 expert heads (CE / WCE+balanced / Logit Adjustment)
  Phase 3: Freeze backbone + experts, train 3 bidding networks with game loss

All phases save resume checkpoints, full per-epoch logs, training curves.

Usage:
    python -m experiments.run_main --dataset mll --backbone swin_t
    python -m experiments.run_main --dataset pbc --variant lt --imb_factor 100
    python -m experiments.run_main --resume  # continue from last checkpoint
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
from common.models import build_backbone, ClassifierHead, BiddingNetwork, PostHocLAHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])

    # Phase 1: backbone training
    p.add_argument("--p1_max_epochs", type=int, default=50)
    p.add_argument("--p1_patience", type=int, default=10)
    p.add_argument("--p1_batch", type=int, default=32)
    p.add_argument("--p1_backbone_lr", type=float, default=1e-5)
    p.add_argument("--p1_head_lr", type=float, default=1e-3)

    # Phase 2: expert pre-training (NEW: cRT-style, derived from P1)
    # Expert A = P1 head (no retraining)
    # Expert B = P1 head + cRT fine-tuning on balanced data
    # Expert C = P1 head + post-hoc Logit Adjustment (no training)
    p.add_argument("--crt_epochs", type=int, default=10,
                   help="cRT epochs for Expert B (small, since starting from P1)")
    p.add_argument("--crt_patience", type=int, default=4)
    p.add_argument("--crt_lr", type=float, default=1e-4,
                   help="cRT lr (small, only perturbing P1 head)")
    p.add_argument("--p2_batch", type=int, default=64)

    # Phase 3: bidding game
    p.add_argument("--p3_max_epochs", type=int, default=30)
    p.add_argument("--p3_patience", type=int, default=7)
    p.add_argument("--p3_batch", type=int, default=64)
    p.add_argument("--p3_lr", type=float, default=5e-4)
    p.add_argument("--p3_alpha", type=float, default=0.5, help="Game loss weight")
    p.add_argument("--expert_costs", type=str, default="0.1,0.5,1.0",
                   help="Comma-separated costs for 3 experts (used when not learnable)")
    p.add_argument("--learnable_costs", action="store_true",
                   help="Learn expert costs as parameters (with softplus to keep positive)")
    p.add_argument("--cost_init", type=str, default="0.1,0.5,1.0",
                   help="Initial values for learnable costs (only used with --learnable_costs)")

    # Phase 2 hyperparameters
    p.add_argument("--la_tau", type=float, default=1.0, help="Logit adjustment tau (Expert C)")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")

    # Skip phases (for partial reruns)
    p.add_argument("--skip_p1", action="store_true", help="Reuse existing P1 checkpoint")
    p.add_argument("--skip_p2", action="store_true")
    return p.parse_args()


def make_run_name(args):
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append("game")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


# ===================== Evaluation =====================
def eval_single(backbone, head, loader, class_names, priors, device, tag=""):
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


def eval_game(backbone, experts, bidders, loader, class_names, priors, device, tag=""):
    backbone.eval()
    for e in experts: e.eval()
    for b in bidders: b.eval()
    preds_all, trues_all, all_bids = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            bids = torch.stack([b(feats) for b in bidders], dim=1)  # (B, 3)
            weights = F.softmax(bids, dim=1)
            logits = torch.stack([e(feats) for e in experts], dim=1)  # (B, 3, K)
            probs = F.softmax(logits, dim=2)
            ensemble = (weights.unsqueeze(-1) * probs).sum(dim=1)
            preds_all.append(ensemble.argmax(1).cpu())
            trues_all.append(labels)
            all_bids.append(bids.cpu())
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    metrics = compute_lt_metrics(preds, trues, class_names, priors, tag=tag)
    metrics["bids"] = torch.cat(all_bids).numpy()
    metrics["labels"] = trues
    return metrics


# ===================== Phase 1 =====================
def phase1(args, run_name, backbone, head, train_loader, val_loader,
           class_names, priors, device, logger):
    logger.info("\n" + "=" * 70)
    logger.info(f"  PHASE 1: Backbone Training (max {args.p1_max_epochs} epochs, patience {args.p1_patience})")
    logger.info("=" * 70)

    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.p1_backbone_lr},
        {"params": head.parameters(), "lr": args.p1_head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.p1_max_epochs, eta_min=1e-7)
    ce_fn = nn.CrossEntropyLoss()

    history = History()
    early = EarlyStopping(patience=args.p1_patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_p1_resume.pth")
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
            logger.info(f"  [P1 RESUME] from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.p1_max_epochs + 1):
        backbone.train(); head.train()
        train_correct = train_total = 0; train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"  P1 E{epoch}/{args.p1_max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = head(backbone(imgs))
            loss = ce_fn(logits, labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*train_correct/train_total:.1f}%")
        scheduler.step()

        val_metrics = eval_single(backbone, head, val_loader, class_names, priors, device,
                                  tag=f"P1 E{epoch} val")
        history.append(epoch,
                       train_loss=train_loss / train_total,
                       train_acc=train_correct / train_total,
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={"backbone": copy.deepcopy(backbone.state_dict()),
                                    "head": copy.deepcopy(head.state_dict())})
        if is_best:
            logger.info(f"  P1 E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  P1 E{epoch} (patience {early.counter}/{args.p1_patience})")

        save_resume_checkpoint(resume_path, epoch=epoch,
                               backbone=backbone.state_dict(), head=head.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               history=history.records,
                               best_f1=early.best_value, best_epoch=early.best_epoch,
                               patience_counter=early.counter, best_state=early.best_state)

        if early.should_stop:
            logger.info(f"  P1 Early stopping at E{epoch} (best={early.best_value:.4f} @ E{early.best_epoch})")
            break

    # Restore best
    if early.best_state is not None:
        backbone.load_state_dict(early.best_state["backbone"])
        head.load_state_dict(early.best_state["head"])

    # Save final P1
    torch.save(backbone.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_backbone.pth"))
    torch.save(head.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_p1_head.pth"))

    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "p1_history.json"))
    history.plot_curves(os.path.join(save_dir, "p1_curves.png"))

    return backbone, head


# ===================== Phase 2 =====================
def train_crt_expert(args, expert_name, backbone, p1_head, balanced_loader, val_loader,
                     class_names, priors, device, logger, run_name, num_classes):
    """cRT (classifier Re-Training): init from P1, fine-tune classifier on balanced data.

    Reference: Kang et al. ICLR 2020 - Decoupling Representation and Classifier
    for Long-Tailed Recognition.

    Key differences from training-from-scratch:
      - Initialize from P1's classifier head (already strong)
      - Only a few epochs (crt_epochs, default 10)
      - Smaller learning rate (crt_lr = 1e-4, 10x smaller than P1's head_lr)
      - Use balanced sampler + weighted CE
    """
    head = copy.deepcopy(p1_head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.crt_lr,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.crt_epochs, eta_min=1e-6)

    # cRT (Kang et al. ICLR 2020) uses balanced sampling + STANDARD CE.
    # Adding weighted CE on top would double-penalize head classes (the
    # balanced sampler already gives rare classes equal chance to appear).
    # Using weighted CE here previously caused Macro F1 to drop from 0.7458 to 0.5933.
    crt_loss_fn = nn.CrossEntropyLoss()

    history = History()
    early = EarlyStopping(patience=args.crt_patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_p2_{expert_name}_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [{expert_name} RESUME] from epoch {ckpt['epoch']}")

    # Initial val (P1 head's performance, before any cRT)
    if start_epoch == 1:
        init_val = eval_single(backbone, head, val_loader, class_names, priors, device,
                               tag=f"{expert_name} init (=P1)")
        logger.info(f"  {expert_name} init macro_f1={init_val['macro_f1']:.4f}")

    for epoch in range(start_epoch, args.crt_epochs + 1):
        head.train()
        correct = total = 0; loss_sum = 0.0
        pbar = tqdm(balanced_loader, desc=f"  {expert_name} E{epoch}/{args.crt_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = head(feats)
            loss = crt_loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = eval_single(backbone, head, val_loader, class_names, priors, device,
                                  tag=f"{expert_name} E{epoch} val")
        history.append(epoch, train_loss=loss_sum/total, train_acc=correct/total,
                       val_acc=val_metrics["acc"], val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"], val_head_f1=val_metrics["head_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state=copy.deepcopy(head.state_dict()))
        if is_best:
            logger.info(f"  {expert_name} E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  {expert_name} E{epoch} (patience {early.counter}/{args.crt_patience})")

        save_resume_checkpoint(resume_path, epoch=epoch, head=head.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               history=history.records,
                               best_f1=early.best_value, best_epoch=early.best_epoch,
                               patience_counter=early.counter, best_state=early.best_state)
        if early.should_stop:
            logger.info(f"  {expert_name} Early stopping at E{epoch}")
            break

    if early.best_state is not None:
        head.load_state_dict(early.best_state)

    save_dir = os.path.join(RESULTS_DIR, run_name)
    history.save_json(os.path.join(save_dir, f"p2_{expert_name}_history.json"))
    history.plot_curves(os.path.join(save_dir, f"p2_{expert_name}_curves.png"))
    return head


def train_one_expert(args, expert_name, backbone, loader, val_loader, loss_fn,
                     class_names, priors, device, logger, run_name, num_classes):
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.p2_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.p2_max_epochs, eta_min=1e-6)

    history = History()
    early = EarlyStopping(patience=args.p2_patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_p2_{expert_name}_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [{expert_name} RESUME] from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.p2_max_epochs + 1):
        head.train()
        correct = total = 0; loss_sum = 0.0
        pbar = tqdm(loader, desc=f"  {expert_name} E{epoch}/{args.p2_max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = head(feats)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = eval_single(backbone, head, val_loader, class_names, priors, device,
                                  tag=f"{expert_name} E{epoch} val")
        history.append(epoch, train_loss=loss_sum/total, train_acc=correct/total,
                       val_acc=val_metrics["acc"], val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"], val_head_f1=val_metrics["head_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state=copy.deepcopy(head.state_dict()))
        if is_best:
            logger.info(f"  {expert_name} E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  {expert_name} E{epoch} (patience {early.counter}/{args.p2_patience})")

        save_resume_checkpoint(resume_path, epoch=epoch, head=head.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               history=history.records,
                               best_f1=early.best_value, best_epoch=early.best_epoch,
                               patience_counter=early.counter, best_state=early.best_state)
        if early.should_stop:
            logger.info(f"  {expert_name} Early stopping at E{epoch}")
            break

    head.load_state_dict(early.best_state)
    save_dir = os.path.join(RESULTS_DIR, run_name)
    history.save_json(os.path.join(save_dir, f"p2_{expert_name}_history.json"))
    history.plot_curves(os.path.join(save_dir, f"p2_{expert_name}_curves.png"))
    return head


def phase2(args, run_name, backbone, p1_head, balanced_loader, val_loader,
           class_names, priors, device, logger, num_classes):
    """New Phase 2: experts derived from P1 (decoupled training paradigm).

    - Expert A: P1 head (no retraining) - head specialist baseline
    - Expert B: cRT - P1 head fine-tuned on balanced data with weighted CE - tail boost
    - Expert C: Post-hoc Logit Adjustment - P1 head + tau*log(pi) shift at inference - rare class boost
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"  PHASE 2: Decoupled Expert Construction")
    logger.info(f"    Expert A = P1 head (no retraining)")
    logger.info(f"    Expert B = cRT ({args.crt_epochs} epochs, lr={args.crt_lr})")
    logger.info(f"    Expert C = Post-hoc LA (tau={args.la_tau}, no training)")
    logger.info("=" * 70)

    # Freeze backbone
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False

    priors_dev = priors.to(device)
    log_priors = torch.log(priors_dev + 1e-8)

    # ========== Expert A: P1 head (no retraining) ==========
    logger.info("\n  --- Expert A: P1 head (cloned, no retraining) ---")
    expert_a = copy.deepcopy(p1_head)
    expert_a.eval()
    for p in expert_a.parameters():
        p.requires_grad = False

    # ========== Expert B: cRT fine-tuning ==========
    logger.info("\n  --- Expert B: cRT (Decoupled Training, ICLR 2020) ---")
    expert_b = train_crt_expert(args, "ExpertB_cRT", backbone, p1_head,
                                 balanced_loader, val_loader,
                                 class_names, priors, device, logger, run_name, num_classes)
    expert_b.eval()
    for p in expert_b.parameters():
        p.requires_grad = False

    # ========== Expert C: Post-hoc Logit Adjustment ==========
    logger.info("\n  --- Expert C: Post-hoc Logit Adjustment (no training) ---")
    expert_c = PostHocLAHead(copy.deepcopy(p1_head), log_priors, tau=args.la_tau).to(device)
    expert_c.eval()
    for p in expert_c.parameters():
        p.requires_grad = False

    # Save final experts (Expert C saves both base_head + la_shift buffer)
    torch.save(expert_a.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_0.pth"))
    torch.save(expert_b.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_1.pth"))
    torch.save(expert_c.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_2.pth"))

    return [expert_a, expert_b, expert_c]


# ===================== Phase 3 =====================
def phase3(args, run_name, backbone, experts, train_loader, val_loader,
           class_names, priors, device, logger):
    logger.info("\n" + "=" * 70)
    logger.info(f"  PHASE 3: Bidding Game (max {args.p3_max_epochs} epochs)")
    logger.info(f"  alpha: {args.p3_alpha}, learnable_costs: {args.learnable_costs}")
    logger.info("=" * 70)

    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False
    for e in experts:
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    bidders = [BiddingNetwork(in_dim=backbone.feat_dim).to(device) for _ in range(3)]

    # Build cost vector (fixed or learnable)
    if args.learnable_costs:
        cost_init = [float(c) for c in args.cost_init.split(",")]
        # Use inverse-softplus init so softplus(raw) = cost_init
        raw_init = [torch.log(torch.expm1(torch.tensor(c))).item() for c in cost_init]
        cost_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32, device=device))
        get_costs = lambda: F.softplus(cost_raw)
        logger.info(f"  Learnable costs init: {cost_init} (raw: {raw_init})")
    else:
        fixed_costs = torch.tensor([float(c) for c in args.expert_costs.split(",")],
                                    dtype=torch.float32, device=device)
        cost_raw = None
        get_costs = lambda: fixed_costs
        logger.info(f"  Fixed expert costs: {args.expert_costs}")

    all_params = [p for b in bidders for p in b.parameters()]
    if args.learnable_costs:
        all_params = all_params + [cost_raw]
    optimizer = torch.optim.AdamW(all_params, lr=args.p3_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.p3_max_epochs, eta_min=1e-6)

    history = History()
    early = EarlyStopping(patience=args.p3_patience, mode="max")
    start_epoch = 1

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_p3_resume.pth")
    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            for i, b in enumerate(bidders):
                b.load_state_dict(ckpt[f"bidder_{i}"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            if cost_raw is not None and ckpt.get("cost_raw") is not None:
                cost_raw.data.copy_(ckpt["cost_raw"].to(device))
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [P3 RESUME] from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.p3_max_epochs + 1):
        for b in bidders: b.train()
        correct = total = 0; epoch_ce = epoch_game = 0.0

        pbar = tqdm(train_loader, desc=f"  P3 E{epoch}/{args.p3_max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
                expert_logits = torch.stack([e(feats) for e in experts], dim=1)
                expert_probs = F.softmax(expert_logits, dim=2)

            bids = torch.stack([b(feats) for b in bidders], dim=1)
            weights = F.softmax(bids, dim=1)
            ensemble_probs = (weights.unsqueeze(-1) * expert_probs).sum(dim=1)

            ensemble_ce = F.nll_loss(torch.log(ensemble_probs + 1e-8), labels)

            rewards = expert_probs.gather(2, labels.unsqueeze(1).unsqueeze(2).expand(-1, 3, -1)).squeeze(2)
            current_costs = get_costs()  # (3,)
            bid_costs = current_costs.unsqueeze(0) * bids.pow(2)
            utilities = weights * rewards - bid_costs
            game_loss = -utilities.sum(dim=1).mean()

            loss = ensemble_ce + args.p3_alpha * game_loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            correct += (ensemble_probs.argmax(1) == labels).sum().item()
            total += len(labels)
            epoch_ce += ensemble_ce.item(); epoch_game += game_loss.item()
            pbar.set_postfix(ce=f"{ensemble_ce.item():.3f}", game=f"{game_loss.item():.3f}",
                             acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = eval_game(backbone, experts, bidders, val_loader, class_names, priors, device,
                                tag=f"P3 E{epoch} val")

        # Log current costs (especially useful for learnable mode)
        current_costs_np = get_costs().detach().cpu().numpy().tolist()
        history_kwargs = dict(
            train_acc=correct/total,
            avg_ce=epoch_ce/len(train_loader),
            avg_game=epoch_game/len(train_loader),
            val_acc=val_metrics["acc"],
            val_macro_f1=val_metrics["macro_f1"],
            val_tail_f1=val_metrics["tail_f1"],
            val_head_f1=val_metrics["head_f1"],
            costs=current_costs_np,
        )
        history.append(epoch, **history_kwargs)
        if args.learnable_costs:
            logger.info(f"  P3 E{epoch} costs: {[f'{c:.3f}' for c in current_costs_np]}")

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={
                                 "bidders": [copy.deepcopy(b.state_dict()) for b in bidders],
                                 "cost_raw": cost_raw.detach().clone() if cost_raw is not None else None,
                             })
        if is_best:
            logger.info(f"  P3 E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  P3 E{epoch} (patience {early.counter}/{args.p3_patience})")

        ckpt_data = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history.records,
            "best_f1": early.best_value, "best_epoch": early.best_epoch,
            "patience_counter": early.counter, "best_state": early.best_state,
            "cost_raw": cost_raw.detach().cpu() if cost_raw is not None else None,
        }
        for i, b in enumerate(bidders):
            ckpt_data[f"bidder_{i}"] = b.state_dict()
        save_resume_checkpoint(resume_path, **ckpt_data)

        if early.should_stop:
            logger.info(f"  P3 Early stopping at E{epoch}")
            break

    # Restore best state
    if early.best_state is not None:
        for i, b in enumerate(bidders):
            b.load_state_dict(early.best_state["bidders"][i])
        if cost_raw is not None and early.best_state.get("cost_raw") is not None:
            cost_raw.data.copy_(early.best_state["cost_raw"])

    for i, b in enumerate(bidders):
        torch.save(b.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_bidder_{i}.pth"))

    # Save final costs for analysis
    final_costs = get_costs().detach().cpu().numpy().tolist()
    logger.info(f"  P3 Final costs: {[f'{c:.4f}' for c in final_costs]}")
    if cost_raw is not None:
        torch.save({"cost_raw": cost_raw.detach().cpu(), "costs": final_costs},
                   os.path.join(CHECKPOINT_DIR, f"{run_name}_costs.pth"))

    save_dir = os.path.join(RESULTS_DIR, run_name)
    history.save_json(os.path.join(save_dir, "p3_history.json"))
    history.plot_curves(os.path.join(save_dir, "p3_curves.png"),
                        metrics=("train_acc", "val_acc", "val_macro_f1", "val_tail_f1"))

    return bidders


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  Main Method (Bidding Game) - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.p1_batch, args.seed, args.num_workers)

    # Backbone
    backbone = build_backbone(args.backbone).to(device)

    # Phase 1
    p1_head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=data["num_classes"]).to(device)
    if args.skip_p1:
        bb_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_backbone.pth")
        head_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_p1_head.pth")
        backbone.load_state_dict(torch.load(bb_path, map_location=device, weights_only=True))
        p1_head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
        logger.info(f"  [SKIP P1] Loaded backbone+head from {bb_path}")
    else:
        backbone, p1_head = phase1(args, run_name, backbone, p1_head,
                                    data["train_loader"], data["val_loader"],
                                    data["class_names"], data["priors"], device, logger)

    p1_test = eval_single(backbone, p1_head, data["test_loader"],
                          data["class_names"], data["priors"], device, tag="P1 Test")

    # Phase 2
    if args.skip_p2:
        # Load all 3 experts (Expert C is PostHocLAHead, others are ClassifierHead)
        log_priors_dev = torch.log(data["priors"].to(device) + 1e-8)

        # Expert A: ClassifierHead
        expert_a = ClassifierHead(in_dim=backbone.feat_dim, num_classes=data["num_classes"]).to(device)
        expert_a.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_0.pth"),
            map_location=device, weights_only=True))

        # Expert B: ClassifierHead
        expert_b = ClassifierHead(in_dim=backbone.feat_dim, num_classes=data["num_classes"]).to(device)
        expert_b.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_1.pth"),
            map_location=device, weights_only=True))

        # Expert C: PostHocLAHead wrapping a ClassifierHead
        c_base = ClassifierHead(in_dim=backbone.feat_dim, num_classes=data["num_classes"]).to(device)
        expert_c = PostHocLAHead(c_base, log_priors_dev, tau=args.la_tau).to(device)
        expert_c.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, f"{run_name}_expert_2.pth"),
            map_location=device, weights_only=True))

        for e in (expert_a, expert_b, expert_c):
            e.eval()
            for p in e.parameters():
                p.requires_grad = False
        experts = [expert_a, expert_b, expert_c]
        logger.info(f"  [SKIP P2] Loaded 3 experts (A=P1, B=cRT, C=PostHocLA)")
    else:
        # P2 uses larger batch
        data_p2 = build_dataset(args.dataset, args.variant, args.imb_factor,
                                args.p2_batch, args.seed, args.num_workers)
        experts = phase2(args, run_name, backbone, p1_head,
                         data_p2["balanced_loader"], data_p2["val_loader"],
                         data["class_names"], data["priors"], device, logger, data["num_classes"])

    # Test each expert
    expert_tests = []
    for i, e in enumerate(experts):
        m = eval_single(backbone, e, data["test_loader"], data["class_names"],
                        data["priors"], device, tag=f"Expert {chr(65+i)} Test")
        expert_tests.append(m)

    # Phase 3
    data_p3 = build_dataset(args.dataset, args.variant, args.imb_factor,
                            args.p3_batch, args.seed, args.num_workers)
    bidders = phase3(args, run_name, backbone, experts,
                     data_p3["train_loader"], data_p3["val_loader"],
                     data["class_names"], data["priors"], device, logger)
    game_test = eval_game(backbone, experts, bidders, data["test_loader"],
                          data["class_names"], data["priors"], device, tag="Game Test")

    # Save final results
    save_dir = os.path.join(RESULTS_DIR, run_name)
    summary = {
        "run_name": run_name,
        "args": vars(args),
        "p1_test": {k: v for k, v in p1_test.items() if k != "report"},
        "expert_a_test": {k: v for k, v in expert_tests[0].items() if k != "report"},
        "expert_b_test": {k: v for k, v in expert_tests[1].items() if k != "report"},
        "expert_c_test": {k: v for k, v in expert_tests[2].items() if k != "report"},
        "game_test": {k: v for k, v in game_test.items() if k not in ["report", "bids", "labels"]},
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"{'Method':<20} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} {'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for name, m in [("P1 Baseline", p1_test),
                         ("Expert A", expert_tests[0]),
                         ("Expert B", expert_tests[1]),
                         ("Expert C", expert_tests[2]),
                         ("Bidding Game", game_test)]:
            f.write(f"{name:<20} {m['acc']:>8.4f} {m['macro_f1']:>10.4f} "
                    f"{m['weighted_f1']:>10.4f} {m['head_f1']:>10.4f} {m['tail_f1']:>10.4f}\n")
        for name, m in [("P1 Baseline", p1_test),
                         ("Expert A", expert_tests[0]),
                         ("Expert B", expert_tests[1]),
                         ("Expert C", expert_tests[2]),
                         ("Bidding Game", game_test)]:
            f.write(f"\n--- {name} ---\n" + m["report"])

    # Save bid analysis data
    import numpy as np
    np.savez(os.path.join(save_dir, "bid_analysis.npz"),
             bids=game_test["bids"], labels=game_test["labels"],
             priors=data["priors"].numpy(), class_names=data["class_names"])

    logger.info(f"\n[DONE] Results saved to {save_dir}")
    logger.info(f"  Bidding Game: Acc={game_test['acc']:.4f}, Macro F1={game_test['macro_f1']:.4f}, "
                f"Tail F1={game_test['tail_f1']:.4f}")


if __name__ == "__main__":
    main()

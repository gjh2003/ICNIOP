"""
Selective Expert Consultation (SEC).

Two-tier game:

  Tier 1 — Default routing:
      The primary expert (CE) handles the input. Compute its confidence.
      If confident enough, answer directly.

  Tier 2 — Specialist consultation (only triggered when Tier 1 is uncertain):
      Three specialists (CE, WCE, LDAM-DRW) compete in an asymmetric-cost
      bidding game. The winner of the auction (weighted soft mixture) gives
      the final answer.

  Final prediction:
      p_final = (1 - gate) * p_primary + gate * p_consult
      where gate = sigmoid(alpha * (threshold - confidence))

Why this should outperform a flat ensemble:

  - On confident samples (~70% of MLL test): gate ≈ 0, output = CE alone.
    No noise injection, performance == CE on these samples.
  - On uncertain samples (~30%): gate ≈ 1, output = bidding ensemble.
    Specialists get a chance to fix CE's mistakes.
  - Worst case: gate stays at 0 → identical to CE.
  - Best case: gate fires exactly when ensemble helps → strict improvement.

Trainable parameters (frozen experts):
  - 3 BiddingNetwork (Tier 2 auction)
  - threshold (Tier 1 → Tier 2 trigger)
  - alpha       (sharpness of the gate)
  - 3 expert costs (optional, --learnable_costs)

Total trainable params: a few thousand.

Usage:
    python -m experiments.run_sec
    python -m experiments.run_sec --gate_init_threshold 0.6
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple

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
from common.models import build_backbone, ClassifierHead, BiddingNetwork


# ===================== Argparse =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])

    # Frozen baseline experts
    p.add_argument("--expert_a_run", default="mll_swin_t_ce",
                   help="Primary expert (typically the strongest baseline)")
    p.add_argument("--expert_b_run", default="mll_swin_t_weighted_ce")
    p.add_argument("--expert_c_run", default="mll_swin_t_ldam_drw")

    # Gate parameters
    p.add_argument("--gate_init_threshold", type=float, default=0.7,
                   help="Initial confidence threshold; below it, consultation triggers")
    p.add_argument("--gate_init_alpha", type=float, default=10.0,
                   help="Initial sharpness of the sigmoid gate")
    p.add_argument("--gate_reg", type=float, default=0.0,
                   help="L1 penalty on gate activation; default 0 to avoid collapsing the gate")
    p.add_argument("--gate_min", type=float, default=0.2,
                   help="Lower bound for learned threshold (prevents gate→0 collapse)")
    p.add_argument("--gate_max", type=float, default=0.85,
                   help="Upper bound for learned threshold")

    # Bidding game
    p.add_argument("--game_alpha", type=float, default=0.1,
                   help="Weight of game loss (utility regularization)")
    p.add_argument("--expert_costs", type=str, default="0.1,0.5,1.0")
    p.add_argument("--learnable_costs", action="store_true")
    p.add_argument("--bidder_hidden", type=int, default=64,
                   help="Hidden dim of bidder MLP (smaller = less overfit risk)")

    # Training
    p.add_argument("--max_epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3,
                   help="Higher weight decay to prevent bidder overfitting on val")

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
    parts.append("sec")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


# ===================== Confidence Gate =====================
class ConfidenceGate(nn.Module):
    """Two learnable parameters that decide when to invoke consultation.

    gate(c) = sigmoid(alpha * (threshold - c))
        c ∈ [0, 1] is the primary expert's confidence (max prob).
        - When c >> threshold: gate → 0 (primary handles it).
        - When c << threshold: gate → 1 (consultation triggers).

    Threshold is BOUNDED in [t_min, t_max] (default [0.2, 0.85]) to prevent the
    optimizer from finding the trivial collapsing solution threshold→0
    (gate always 0, output ≡ CE).

    Reparameterizations:
        threshold = t_min + (t_max - t_min) * sigmoid(raw_threshold)
        alpha     = softplus(raw_alpha)     > 0
    """

    def __init__(self, init_threshold: float = 0.7, init_alpha: float = 10.0,
                 t_min: float = 0.2, t_max: float = 0.85):
        super().__init__()
        assert t_min < init_threshold < t_max, \
            f"init_threshold={init_threshold} must be in ({t_min}, {t_max})"
        self.t_min = t_min
        self.t_max = t_max
        # Inverse of: t_min + (t_max - t_min)*sigmoid(raw) = init_threshold
        rel = (init_threshold - t_min) / (t_max - t_min)
        raw_t = float(np.log(rel / (1 - rel)))
        raw_a = float(np.log(np.expm1(init_alpha)))
        self.raw_threshold = nn.Parameter(torch.tensor(raw_t, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_a, dtype=torch.float32))

    def get_threshold(self):
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(self.raw_threshold)

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        threshold = self.get_threshold()
        alpha = self.get_alpha()
        return torch.sigmoid(alpha * (threshold - confidence))


# ===================== Loading frozen baselines =====================
def load_baseline(name: str, device, num_classes: int, backbone_type: str):
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


# ===================== Forward / Eval =====================
def forward_sec(experts, bidders, gate_module, costs_fn, x):
    """One unified forward: returns p_final, p_primary, p_consult, gate, bids, weights."""
    bb_a, hd_a = experts[0]
    bb_b, hd_b = experts[1]
    bb_c, hd_c = experts[2]

    with torch.no_grad():
        feats_a = bb_a(x)
        logits_a = hd_a(feats_a)
        logits_b = hd_b(bb_b(x))
        logits_c = hd_c(bb_c(x))

    p_a = F.softmax(logits_a, dim=1)
    p_b = F.softmax(logits_b, dim=1)
    p_c = F.softmax(logits_c, dim=1)
    expert_probs = torch.stack([p_a, p_b, p_c], dim=1)  # (B, 3, K)

    # Tier 1: confidence
    confidence = p_a.max(dim=1)[0]            # (B,)
    gate = gate_module(confidence)            # (B,)

    # Tier 2: bidding (only meaningful when gate > 0)
    bids = torch.stack([b(feats_a) for b in bidders], dim=1)  # (B, 3)
    weights = F.softmax(bids, dim=1)
    p_consult = (weights.unsqueeze(-1) * expert_probs).sum(dim=1)  # (B, K)

    # Mix
    g = gate.unsqueeze(-1)  # (B, 1)
    p_final = (1 - g) * p_a + g * p_consult

    return {
        "p_final": p_final,
        "p_primary": p_a,
        "p_consult": p_consult,
        "expert_probs": expert_probs,
        "confidence": confidence,
        "gate": gate,
        "bids": bids,
        "weights": weights,
    }


@torch.no_grad()
def eval_expert(backbone, head, loader, class_names, priors, device, tag=""):
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
def eval_sec(experts, bidders, gate_module, costs_fn, loader, class_names, priors, device, tag=""):
    for bb, hd in experts:
        bb.eval(); hd.eval()
    for b in bidders: b.eval()
    gate_module.eval()
    preds_all, trues_all = [], []
    gate_values = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = forward_sec(experts, bidders, gate_module, costs_fn, imgs)
        preds_all.append(out["p_final"].argmax(1).cpu())
        trues_all.append(labels)
        gate_values.append(out["gate"].cpu())
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
    logger.info(f"  Selective Expert Consultation (SEC) - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)
    train_loader, val_loader, test_loader = data["train_loader"], data["val_loader"], data["test_loader"]
    priors, class_names, num_classes = data["priors"], data["class_names"], data["num_classes"]

    # Frozen baselines
    logger.info("[2] Loading 3 frozen baseline experts...")
    expert_a = load_baseline(args.expert_a_run, device, num_classes, args.backbone)
    expert_b = load_baseline(args.expert_b_run, device, num_classes, args.backbone)
    expert_c = load_baseline(args.expert_c_run, device, num_classes, args.backbone)
    experts: List[Tuple[nn.Module, nn.Module]] = [expert_a, expert_b, expert_c]
    expert_runs = [args.expert_a_run, args.expert_b_run, args.expert_c_run]

    # Sanity: each expert solo on test set
    logger.info("\n[3] Sanity-check each expert on test set...")
    expert_test_metrics = []
    for i, ((bb, hd), name) in enumerate(zip(experts, expert_runs)):
        m = eval_expert(bb, hd, test_loader, class_names, priors, device,
                        tag=f"Expert {chr(65+i)} ({name})")
        expert_test_metrics.append(m)

    # Trainable modules
    feat_dim = expert_a[0].feat_dim
    # Use smaller bidder hidden size to reduce overfitting risk
    bidders = [BiddingNetwork(in_dim=feat_dim, hidden_dim=args.bidder_hidden).to(device)
               for _ in range(3)]
    gate_module = ConfidenceGate(args.gate_init_threshold, args.gate_init_alpha,
                                  t_min=args.gate_min, t_max=args.gate_max).to(device)

    # Costs
    if args.learnable_costs:
        cost_init = [float(c) for c in args.expert_costs.split(",")]
        raw_init = [torch.log(torch.expm1(torch.tensor(c))).item() for c in cost_init]
        cost_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32, device=device))
        get_costs = lambda: F.softplus(cost_raw)
        logger.info(f"  Learnable costs init: {cost_init}")
    else:
        fixed_costs = torch.tensor([float(c) for c in args.expert_costs.split(",")],
                                    dtype=torch.float32, device=device)
        cost_raw = None
        get_costs = lambda: fixed_costs
        logger.info(f"  Fixed costs: {args.expert_costs}")

    logger.info(f"  Gate init: threshold={gate_module.get_threshold().item():.3f}, "
                f"alpha={gate_module.get_alpha().item():.2f}")

    # Trainable params
    all_params = [p for b in bidders for p in b.parameters()]
    all_params += list(gate_module.parameters())
    if cost_raw is not None:
        all_params += [cost_raw]
    n_trainable = sum(p.numel() for p in all_params)
    logger.info(f"  Total trainable params: {n_trainable}")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1
    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")

    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            for i, b in enumerate(bidders):
                b.load_state_dict(ckpt[f"bidder_{i}"])
            gate_module.load_state_dict(ckpt["gate"])
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
            logger.info(f"  [RESUME] from epoch {ckpt['epoch']}")

    # Init eval (gate at init values, bidders random)
    if start_epoch == 1:
        init_val = eval_sec(experts, bidders, gate_module, get_costs,
                            val_loader, class_names, priors, device, tag="Init Val")
        logger.info(f"  Init SEC val Macro F1 = {init_val['macro_f1']:.4f}, "
                    f"mean_gate = {init_val['mean_gate']:.3f}")

    # Training
    logger.info("\n[4] Training (bidders + gate + optionally costs)...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        for b in bidders: b.train()
        gate_module.train()
        correct = total = 0
        epoch_ce = epoch_game = epoch_gate_sum = 0.0

        pbar = tqdm(train_loader, desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            out = forward_sec(experts, bidders, gate_module, get_costs, imgs)

            p_final = out["p_final"]
            ensemble_ce = F.nll_loss(torch.log(p_final + 1e-8), labels)

            # Game loss (only meaningful when gate is active)
            current_costs = get_costs()
            rewards = out["expert_probs"].gather(
                2, labels.unsqueeze(1).unsqueeze(2).expand(-1, 3, -1)
            ).squeeze(2)
            bid_costs = current_costs.unsqueeze(0) * out["bids"].pow(2)
            utilities = out["weights"] * rewards - bid_costs
            game_loss = -utilities.sum(dim=1).mean()

            # Gate sparsity regularization (encourage Tier-1 default)
            gate_reg_term = out["gate"].mean()

            loss = (ensemble_ce
                    + args.game_alpha * game_loss
                    + args.gate_reg * gate_reg_term)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            correct += (p_final.argmax(1) == labels).sum().item()
            total += len(labels)
            epoch_ce += ensemble_ce.item()
            epoch_game += game_loss.item()
            epoch_gate_sum += gate_reg_term.item()

            pbar.set_postfix(ce=f"{ensemble_ce.item():.3f}",
                             game=f"{game_loss.item():.3f}",
                             gate=f"{gate_reg_term.item():.2f}",
                             acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        # Validate
        val_metrics = eval_sec(experts, bidders, gate_module, get_costs,
                                val_loader, class_names, priors, device,
                                tag=f"E{epoch} val")

        # Snapshot of learned parameters
        cur_threshold = gate_module.get_threshold().item()
        cur_alpha = gate_module.get_alpha().item()
        cur_costs = get_costs().detach().cpu().numpy().tolist()

        history.append(epoch,
                       train_acc=correct/total,
                       avg_ce=epoch_ce/len(train_loader),
                       avg_game=epoch_game/len(train_loader),
                       mean_gate_train=epoch_gate_sum/len(train_loader),
                       mean_gate_val=val_metrics["mean_gate"],
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"],
                       gate_threshold=cur_threshold,
                       gate_alpha=cur_alpha,
                       costs=cur_costs)

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={
                                 "bidders": [copy.deepcopy(b.state_dict()) for b in bidders],
                                 "gate": copy.deepcopy(gate_module.state_dict()),
                                 "cost_raw": cost_raw.detach().clone() if cost_raw is not None else None,
                             })
        marker = "*best*" if is_best else f"(patience {early.counter}/{args.patience})"
        logger.info(f"  E{epoch} {marker}  macro_f1={val_metrics['macro_f1']:.4f}, "
                    f"tail_f1={val_metrics['tail_f1']:.4f}, "
                    f"thr={cur_threshold:.3f}, alpha={cur_alpha:.1f}, "
                    f"mean_gate={val_metrics['mean_gate']:.3f}")

        ckpt_data = {
            "epoch": epoch,
            "gate": gate_module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history.records,
            "best_f1": early.best_value,
            "best_epoch": early.best_epoch,
            "patience_counter": early.counter,
            "best_state": early.best_state,
            "cost_raw": cost_raw.detach().cpu() if cost_raw is not None else None,
        }
        for i, b in enumerate(bidders):
            ckpt_data[f"bidder_{i}"] = b.state_dict()
        save_resume_checkpoint(resume_path, **ckpt_data)

        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch}")
            break

    elapsed = (time.time() - t_start) / 60

    # Restore best
    if early.best_state is not None:
        for i, b in enumerate(bidders):
            b.load_state_dict(early.best_state["bidders"][i])
        gate_module.load_state_dict(early.best_state["gate"])
        if cost_raw is not None and early.best_state.get("cost_raw") is not None:
            cost_raw.data.copy_(early.best_state["cost_raw"].to(device))

    # Save final modules
    for i, b in enumerate(bidders):
        torch.save(b.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_bidder_{i}.pth"))
    torch.save(gate_module.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_gate.pth"))

    final_threshold = gate_module.get_threshold().item()
    final_alpha = gate_module.get_alpha().item()
    final_costs = get_costs().detach().cpu().numpy().tolist()
    logger.info(f"\n[5] Final gate: threshold={final_threshold:.4f}, alpha={final_alpha:.2f}")
    logger.info(f"  Final costs: {[f'{c:.4f}' for c in final_costs]}")

    # Test
    test_metrics = eval_sec(experts, bidders, gate_module, get_costs,
                             test_loader, class_names, priors, device, tag="Test")

    # Save outputs
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"),
                        metrics=("train_acc", "val_acc", "val_macro_f1",
                                 "val_tail_f1", "mean_gate_val", "gate_threshold"))

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "expert_a_test": {k: v for k, v in expert_test_metrics[0].items() if k != "report"},
        "expert_b_test": {k: v for k, v in expert_test_metrics[1].items() if k != "report"},
        "expert_c_test": {k: v for k, v in expert_test_metrics[2].items() if k != "report"},
        "sec_test": {k: v for k, v in test_metrics.items() if k != "report"},
        "final_gate": {"threshold": final_threshold, "alpha": final_alpha},
        "final_costs": final_costs,
        "training_minutes": elapsed,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Best val macro_f1: {early.best_value:.4f} @ E{early.best_epoch}\n")
        f.write(f"Training time: {elapsed:.1f} min\n")
        f.write(f"Final gate: threshold={final_threshold:.4f}, alpha={final_alpha:.2f}\n")
        f.write(f"Final costs: {final_costs}\n\n")
        f.write(f"{'Method':<30} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} {'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for name, m in [(f"Expert A ({args.expert_a_run})", expert_test_metrics[0]),
                         (f"Expert B ({args.expert_b_run})", expert_test_metrics[1]),
                         (f"Expert C ({args.expert_c_run})", expert_test_metrics[2]),
                         ("SEC (Ours)", test_metrics)]:
            f.write(f"{name:<30} {m['acc']:>8.4f} {m['macro_f1']:>10.4f} "
                    f"{m['weighted_f1']:>10.4f} {m['head_f1']:>10.4f} {m['tail_f1']:>10.4f}\n")
        f.write(f"\nMean gate (test): {test_metrics['mean_gate']:.4f}\n")
        f.write(f"Test Acc:        {test_metrics['acc']:.4f}\n")
        f.write(f"Test Macro F1:   {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Test Wtd F1:     {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Test Head F1:    {test_metrics['head_f1']:.4f}\n")
        f.write(f"Test Tail F1:    {test_metrics['tail_f1']:.4f}\n\n")
        f.write(test_metrics["report"])

    # Final summary line
    a_mf1 = expert_test_metrics[0]["macro_f1"]
    b_mf1 = expert_test_metrics[1]["macro_f1"]
    c_mf1 = expert_test_metrics[2]["macro_f1"]
    sec_mf1 = test_metrics["macro_f1"]
    best_single = max(a_mf1, b_mf1, c_mf1)
    delta = sec_mf1 - best_single

    logger.info(f"\n{'='*70}")
    logger.info(f"  SEC RESULT")
    logger.info(f"{'='*70}")
    logger.info(f"  Expert A (CE):   Macro F1 = {a_mf1:.4f}")
    logger.info(f"  Expert B (WCE):  Macro F1 = {b_mf1:.4f}")
    logger.info(f"  Expert C (LDAM): Macro F1 = {c_mf1:.4f}")
    logger.info(f"  Best single expert:    {best_single:.4f}")
    logger.info(f"  ----------------------------------------------------------------")
    logger.info(f"  SEC (Ours):             {sec_mf1:.4f}")
    logger.info(f"  Delta:                  {delta:+.4f}")
    logger.info(f"  Mean gate (test):       {test_metrics['mean_gate']:.4f}")
    logger.info(f"  Tail F1:                {test_metrics['tail_f1']:.4f}  "
                f"(CE: {expert_test_metrics[0]['tail_f1']:.4f})")
    if delta > 0.005:
        logger.info(f"  ✓ SUCCESS: SEC beats best single by {delta*100:.2f}%")
    else:
        logger.info(f"  ✗ Marginal: gain {delta*100:.2f}%")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

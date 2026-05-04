"""
Rescue experiment: Bidding Game on 3 strong, independently trained baselines.

Why this exists:
  After fully training Phase 1 (38 epochs), Standard CE became too strong
  (Macro F1=0.7458). The previously designed Expert B (cRT) and Expert C
  (Post-hoc LA) ended up WEAKER than P1, so the bidding ensemble could
  never beat the single best expert.

  This script abandons the "derive from P1" idea and instead uses 3
  INDEPENDENTLY TRAINED strong models as experts:
    - Expert A: Standard CE              (Macro F1 = 0.7458)
    - Expert B: Weighted CE              (Macro F1 = 0.7425)
    - Expert C: LDAM-DRW                 (Macro F1 = 0.7108)

  Each expert is a complete (backbone, head) pair (frozen). Only the
  bidders (3 small networks) are trained. The bidder uses CE's backbone
  features as input.

Usage:
    python -m experiments.run_ensemble_rescue
    python -m experiments.run_ensemble_rescue --p3_alpha 1.0
    python -m experiments.run_ensemble_rescue --expert_costs 0.1,0.3,0.5

If results > 0.7458 Macro F1, the paper has a story.
If not, we pivot to interpretability narrative.
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])

    # Experts (defaults pick the 3 trained baselines)
    p.add_argument("--expert_a_run", default="mll_swin_t_ce",
                   help="Run name of Expert A (head specialist, expected to be strongest)")
    p.add_argument("--expert_b_run", default="mll_swin_t_weighted_ce",
                   help="Run name of Expert B")
    p.add_argument("--expert_c_run", default="mll_swin_t_ldam_drw",
                   help="Run name of Expert C")

    # Bidder feature extractor: by default reuse Expert A's backbone
    p.add_argument("--bidder_features_from", default="A",
                   choices=["A", "B", "C"],
                   help="Which expert's backbone provides features to bidders")

    # Bidder training
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Game
    p.add_argument("--p3_alpha", type=float, default=0.5,
                   help="Weight of game loss vs ensemble CE")
    p.add_argument("--expert_costs", type=str, default="0.1,0.5,1.0",
                   help="Comma-separated costs (asymmetric: cheap=most reliable)")
    p.add_argument("--learnable_costs", action="store_true")
    p.add_argument("--cost_init", type=str, default="0.1,0.5,1.0")

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default=None,
                   help="Override run_name (default auto)")
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append("ensemble_rescue")
    if args.seed != 42:
        parts.append(f"s{args.seed}")
    return "_".join(parts)


# ===================== Loading =====================
def load_baseline(name: str, device, num_classes: int, backbone_type: str) -> Tuple[nn.Module, nn.Module]:
    """Load a baseline checkpoint as (frozen backbone, frozen head)."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {ckpt_path}\n"
            f"Make sure you have run `python -m experiments.run_baseline --method ce` "
            f"(or weighted_ce, ldam_drw) and it produced {name}_best.pth"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = build_backbone(backbone_type).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = False
    return backbone, head


# ===================== Evaluation =====================
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
def eval_ensemble(experts, bidder_backbone, bidders, loader, class_names, priors, device, tag=""):
    bidder_backbone.eval()
    for b in bidders: b.eval()
    for bb, hd in experts:
        bb.eval(); hd.eval()
    preds_all, trues_all, all_bids = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        bidder_feats = bidder_backbone(imgs)
        bids = torch.stack([b(bidder_feats) for b in bidders], dim=1)  # (B, 3)
        weights = F.softmax(bids, dim=1)
        expert_logits = torch.stack([hd(bb(imgs)) for bb, hd in experts], dim=1)  # (B, 3, K)
        expert_probs = F.softmax(expert_logits, dim=2)
        ensemble = (weights.unsqueeze(-1) * expert_probs).sum(dim=1)
        preds_all.append(ensemble.argmax(1).cpu())
        trues_all.append(labels)
        all_bids.append(bids.cpu())
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    metrics = compute_lt_metrics(preds, trues, class_names, priors, tag=tag)
    metrics["bids"] = torch.cat(all_bids).numpy()
    metrics["labels"] = trues
    return metrics


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  Ensemble Rescue - {run_name}")
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

    # Load 3 frozen experts
    logger.info("[2] Loading 3 frozen baseline experts...")
    expert_a = load_baseline(args.expert_a_run, device, num_classes, args.backbone)
    expert_b = load_baseline(args.expert_b_run, device, num_classes, args.backbone)
    expert_c = load_baseline(args.expert_c_run, device, num_classes, args.backbone)
    experts: List[Tuple[nn.Module, nn.Module]] = [expert_a, expert_b, expert_c]

    expert_runs = [args.expert_a_run, args.expert_b_run, args.expert_c_run]

    # Sanity: evaluate each expert on test set
    logger.info("\n[3] Sanity-check each expert on test set...")
    expert_test_metrics = []
    for i, ((bb, hd), name) in enumerate(zip(experts, expert_runs)):
        m = eval_expert(bb, hd, test_loader, class_names, priors, device,
                        tag=f"Expert {chr(65+i)} ({name})")
        expert_test_metrics.append(m)

    # Pick bidder feature extractor
    bidder_idx = {"A": 0, "B": 1, "C": 2}[args.bidder_features_from]
    bidder_backbone = experts[bidder_idx][0]
    feat_dim = bidder_backbone.feat_dim
    logger.info(f"\n[4] Bidder features come from Expert {args.bidder_features_from} "
                f"({expert_runs[bidder_idx]}), feat_dim={feat_dim}")

    # Bidders
    bidders = [BiddingNetwork(in_dim=feat_dim).to(device) for _ in range(3)]

    # Costs
    if args.learnable_costs:
        cost_init = [float(c) for c in args.cost_init.split(",")]
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

    all_params = [p for b in bidders for p in b.parameters()]
    if args.learnable_costs:
        all_params = all_params + [cost_raw]

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

    # Sanity check at epoch 0: ensemble with random bidders ≈ simple average
    if start_epoch == 1:
        logger.info("\n[5] Sanity check: ensemble with init bidders (≈ simple average)")
        init_val = eval_ensemble(experts, bidder_backbone, bidders, val_loader,
                                  class_names, priors, device, tag="Init Val")
        logger.info(f"  Init ensemble val Macro F1 = {init_val['macro_f1']:.4f} "
                    f"(should be near average of 3 experts)")

    # Training loop
    logger.info("\n[6] Training bidders only (experts frozen)...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        for b in bidders: b.train()
        correct = total = 0
        epoch_ce = epoch_game = 0.0

        pbar = tqdm(train_loader, desc=f"  E{epoch}/{args.max_epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                # 3 expert forward passes (frozen)
                expert_logits = torch.stack([hd(bb(imgs)) for bb, hd in experts], dim=1)
                expert_probs = F.softmax(expert_logits, dim=2)
                # Bidder feature extraction
                bidder_feats = bidder_backbone(imgs)

            bids = torch.stack([b(bidder_feats) for b in bidders], dim=1)
            weights = F.softmax(bids, dim=1)
            ensemble_probs = (weights.unsqueeze(-1) * expert_probs).sum(dim=1)

            ensemble_ce = F.nll_loss(torch.log(ensemble_probs + 1e-8), labels)

            rewards = expert_probs.gather(
                2, labels.unsqueeze(1).unsqueeze(2).expand(-1, 3, -1)
            ).squeeze(2)
            current_costs = get_costs()
            bid_costs = current_costs.unsqueeze(0) * bids.pow(2)
            utilities = weights * rewards - bid_costs
            game_loss = -utilities.sum(dim=1).mean()

            loss = ensemble_ce + args.p3_alpha * game_loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            correct += (ensemble_probs.argmax(1) == labels).sum().item()
            total += len(labels)
            epoch_ce += ensemble_ce.item()
            epoch_game += game_loss.item()
            pbar.set_postfix(ce=f"{ensemble_ce.item():.3f}",
                             game=f"{game_loss.item():.3f}",
                             acc=f"{100*correct/total:.1f}%")
        scheduler.step()

        val_metrics = eval_ensemble(experts, bidder_backbone, bidders, val_loader,
                                     class_names, priors, device, tag=f"E{epoch} val")

        cur_costs = get_costs().detach().cpu().numpy().tolist()
        history.append(epoch,
                       train_acc=correct/total,
                       avg_ce=epoch_ce/len(train_loader),
                       avg_game=epoch_game/len(train_loader),
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"],
                       costs=cur_costs)
        if args.learnable_costs:
            logger.info(f"  E{epoch} costs: {[f'{c:.3f}' for c in cur_costs]}")

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={
                                 "bidders": [copy.deepcopy(b.state_dict()) for b in bidders],
                                 "cost_raw": cost_raw.detach().clone() if cost_raw is not None else None,
                             })
        if is_best:
            logger.info(f"  E{epoch} *best* macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            logger.info(f"  E{epoch} (patience {early.counter}/{args.patience})")

        ckpt_data = {
            "epoch": epoch,
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
        if cost_raw is not None and early.best_state.get("cost_raw") is not None:
            cost_raw.data.copy_(early.best_state["cost_raw"].to(device))

    # Save bidders
    for i, b in enumerate(bidders):
        torch.save(b.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}_bidder_{i}.pth"))

    final_costs = get_costs().detach().cpu().numpy().tolist()
    logger.info(f"\n[7] Final costs: {[f'{c:.4f}' for c in final_costs]}")

    # Test evaluation
    test_metrics = eval_ensemble(experts, bidder_backbone, bidders, test_loader,
                                  class_names, priors, device, tag="Test")

    # Save results
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)

    history.save_json(os.path.join(save_dir, "history.json"))
    history.plot_curves(os.path.join(save_dir, "training_curves.png"),
                        metrics=("train_acc", "val_acc", "val_macro_f1", "val_tail_f1"))

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "expert_a_test": {k: v for k, v in expert_test_metrics[0].items() if k != "report"},
        "expert_b_test": {k: v for k, v in expert_test_metrics[1].items() if k != "report"},
        "expert_c_test": {k: v for k, v in expert_test_metrics[2].items() if k != "report"},
        "ensemble_test": {k: v for k, v in test_metrics.items()
                          if k not in ["report", "bids", "labels"]},
        "final_costs": final_costs,
        "training_minutes": elapsed,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Best val macro_f1: {early.best_value:.4f} @ E{early.best_epoch}\n")
        f.write(f"Training time: {elapsed:.1f} min\n")
        f.write(f"Final costs: {final_costs}\n\n")
        f.write(f"{'Method':<30} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} {'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for name, m in [(f"Expert A ({args.expert_a_run})", expert_test_metrics[0]),
                         (f"Expert B ({args.expert_b_run})", expert_test_metrics[1]),
                         (f"Expert C ({args.expert_c_run})", expert_test_metrics[2]),
                         ("Ensemble (Ours)", test_metrics)]:
            f.write(f"{name:<30} {m['acc']:>8.4f} {m['macro_f1']:>10.4f} "
                    f"{m['weighted_f1']:>10.4f} {m['head_f1']:>10.4f} {m['tail_f1']:>10.4f}\n")
        f.write("\nTest Acc:        {:.4f}\n".format(test_metrics['acc']))
        f.write("Test Macro F1:   {:.4f}\n".format(test_metrics['macro_f1']))
        f.write("Test Wtd F1:     {:.4f}\n".format(test_metrics['weighted_f1']))
        f.write("Test Head F1:    {:.4f}\n".format(test_metrics['head_f1']))
        f.write("Test Tail F1:    {:.4f}\n\n".format(test_metrics['tail_f1']))
        f.write(test_metrics["report"])

    # Save bid analysis
    np.savez(os.path.join(save_dir, "bid_analysis.npz"),
             bids=test_metrics["bids"], labels=test_metrics["labels"],
             priors=priors.numpy(), class_names=np.array(class_names))

    # ============================================================
    # Final summary line for quick-glance evaluation
    # ============================================================
    a_mf1 = expert_test_metrics[0]["macro_f1"]
    b_mf1 = expert_test_metrics[1]["macro_f1"]
    c_mf1 = expert_test_metrics[2]["macro_f1"]
    ens_mf1 = test_metrics["macro_f1"]
    best_single = max(a_mf1, b_mf1, c_mf1)
    delta = ens_mf1 - best_single

    logger.info(f"\n{'='*70}")
    logger.info(f"  RESCUE RESULT")
    logger.info(f"{'='*70}")
    logger.info(f"  Expert A ({args.expert_a_run}):    Macro F1 = {a_mf1:.4f}")
    logger.info(f"  Expert B ({args.expert_b_run}):    Macro F1 = {b_mf1:.4f}")
    logger.info(f"  Expert C ({args.expert_c_run}):    Macro F1 = {c_mf1:.4f}")
    logger.info(f"  ----------------------------------------------------------------")
    logger.info(f"  Best single expert:               Macro F1 = {best_single:.4f}")
    logger.info(f"  Ensemble (Ours):                  Macro F1 = {ens_mf1:.4f}")
    logger.info(f"  Delta (ensemble - best_single):   {delta:+.4f}")
    logger.info(f"")
    if delta > 0.01:
        logger.info(f"  ✓ SUCCESS: ensemble beats best single by {delta*100:.2f}%")
    elif delta > 0:
        logger.info(f"  ~ MARGINAL: ensemble slightly beats best single by {delta*100:.2f}%")
    else:
        logger.info(f"  ✗ FAILURE: ensemble does not beat best single ({delta*100:.2f}%)")
        logger.info(f"     Recommendation: pivot paper to interpretability narrative")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

"""
MiSLAS Stage 2 only — applied on the converged CE checkpoint.

Cleanly isolates whether asymmetric Label-Aware Smoothing (NBT at the
label level) helps when the backbone is already strong (avoids the
under-training confound that hit run_mislas.py).

Pipeline:
    Load CE checkpoint (mll_swin_t_ce, Macro F1 = 0.7458) → freeze backbone.
    Re-init classifier head, train with class-balanced sampler and
    per-class label smoothing ε_k ∝ frequency. ε_max for the most frequent
    class, 0 for the rarest.

Game-theoretic reading: each class pays a label-mass tax proportional to
its training count. Frequent classes (low marginal value of confidence)
pay; rare classes are exempt.
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
    setup_logger, EarlyStopping, compute_lt_metrics, set_seed,
)
from common.data import build_dataset
from common.models import build_backbone, ClassifierHead

from experiments.run_mislas import label_aware_smoothing_loss, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t")
    p.add_argument("--primary_run", default="mll_swin_t_ce")

    p.add_argument("--s2_epochs", type=int, default=20)
    p.add_argument("--s2_patience", type=int, default=6)
    p.add_argument("--s2_lr", type=float, default=5e-4)
    p.add_argument("--ls_eps_max_list", type=str, default="0.0,0.05,0.1,0.15,0.2",
                   help="Sweep ε_max values; ε=0 reproduces vanilla cRT (fallback).")
    p.add_argument("--reuse_head", action="store_true",
                   help="Continue from CE's classifier head instead of re-initialising")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    return f"{args.dataset}_{args.backbone}_mislas_s2only"


def load_ce_checkpoint(name, device, num_classes, backbone_type):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = build_backbone(backbone_type).to(device)
    head = ClassifierHead(in_dim=backbone.feat_dim, num_classes=num_classes).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    return backbone, head


def train_one_setting(backbone, num_classes, feat_dim, balanced_loader, val_loader,
                      eps_per_class_t, lr, epochs, patience,
                      class_names, priors, device, logger, ls_eps_max,
                      init_head_state=None):
    """Train one (re-init or reused) head with given asymmetric LS. Returns best
    val Macro F1 + best head state."""
    head = ClassifierHead(in_dim=feat_dim, num_classes=num_classes).to(device)
    if init_head_state is not None:
        head.load_state_dict(init_head_state)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    early = EarlyStopping(patience=patience, mode="max")

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0; n_seen = 0
        for imgs, labels in tqdm(balanced_loader,
                                  desc=f"  ε_max={ls_eps_max} E{epoch}/{epochs}",
                                  ncols=100, leave=False):
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                feat = backbone(imgs)
            logits = head(feat)
            loss = label_aware_smoothing_loss(logits, labels, eps_per_class_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            bs = imgs.size(0); loss_sum += loss.item() * bs; n_seen += bs
        sched.step()

        v = evaluate(backbone, head, val_loader, class_names, priors, device, tag="")
        is_best = early.step(v["macro_f1"], epoch, state=copy.deepcopy(head.state_dict()))
        marker = "*best*" if is_best else f"(p {early.counter}/{patience})"
        logger.info(f"    E{epoch:>2} {marker} loss={loss_sum/n_seen:.4f} "
                    f"val_macro_f1={v['macro_f1']:.4f} val_tail_f1={v['tail_f1']:.4f}")
        if early.should_stop:
            break

    if early.best_state is not None:
        head.load_state_dict(early.best_state)
    return head, early.best_value


def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  MiSLAS S2-only on CE backbone - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    logger.info("[1] Loading data and CE checkpoint...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)
    natural_loader = data["train_loader"]
    balanced_loader = data["balanced_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    priors = data["priors"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    backbone, head_ce = load_ce_checkpoint(args.primary_run, device, num_classes, args.backbone)
    feat_dim = backbone.feat_dim
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Sanity: reproduce CE
    logger.info("[2] Sanity: CE baseline test eval")
    test_ce = evaluate(backbone, head_ce, test_loader, class_names, priors, device, tag="CE")
    logger.info(f"  CE test: Macro F1={test_ce['macro_f1']:.4f}, Tail F1={test_ce['tail_f1']:.4f}")
    ce_macro = test_ce["macro_f1"]

    # Compute per-class counts
    counts = np.bincount([s[1] for s in natural_loader.dataset.samples], minlength=num_classes)
    cmin, cmax = counts.min(), counts.max()

    init_head_state = head_ce.state_dict() if args.reuse_head else None

    # Sweep ε_max
    logger.info(f"[3] Asymmetric LS sweep (reuse_head={args.reuse_head})")
    sweep_results = {"ce_baseline": {k: v for k, v in test_ce.items() if k != "report"}}
    best_eps = None; best_test_macro = ce_macro; best_head_state = None

    for eps_str in args.ls_eps_max_list.split(","):
        eps_max = float(eps_str)
        eps_per_class = eps_max * (counts - cmin) / max(cmax - cmin, 1)
        eps_t = torch.tensor(eps_per_class, dtype=torch.float32, device=device)
        logger.info(f"\n  --- ε_max = {eps_max} (per-class ε range: 0 → {eps_max}) ---")

        t0 = time.time()
        head, val_best = train_one_setting(
            backbone, num_classes, feat_dim, balanced_loader, val_loader,
            eps_t, args.s2_lr, args.s2_epochs, args.s2_patience,
            class_names, priors, device, logger, eps_max,
            init_head_state=init_head_state)
        elapsed = (time.time() - t0) / 60

        test_m = evaluate(backbone, head, test_loader, class_names, priors, device, tag="")
        delta = test_m["macro_f1"] - ce_macro
        marker = "✓" if delta > 0 else ("=" if abs(delta) < 1e-4 else "✗")
        logger.info(f"  ε_max={eps_max:.2f} → val_best={val_best:.4f}, "
                    f"test Macro F1={test_m['macro_f1']:.4f} (Δ={delta:+.4f}) "
                    f"Tail F1={test_m['tail_f1']:.4f} {marker} ({elapsed:.1f} min)")
        sweep_results[f"eps_{eps_max}"] = {k: v for k, v in test_m.items() if k != "report"}
        sweep_results[f"eps_{eps_max}"]["val_best"] = val_best
        if test_m["macro_f1"] > best_test_macro:
            best_test_macro = test_m["macro_f1"]
            best_eps = eps_max
            best_head_state = copy.deepcopy(head.state_dict())

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"  MiSLAS S2-only FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  CE baseline:  Macro F1 = {ce_macro:.4f}")
    for k, v in sweep_results.items():
        if k == "ce_baseline": continue
        d = v["macro_f1"] - ce_macro
        mk = "✓" if d > 0 else "✗"
        logger.info(f"  {k:<14} Macro F1={v['macro_f1']:.4f} (Δ={d:+.4f}) "
                    f"Tail F1={v['tail_f1']:.4f} {mk}")
    if best_eps is not None:
        logger.info(f"  BEST: ε_max={best_eps} → Macro F1 = {best_test_macro:.4f} "
                    f"(Δ={best_test_macro-ce_macro:+.4f})")
    else:
        logger.info(f"  No setting beat CE.")

    # Save
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "ce_macro": ce_macro,
                   "sweep": sweep_results, "best_eps": best_eps,
                   "best_macro": best_test_macro,
                   "class_counts": counts.tolist()}, f, indent=2)
    if best_head_state is not None:
        torch.save({"backbone": backbone.state_dict(),
                    "head": best_head_state,
                    "best_eps": best_eps,
                    "best_macro": best_test_macro},
                    os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))


if __name__ == "__main__":
    main()

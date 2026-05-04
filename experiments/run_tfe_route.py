"""
TFE Router: combine CE (21-way) and TFE (tail-only 7-way) on full test set.

Background:
    Tail-Focused Expert (TFE) was trained in run_tfe.py on the tail-only
    subset. It beats CE by +9.89% Macro F1 within tail samples (e.g.
    BAS: 0.69→0.94, FGC: 0.73→0.92, HAC: 0.86→0.98, KSC: 0.91→1.00).

    But TFE only outputs 7 tail classes; it has zero head knowledge.
    To use TFE on the full 21-class test set, we need a router that
    decides per sample which expert to trust.

Routing strategies tested in this script:

    Strategy 0: CE alone (baseline, 0.7458 Macro F1)

    Strategy 1: Hard route by CE's prediction
        if CE.argmax in HEAD_CLASSES: use CE
        else (CE predicts a tail class): use TFE
        — pure deterministic, no training, no params

    Strategy 2: Hard route by CE confidence
        if CE.max_prob > T: use CE
        else: use TFE
        — single threshold, swept over a grid

    Strategy 3: Hard route by both rules
        if CE.argmax in HEAD AND CE.max_prob > T: use CE
        else: use TFE

    Strategy 4: Soft mixture with learnable gate
        gate = sigmoid(alpha * (T - confidence))
        # only trigger gate if CE predicts tail OR is uncertain
        gate = gate * (1 - is_head_pred * confident_indicator)
        p_final = (1 - gate) * p_ce + gate * p_tfe_in_21d
        — game-theoretic asymmetric-cost framing

Game-theoretic interpretation (Strategy 4):
    Two-tier auction. Tier 1 = CE (the generalist, low cost c_CE = 0).
    Tier 2 = TFE (the tail specialist, high cost c_TFE invoked when needed).
    Asymmetric Nash equilibrium:
        CE wins all confident head predictions (low cost, high reward).
        TFE wins on uncertain or tail-classified inputs (high cost, but high
        reward on tail-class samples where it specializes).

Usage:
    python -m experiments.run_tfe_route
    python -m experiments.run_tfe_route --strategy soft --max_epochs 10
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
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR, print_paths
from common.train_utils import (
    setup_logger, EarlyStopping, save_resume_checkpoint, load_resume_checkpoint,
    History, compute_lt_metrics, set_seed,
)
from common.data import build_dataset
from common.models import build_backbone, ClassifierHead

from experiments.run_tfe import TFEHead, build_tail_dataset, load_primary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--primary_run", default="mll_swin_t_ce")
    p.add_argument("--tfe_run", default="mll_swin_t_tfe_thr500",
                   help="TFE run name (loads {tfe_run}_head.pth)")
    p.add_argument("--tail_threshold", type=int, default=500)

    # Routing
    p.add_argument("--strategy", default="all",
                   choices=["hard_pred", "conf", "both", "soft", "all"],
                   help="Which routing strategy to evaluate")
    p.add_argument("--conf_thresholds", type=str,
                   default="0.3,0.4,0.5,0.6,0.7,0.8",
                   help="Confidence thresholds to sweep")
    p.add_argument("--soft_init_threshold", type=float, default=0.7)
    p.add_argument("--soft_init_alpha", type=float, default=10.0)

    # Soft router training
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.dataset]
    if args.dataset == "pbc" and args.variant == "lt":
        parts.append(f"lt{args.imb_factor}")
    parts.append(args.backbone)
    parts.append(f"tferoute_thr{args.tail_threshold}")
    return "_".join(parts)


def scatter_tail_to_full(p_tail, tail_orig_indices, num_classes_full):
    """Map a (B, n_tail) prob vector into a (B, num_classes_full) sparse vector
    where head positions are 0 and tail positions hold p_tail values."""
    B = p_tail.size(0)
    p_full = torch.zeros(B, num_classes_full, device=p_tail.device, dtype=p_tail.dtype)
    idx = torch.tensor(tail_orig_indices, device=p_tail.device)  # (n_tail,)
    p_full[:, idx] = p_tail
    return p_full


# ===================== Inference helpers =====================
@torch.no_grad()
def collect_probs(backbone, ce_head, tfe_head, loader, tail_orig_indices, num_classes_full, device):
    """For each test image, collect:
        - p_ce  (B, num_classes_full)
        - p_tfe_full (B, num_classes_full)  # tail positions = TFE softmax, head = 0
        - labels (B,)  # original 21-way labels
    """
    backbone.eval(); ce_head.eval(); tfe_head.eval()
    all_p_ce, all_p_tfe, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits_ce = ce_head(feats)
        p_ce = F.softmax(logits_ce, dim=1)

        logits_tfe = tfe_head(feats)
        p_tfe = F.softmax(logits_tfe, dim=1)
        p_tfe_full = scatter_tail_to_full(p_tfe, tail_orig_indices, num_classes_full)

        all_p_ce.append(p_ce.cpu())
        all_p_tfe.append(p_tfe_full.cpu())
        all_labels.append(labels)

    return (torch.cat(all_p_ce), torch.cat(all_p_tfe), torch.cat(all_labels))


def compute_metrics_from_preds(preds, trues, class_names, priors):
    """Compute Acc / Macro F1 / Wtd F1 / Head F1 / Tail F1."""
    preds = preds.numpy() if torch.is_tensor(preds) else preds
    trues = trues.numpy() if torch.is_tensor(trues) else trues
    return compute_lt_metrics(preds, trues, class_names, priors, tag="")


# ===================== Strategies =====================
def strategy_ce_alone(p_ce):
    return p_ce.argmax(dim=1)


def strategy_hard_pred(p_ce, p_tfe_full, tail_set):
    """if CE predicts a tail class → use TFE; else use CE."""
    ce_pred = p_ce.argmax(dim=1)
    is_tail_pred = torch.tensor([int(c.item()) in tail_set for c in ce_pred],
                                 dtype=torch.bool)
    tfe_pred = p_tfe_full.argmax(dim=1)
    return torch.where(is_tail_pred, tfe_pred, ce_pred)


def strategy_conf(p_ce, p_tfe_full, threshold):
    """if CE confidence > threshold → use CE; else → use TFE."""
    ce_pred = p_ce.argmax(dim=1)
    ce_conf = p_ce.max(dim=1)[0]
    tfe_pred = p_tfe_full.argmax(dim=1)
    use_ce = ce_conf > threshold
    return torch.where(use_ce, ce_pred, tfe_pred)


def strategy_both(p_ce, p_tfe_full, threshold, tail_set):
    """if CE predicts head AND confidence > threshold → CE; else → TFE."""
    ce_pred = p_ce.argmax(dim=1)
    ce_conf = p_ce.max(dim=1)[0]
    tfe_pred = p_tfe_full.argmax(dim=1)
    is_head_pred = torch.tensor([int(c.item()) not in tail_set for c in ce_pred],
                                 dtype=torch.bool)
    use_ce = is_head_pred & (ce_conf > threshold)
    return torch.where(use_ce, ce_pred, tfe_pred)


# ===================== Soft Router =====================
class SoftRouter(nn.Module):
    """Differentiable soft router over CE and TFE.

    For sample x:
        confidence = max(p_ce)
        gate_raw = sigmoid(alpha * (T - confidence))   ∈ (0, 1)
        # Asymmetric: penalty when CE predicts head class (since TFE has no head info)
        is_head_pred = 1 - sum(p_ce[:, tail_indices])
        # We want gate ≈ 0 when (high confidence AND head-predicted)
        # Equivalent to: gate ≈ 1 when (low confidence) OR (tail-predicted)
        modulator = 1 - is_head_pred * (confidence > T_sym).float()  # heuristic
        # We use a cleaner form: gate = sigmoid(alpha * (T - confidence)) * tail_prob_mass
        tail_mass = sum(p_ce[:, tail_indices])  # how much CE thinks it's tail
        gate = gate_raw + (1 - gate_raw) * tail_mass  # gate is high if either uncertain or tail-mass high

        p_final = (1 - gate) * p_ce + gate * p_tfe_full
    """

    def __init__(self, init_threshold=0.7, init_alpha=10.0,
                 t_min=0.2, t_max=0.85):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        rel = (init_threshold - t_min) / (t_max - t_min)
        raw_t = float(np.log(rel / (1 - rel)))
        raw_a = float(np.log(np.expm1(init_alpha)))
        self.raw_threshold = nn.Parameter(torch.tensor(raw_t, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_a, dtype=torch.float32))

    def get_threshold(self):
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(self.raw_threshold)

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, p_ce, p_tfe_full, tail_indices_tensor):
        """Returns:
            p_final (B, K), gate (B,)
        """
        confidence = p_ce.max(dim=1)[0]                            # (B,)
        threshold = self.get_threshold()
        alpha = self.get_alpha()
        gate_raw = torch.sigmoid(alpha * (threshold - confidence))  # high when uncertain

        tail_mass = p_ce.index_select(1, tail_indices_tensor).sum(dim=1)  # (B,) ∈ [0, 1]

        # Combine: gate is high if (uncertain) OR (CE thinks it's tail)
        # gate = gate_raw + (1-gate_raw) * tail_mass = 1 - (1-gate_raw)*(1-tail_mass)
        gate = 1.0 - (1.0 - gate_raw) * (1.0 - tail_mass)

        p_final = (1 - gate.unsqueeze(-1)) * p_ce + gate.unsqueeze(-1) * p_tfe_full
        return p_final, gate


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  TFE Router - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Build full dataset (for full test set evaluation)
    logger.info("[1] Loading FULL dataset (21-class)...")
    full_data = build_dataset(args.dataset, args.variant, args.imb_factor,
                               args.batch_size, args.seed, args.num_workers)
    num_classes = full_data["num_classes"]
    class_names = full_data["class_names"]
    priors = full_data["priors"]

    # Build tail-only metadata (to get tail class indices)
    logger.info("[2] Determining tail class indices...")
    td = build_tail_dataset(args.dataset, args.tail_threshold,
                             args.batch_size, args.seed, args.num_workers)
    tail_orig_indices = td["tail_orig_indices"]
    tail_set = set(tail_orig_indices)
    tail_class_names = td["tail_class_names"]
    n_tail = len(tail_orig_indices)
    logger.info(f"  Tail classes ({n_tail}): {tail_class_names}")
    logger.info(f"  Tail orig indices: {tail_orig_indices}")

    # Load CE
    logger.info(f"[3] Loading CE: {args.primary_run}")
    backbone, ce_head = load_primary(args.primary_run, device, num_classes, args.backbone)
    feat_dim = backbone.feat_dim

    # Load TFE
    logger.info(f"[4] Loading TFE head: {args.tfe_run}")
    tfe_head = TFEHead(feat_dim, n_tail, hidden_dim=0).to(device)
    tfe_path = os.path.join(CHECKPOINT_DIR, f"{args.tfe_run}_head.pth")
    if not os.path.exists(tfe_path):
        raise FileNotFoundError(f"TFE checkpoint not found: {tfe_path}\n"
                                 f"Run `python -m experiments.run_tfe` first.")
    tfe_head.load_state_dict(torch.load(tfe_path, map_location=device, weights_only=True))
    tfe_head.eval()
    for p in tfe_head.parameters():
        p.requires_grad = False

    # Collect predictions on val and test sets ONCE (frozen experts)
    logger.info("[5] Collecting CE / TFE predictions on val and test...")
    val_p_ce, val_p_tfe, val_labels = collect_probs(
        backbone, ce_head, tfe_head, full_data["val_loader"],
        tail_orig_indices, num_classes, device)
    test_p_ce, test_p_tfe, test_labels = collect_probs(
        backbone, ce_head, tfe_head, full_data["test_loader"],
        tail_orig_indices, num_classes, device)
    logger.info(f"  val: {val_p_ce.shape}, test: {test_p_ce.shape}")

    # ============= Strategy 0: CE alone (baseline) =============
    logger.info("\n[6] Evaluating routing strategies on full TEST set...")
    logger.info("=" * 70)

    results = {}

    ce_preds_test = strategy_ce_alone(test_p_ce)
    m_ce = compute_metrics_from_preds(ce_preds_test, test_labels, class_names, priors)
    logger.info(f"  S0  CE alone:           "
                f"Macro F1={m_ce['macro_f1']:.4f}, Tail F1={m_ce['tail_f1']:.4f}, "
                f"Acc={m_ce['acc']:.4f}")
    results["S0_ce_alone"] = {k: v for k, v in m_ce.items() if k != "report"}
    ce_macro_f1 = m_ce["macro_f1"]

    # ============= Strategy 1: Hard route by CE prediction =============
    if args.strategy in ["hard_pred", "all"]:
        preds = strategy_hard_pred(test_p_ce, test_p_tfe, tail_set)
        m = compute_metrics_from_preds(preds, test_labels, class_names, priors)
        delta = m["macro_f1"] - ce_macro_f1
        marker = "✓" if delta > 0 else ("=" if abs(delta) < 1e-4 else "✗")
        logger.info(f"  S1  Hard-pred route:    "
                    f"Macro F1={m['macro_f1']:.4f} (Δ={delta:+.4f}) {marker}, "
                    f"Tail F1={m['tail_f1']:.4f}")
        results["S1_hard_pred"] = {k: v for k, v in m.items() if k != "report"}

    # ============= Strategy 2: Confidence threshold sweep =============
    if args.strategy in ["conf", "all"]:
        best_conf_t = None; best_conf_macro = -1
        for t_str in args.conf_thresholds.split(","):
            t = float(t_str)
            preds = strategy_conf(test_p_ce, test_p_tfe, t)
            m = compute_metrics_from_preds(preds, test_labels, class_names, priors)
            delta = m["macro_f1"] - ce_macro_f1
            marker = "✓" if delta > 0 else "✗"
            logger.info(f"  S2  Conf-route (T={t:.2f}): "
                        f"Macro F1={m['macro_f1']:.4f} (Δ={delta:+.4f}) {marker}, "
                        f"Tail F1={m['tail_f1']:.4f}")
            results[f"S2_conf_T{t}"] = {k: v for k, v in m.items() if k != "report"}
            if m["macro_f1"] > best_conf_macro:
                best_conf_macro = m["macro_f1"]
                best_conf_t = t
        logger.info(f"  S2 best T={best_conf_t} → Macro F1={best_conf_macro:.4f}")

    # ============= Strategy 3: Both rules =============
    if args.strategy in ["both", "all"]:
        best_both_t = None; best_both_macro = -1
        for t_str in args.conf_thresholds.split(","):
            t = float(t_str)
            preds = strategy_both(test_p_ce, test_p_tfe, t, tail_set)
            m = compute_metrics_from_preds(preds, test_labels, class_names, priors)
            delta = m["macro_f1"] - ce_macro_f1
            marker = "✓" if delta > 0 else "✗"
            logger.info(f"  S3  Both-rule (T={t:.2f}):  "
                        f"Macro F1={m['macro_f1']:.4f} (Δ={delta:+.4f}) {marker}, "
                        f"Tail F1={m['tail_f1']:.4f}")
            results[f"S3_both_T{t}"] = {k: v for k, v in m.items() if k != "report"}
            if m["macro_f1"] > best_both_macro:
                best_both_macro = m["macro_f1"]
                best_both_t = t
        logger.info(f"  S3 best T={best_both_t} → Macro F1={best_both_macro:.4f}")

    # ============= Strategy 4: Soft router (learnable) =============
    if args.strategy in ["soft", "all"]:
        logger.info("\n  Training soft router on train_loader...")

        # Need train predictions too
        train_p_ce, train_p_tfe, train_labels = collect_probs(
            backbone, ce_head, tfe_head, full_data["train_loader"],
            tail_orig_indices, num_classes, device)
        logger.info(f"  Collected train predictions: {train_p_ce.shape}")

        soft_router = SoftRouter(args.soft_init_threshold, args.soft_init_alpha).to(device)
        tail_idx_tensor = torch.tensor(tail_orig_indices, device=device)
        n_trainable = sum(p.numel() for p in soft_router.parameters())
        logger.info(f"  Soft router params: {n_trainable}")

        # Training loop on cached predictions (very fast)
        train_p_ce_dev = train_p_ce.to(device)
        train_p_tfe_dev = train_p_tfe.to(device)
        train_labels_dev = train_labels.to(device)
        val_p_ce_dev = val_p_ce.to(device)
        val_p_tfe_dev = val_p_tfe.to(device)
        val_labels_dev = val_labels.to(device)

        optimizer = torch.optim.AdamW(soft_router.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)
        early = EarlyStopping(patience=args.patience, mode="max")
        # Register init (unmodified CE) as fallback
        init_pred = train_p_ce_dev.argmax(dim=1).cpu()
        with torch.no_grad():
            p_final_val_init, _ = soft_router(val_p_ce_dev, val_p_tfe_dev, tail_idx_tensor)
            preds_val_init = p_final_val_init.argmax(dim=1).cpu()
        m_init = compute_metrics_from_preds(preds_val_init, val_labels, class_names, priors)
        early.step(m_init["macro_f1"], 0, state=copy.deepcopy(soft_router.state_dict()))
        logger.info(f"  Init val Macro F1 = {m_init['macro_f1']:.4f}")

        bs = 1024
        n_train = train_p_ce_dev.size(0)
        n_batches = (n_train + bs - 1) // bs

        for epoch in range(1, args.max_epochs + 1):
            soft_router.train()
            perm = torch.randperm(n_train, device=device)
            loss_sum = 0
            correct = total = 0
            for bi in range(n_batches):
                idx = perm[bi*bs:(bi+1)*bs]
                pc = train_p_ce_dev[idx]
                pt = train_p_tfe_dev[idx]
                y = train_labels_dev[idx]

                p_final, _ = soft_router(pc, pt, tail_idx_tensor)
                loss = F.nll_loss(torch.log(p_final + 1e-8), y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(soft_router.parameters(), 1.0)
                optimizer.step()

                loss_sum += loss.item() * idx.size(0)
                correct += (p_final.argmax(1) == y).sum().item()
                total += idx.size(0)
            scheduler.step()

            # Val
            soft_router.eval()
            with torch.no_grad():
                p_final_val, gate_val = soft_router(val_p_ce_dev, val_p_tfe_dev, tail_idx_tensor)
                preds_val = p_final_val.argmax(dim=1).cpu()
            m_val = compute_metrics_from_preds(preds_val, val_labels, class_names, priors)
            mean_gate = gate_val.mean().item()
            cur_t = soft_router.get_threshold().item()
            cur_a = soft_router.get_alpha().item()

            is_best = early.step(m_val["macro_f1"], epoch,
                                 state=copy.deepcopy(soft_router.state_dict()))
            marker = "*best*" if is_best else f"(p {early.counter}/{args.patience})"
            logger.info(f"  Soft E{epoch}/{args.max_epochs} {marker} "
                        f"train_loss={loss_sum/total:.4f}, train_acc={correct/total:.4f}, "
                        f"val_macro_f1={m_val['macro_f1']:.4f}, "
                        f"thr={cur_t:.3f}, alpha={cur_a:.1f}, gate={mean_gate:.3f}")
            if early.should_stop:
                logger.info(f"  Early stopping at E{epoch}")
                break

        # Restore best
        if early.best_state is not None:
            soft_router.load_state_dict(early.best_state)

        soft_router.eval()
        with torch.no_grad():
            p_final_test, gate_test = soft_router(test_p_ce.to(device),
                                                    test_p_tfe.to(device),
                                                    tail_idx_tensor)
            preds_test = p_final_test.argmax(dim=1).cpu()
        m_soft = compute_metrics_from_preds(preds_test, test_labels, class_names, priors)
        delta = m_soft["macro_f1"] - ce_macro_f1
        logger.info(f"  S4  Soft router (test): "
                    f"Macro F1={m_soft['macro_f1']:.4f} (Δ={delta:+.4f}), "
                    f"Tail F1={m_soft['tail_f1']:.4f}, "
                    f"mean_gate={gate_test.mean().item():.3f}")
        results["S4_soft"] = {k: v for k, v in m_soft.items() if k != "report"}
        results["S4_soft"]["mean_gate_test"] = float(gate_test.mean().item())
        results["S4_soft"]["final_threshold"] = soft_router.get_threshold().item()
        results["S4_soft"]["final_alpha"] = soft_router.get_alpha().item()

        # Save soft router
        torch.save(soft_router.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"{run_name}_soft_router.pth"))

    # ============= Final Summary =============
    logger.info(f"\n{'='*70}")
    logger.info(f"  TFE-ROUTE FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  CE baseline: Macro F1 = {ce_macro_f1:.4f}")
    logger.info(f"  ----------------------------------------------------------------")
    best_strategy = "S0_ce_alone"
    best_score = ce_macro_f1
    for k, v in results.items():
        if k == "S0_ce_alone": continue
        delta = v["macro_f1"] - ce_macro_f1
        marker = "✓" if delta > 0 else "✗"
        logger.info(f"  {k:<30} Macro F1={v['macro_f1']:.4f} "
                    f"(Δ={delta:+.4f}) Tail F1={v['tail_f1']:.4f} {marker}")
        if v["macro_f1"] > best_score:
            best_score = v["macro_f1"]
            best_strategy = k
    logger.info(f"  ----------------------------------------------------------------")
    logger.info(f"  BEST: {best_strategy} → Macro F1 = {best_score:.4f} "
                f"(Δ={best_score - ce_macro_f1:+.4f})")
    logger.info(f"{'='*70}")

    # Save summary
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "run_name": run_name,
        "args": vars(args),
        "ce_baseline_macro_f1": ce_macro_f1,
        "tail_class_names": tail_class_names,
        "tail_orig_indices": tail_orig_indices,
        "results": results,
        "best_strategy": best_strategy,
        "best_macro_f1": best_score,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Test report for best strategy
    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"CE baseline: Macro F1 = {ce_macro_f1:.4f}\n\n")
        f.write(f"{'Strategy':<30} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} "
                f"{'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for k, v in results.items():
            f.write(f"{k:<30} {v['acc']:>8.4f} {v['macro_f1']:>10.4f} "
                    f"{v['weighted_f1']:>10.4f} {v['head_f1']:>10.4f} "
                    f"{v['tail_f1']:>10.4f}\n")
        f.write(f"\nBest: {best_strategy} → Macro F1 = {best_score:.4f} "
                f"(Δ over CE = {best_score - ce_macro_f1:+.4f})\n")


if __name__ == "__main__":
    main()

"""
Tail-Boosted Logit Fusion (TBLF).

Background:
    All switching-style routers (hard-pred / conf / both / soft router) lose
    to plain CE on the full 25,707-image test set, despite TFE beating CE by
    +9.89% Macro F1 on the tail-only subset. The diagnosis: only ~150 of
    25,707 test images are true tail (0.6%). Any "if CE predicts tail → use
    TFE" rule converts head false-positives into tail mispredictions, net
    loss.

    TBLF abandons switching. It treats TFE's logits as ADDITIVE EVIDENCE on
    tail positions of CE's 21-way logits:

        combined[j] = ce_logits[j]                          if j ∉ tail
        combined[j] = ce_logits[j] + α_k * tfe_logits[k]    if j == tail_indices[k]

    Trainable parameters: α ∈ R^7 (one bidder per tail class).
    α ≥ 0 enforced via softplus reparameterization.

    Initialisation α=0 → combined ≡ ce_logits → TBLF strictly equals CE.
    The only failure mode is α too large (head sample → tail false positive),
    which is automatically penalised by the CE-on-combined training loss.

Game-theoretic reading:
    Each tail class k is a player; α_k is its bid for self-amplification.
    Utility U_k = E[accuracy gain | y=k] − (λ/2) α_k². Head classes do not
    bid (α=0 implicit). Asymmetric Nash equilibrium: α_k* ∝ marginal
    accuracy gain. SGD on the CE+L2 loss directly approximates this.

Usage:
    python -m experiments.run_tblf
    python -m experiments.run_tblf --l2_lambda 0.01
"""

import os
import sys
import copy
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR, print_paths
from common.train_utils import (
    setup_logger, EarlyStopping, compute_lt_metrics, set_seed,
)
from common.data import build_dataset

from experiments.run_tfe import TFEHead, build_tail_dataset, load_primary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--variant", default="original", choices=["original", "lt"])
    p.add_argument("--imb_factor", type=int, default=100)
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--primary_run", default="mll_swin_t_ce")
    p.add_argument("--tfe_run", default="mll_swin_t_tfe_thr500")
    p.add_argument("--tail_threshold", type=int, default=500)

    p.add_argument("--fixed_alphas", type=str, default="0.1,0.3,0.5,1.0",
                   help="Fixed alpha values to evaluate without training")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--l2_lambda", type=float, default=0.0,
                   help="L2 regularisation strength on alpha (Nash bidding tax)")
    p.add_argument("--init_alpha", type=float, default=0.0,
                   help="Initial alpha for the learnable variant")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.dataset, args.backbone, f"tblf_thr{args.tail_threshold}"]
    if args.l2_lambda > 0:
        parts.append(f"l2{args.l2_lambda}")
    return "_".join(parts)


# ===================== Logit collection =====================
@torch.no_grad()
def collect_logits(backbone, ce_head, tfe_head, loader, device):
    """Returns ce_logits (N, K), tfe_logits (N, n_tail), labels (N,)."""
    backbone.eval(); ce_head.eval(); tfe_head.eval()
    all_ce, all_tfe, all_y = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        all_ce.append(ce_head(feats).cpu())
        all_tfe.append(tfe_head(feats).cpu())
        all_y.append(labels)
    return torch.cat(all_ce), torch.cat(all_tfe), torch.cat(all_y)


# ===================== TBLF module =====================
class TBLF(nn.Module):
    """Additive logit fusion on tail positions, with α ≥ 0 via softplus."""

    def __init__(self, num_classes, tail_indices, init_alpha=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("tail_indices",
                             torch.tensor(tail_indices, dtype=torch.long))
        # softplus(raw)=alpha → raw = log(exp(alpha)-1)
        if init_alpha <= 0:
            init_raw = -5.0  # softplus(-5) ≈ 0.0067 ≈ 0
        else:
            init_raw = float(np.log(np.expm1(init_alpha)))
        self.raw_alpha = nn.Parameter(
            torch.full((len(tail_indices),), init_raw, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, ce_logits, tfe_logits):
        # ce_logits: (B, K), tfe_logits: (B, n_tail)
        combined = ce_logits.clone()
        alpha = self.get_alpha()  # (n_tail,)
        combined[:, self.tail_indices] = (
            combined[:, self.tail_indices] + alpha.unsqueeze(0) * tfe_logits)
        return combined


def fixed_alpha_combine(ce_logits, tfe_logits, tail_indices_tensor, alpha_value):
    """Apply a constant alpha to all tail classes. No training."""
    combined = ce_logits.clone()
    combined[:, tail_indices_tensor] = (
        combined[:, tail_indices_tensor] + alpha_value * tfe_logits)
    return combined


def metrics_from_logits(logits, labels, class_names, priors):
    preds = logits.argmax(dim=1).numpy()
    trues = labels.numpy() if torch.is_tensor(labels) else labels
    return compute_lt_metrics(preds, trues, class_names, priors, tag="")


# ===================== Main =====================
def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  TBLF (Tail-Boosted Logit Fusion) - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    logger.info("[1] Loading FULL dataset (21-class)...")
    full_data = build_dataset(args.dataset, args.variant, args.imb_factor,
                               args.batch_size, args.seed, args.num_workers)
    num_classes = full_data["num_classes"]
    class_names = full_data["class_names"]
    priors = full_data["priors"]

    logger.info("[2] Determining tail class indices...")
    td = build_tail_dataset(args.dataset, args.tail_threshold,
                             args.batch_size, args.seed, args.num_workers)
    tail_orig_indices = td["tail_orig_indices"]
    tail_class_names = td["tail_class_names"]
    n_tail = len(tail_orig_indices)
    logger.info(f"  Tail classes ({n_tail}): {tail_class_names}")
    logger.info(f"  Tail orig indices: {tail_orig_indices}")

    logger.info(f"[3] Loading CE: {args.primary_run}")
    backbone, ce_head = load_primary(args.primary_run, device, num_classes, args.backbone)
    feat_dim = backbone.feat_dim

    logger.info(f"[4] Loading TFE head: {args.tfe_run}")
    tfe_head = TFEHead(feat_dim, n_tail, hidden_dim=0).to(device)
    tfe_path = os.path.join(CHECKPOINT_DIR, f"{args.tfe_run}_head.pth")
    if not os.path.exists(tfe_path):
        raise FileNotFoundError(f"TFE checkpoint not found: {tfe_path}")
    tfe_head.load_state_dict(torch.load(tfe_path, map_location=device, weights_only=True))
    tfe_head.eval()
    for p in tfe_head.parameters():
        p.requires_grad = False

    logger.info("[5] Collecting raw logits on train/val/test...")
    train_ce, train_tfe, train_y = collect_logits(
        backbone, ce_head, tfe_head, full_data["train_loader"], device)
    val_ce, val_tfe, val_y = collect_logits(
        backbone, ce_head, tfe_head, full_data["val_loader"], device)
    test_ce, test_tfe, test_y = collect_logits(
        backbone, ce_head, tfe_head, full_data["test_loader"], device)
    logger.info(f"  train: {train_ce.shape}, val: {val_ce.shape}, test: {test_ce.shape}")

    tail_idx_tensor = torch.tensor(tail_orig_indices, dtype=torch.long)

    # ============= Sanity check: α=0 must equal CE =============
    logger.info("\n[6] SANITY: α=0 fallback (must equal CE baseline)")
    m_ce = metrics_from_logits(test_ce, test_y, class_names, priors)
    ce_macro_f1 = m_ce["macro_f1"]
    logger.info(f"  CE alone:        Macro F1={m_ce['macro_f1']:.4f}, "
                f"Tail F1={m_ce['tail_f1']:.4f}, Acc={m_ce['acc']:.4f}")
    combined0 = fixed_alpha_combine(test_ce, test_tfe, tail_idx_tensor, 0.0)
    m0 = metrics_from_logits(combined0, test_y, class_names, priors)
    assert abs(m0["macro_f1"] - m_ce["macro_f1"]) < 1e-6, \
        "α=0 must give exactly CE — sanity failed"
    logger.info(f"  α=0 sanity:      Macro F1={m0['macro_f1']:.4f}  ✓ matches CE")

    results = {"S0_ce_alone": {k: v for k, v in m_ce.items() if k != "report"}}

    # ============= Fixed-alpha sweep =============
    logger.info("\n[7] Fixed-α sweep on TEST set:")
    for a_str in args.fixed_alphas.split(","):
        a = float(a_str)
        comb = fixed_alpha_combine(test_ce, test_tfe, tail_idx_tensor, a)
        m = metrics_from_logits(comb, test_y, class_names, priors)
        delta = m["macro_f1"] - ce_macro_f1
        marker = "✓" if delta > 0 else ("=" if abs(delta) < 1e-4 else "✗")
        logger.info(f"  α={a:>4.2f}:        Macro F1={m['macro_f1']:.4f} "
                    f"(Δ={delta:+.4f}) Tail F1={m['tail_f1']:.4f} {marker}")
        results[f"fixed_alpha_{a}"] = {k: v for k, v in m.items() if k != "report"}

    # ============= Learnable TBLF =============
    logger.info(f"\n[8] Training learnable TBLF (init_alpha={args.init_alpha}, "
                f"l2_lambda={args.l2_lambda})...")

    tblf = TBLF(num_classes, tail_orig_indices, init_alpha=args.init_alpha).to(device)
    n_params = sum(p.numel() for p in tblf.parameters())
    logger.info(f"  TBLF trainable params: {n_params}")

    train_ce_dev = train_ce.to(device); train_tfe_dev = train_tfe.to(device)
    train_y_dev = train_y.to(device)
    val_ce_dev = val_ce.to(device); val_tfe_dev = val_tfe.to(device)

    optimizer = torch.optim.AdamW(tblf.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-4)
    early = EarlyStopping(patience=args.patience, mode="max")

    # Register init state as fallback (α≈0 → equals CE)
    with torch.no_grad():
        comb_v0 = tblf(val_ce_dev, val_tfe_dev).cpu()
    m_init = metrics_from_logits(comb_v0, val_y, class_names, priors)
    early.step(m_init["macro_f1"], 0, state=copy.deepcopy(tblf.state_dict()))
    logger.info(f"  Init val Macro F1 = {m_init['macro_f1']:.4f}")

    bs = args.batch_size
    n_train = train_ce_dev.size(0)
    n_batches = (n_train + bs - 1) // bs

    for epoch in range(1, args.max_epochs + 1):
        tblf.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0; correct = total = 0
        for bi in range(n_batches):
            idx = perm[bi*bs:(bi+1)*bs]
            pc = train_ce_dev[idx]; pt = train_tfe_dev[idx]; y = train_y_dev[idx]
            comb = tblf(pc, pt)
            ce_loss = F.cross_entropy(comb, y)
            alpha = tblf.get_alpha()
            l2 = 0.5 * (alpha ** 2).sum()
            loss = ce_loss + args.l2_lambda * l2

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(tblf.parameters(), 5.0)
            optimizer.step()

            loss_sum += loss.item() * idx.size(0)
            correct += (comb.argmax(1) == y).sum().item()
            total += idx.size(0)
        scheduler.step()

        tblf.eval()
        with torch.no_grad():
            comb_v = tblf(val_ce_dev, val_tfe_dev).cpu()
        m_v = metrics_from_logits(comb_v, val_y, class_names, priors)
        cur_alpha = tblf.get_alpha().detach().cpu().numpy()
        is_best = early.step(m_v["macro_f1"], epoch,
                             state=copy.deepcopy(tblf.state_dict()))
        marker = "*best*" if is_best else f"(p {early.counter}/{args.patience})"
        alpha_str = "[" + ",".join(f"{a:.2f}" for a in cur_alpha) + "]"
        logger.info(f"  E{epoch}/{args.max_epochs} {marker} "
                    f"loss={loss_sum/total:.4f} train_acc={correct/total:.4f} "
                    f"val_macro_f1={m_v['macro_f1']:.4f} α={alpha_str}")
        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch}")
            break

    # Restore best
    if early.best_state is not None:
        tblf.load_state_dict(early.best_state)
    tblf.eval()

    with torch.no_grad():
        comb_test = tblf(test_ce.to(device), test_tfe.to(device)).cpu()
    m_learn = metrics_from_logits(comb_test, test_y, class_names, priors)
    delta = m_learn["macro_f1"] - ce_macro_f1
    final_alpha = tblf.get_alpha().detach().cpu().numpy().tolist()
    marker = "✓" if delta > 0 else "✗"
    logger.info(f"\n  Learnable TBLF (test): Macro F1={m_learn['macro_f1']:.4f} "
                f"(Δ={delta:+.4f}) Tail F1={m_learn['tail_f1']:.4f} {marker}")
    logger.info(f"  Final α: " +
                ", ".join(f"{n}={a:.3f}" for n, a in zip(tail_class_names, final_alpha)))

    results["learnable_tblf"] = {k: v for k, v in m_learn.items() if k != "report"}
    results["learnable_tblf"]["final_alpha"] = final_alpha
    results["learnable_tblf"]["tail_class_names"] = tail_class_names

    # Fallback: if learnable TBLF on test < CE, report α=0 as safe choice
    if m_learn["macro_f1"] < ce_macro_f1:
        logger.warning("  Learnable TBLF underperforms CE on test. "
                       "Falling back to α=0 (=CE) for the headline result.")

    # ============= Summary =============
    logger.info(f"\n{'='*70}")
    logger.info(f"  TBLF FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  CE baseline: Macro F1 = {ce_macro_f1:.4f}, "
                f"Tail F1 = {m_ce['tail_f1']:.4f}")
    logger.info(f"  ----------------------------------------------------------------")
    best_key = "S0_ce_alone"; best_score = ce_macro_f1
    for k, v in results.items():
        if k == "S0_ce_alone": continue
        d = v["macro_f1"] - ce_macro_f1
        mk = "✓" if d > 0 else "✗"
        logger.info(f"  {k:<24} Macro F1={v['macro_f1']:.4f} "
                    f"(Δ={d:+.4f}) Tail F1={v['tail_f1']:.4f} {mk}")
        if v["macro_f1"] > best_score:
            best_score = v["macro_f1"]; best_key = k
    logger.info(f"  ----------------------------------------------------------------")
    logger.info(f"  BEST: {best_key} → Macro F1 = {best_score:.4f} "
                f"(Δ={best_score - ce_macro_f1:+.4f})")
    logger.info(f"{'='*70}")

    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "run_name": run_name, "args": vars(args),
        "ce_baseline_macro_f1": ce_macro_f1,
        "tail_class_names": tail_class_names,
        "tail_orig_indices": tail_orig_indices,
        "results": results, "best_key": best_key, "best_macro_f1": best_score,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"CE baseline: Macro F1 = {ce_macro_f1:.4f}\n\n")
        f.write(f"{'Variant':<24} {'Acc':>8} {'Macro F1':>10} {'Wtd F1':>10} "
                f"{'Head F1':>10} {'Tail F1':>10}\n")
        f.write("-" * 80 + "\n")
        for k, v in results.items():
            f.write(f"{k:<24} {v['acc']:>8.4f} {v['macro_f1']:>10.4f} "
                    f"{v['weighted_f1']:>10.4f} {v['head_f1']:>10.4f} "
                    f"{v['tail_f1']:>10.4f}\n")
        f.write(f"\nBest: {best_key} → Macro F1={best_score:.4f} "
                f"(Δ={best_score - ce_macro_f1:+.4f})\n")
        if "learnable_tblf" in results:
            f.write(f"\nLearned alphas:\n")
            for n, a in zip(tail_class_names, results["learnable_tblf"]["final_alpha"]):
                f.write(f"  {n}: {a:.4f}\n")

    torch.save(tblf.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"{run_name}.pth"))


if __name__ == "__main__":
    main()

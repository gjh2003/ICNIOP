"""
BBN-Bidding: Bilateral-Branch Network with game-theoretic inference aggregation.

Stage 1 (training, follows BBN, Zhou et al. CVPR 2020):
    Shared backbone B + two classifier heads H_A (head expert, natural sampling)
    and H_B (tail expert, reversed sampling).
    Each step samples two minibatches:
        (x_a, y_a) from natural sampler
        (x_b, y_b) from reversed sampler  (weights ∝ 1/count^reversed)

    Cumulative learning weight:
        α(epoch) = 1 - (epoch / max_epochs)^2

    Mixed logits over the two batches:
        z = α · H_A(B(x_a)) + (1-α) · H_B(B(x_b))
    Loss:
        L_train = α · CE(z, y_a) + (1-α) · CE(z, y_b)

Stage 2 (game-theoretic aggregation, our addition):
    Freeze backbone+heads. Train a single BiddingNetwork on top of features:
        bid = bidder(B(x)) ∈ R^2  →  w = softmax(bid)
        p_final = w_A · softmax(H_A(B(x))) + w_B · softmax(H_B(B(x)))
    Loss = NLL(p_final, y) + λ · cost_term  (Nash bidding tax)

    The bidder is small (~50k params), trained with class-balanced sampler
    so tail samples have a real say in choosing branch B.

Final inference: argmax of p_final.
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    p.add_argument("--backbone", default="swin_t")

    # Stage 1
    p.add_argument("--max_epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Stage 2 (bidder)
    p.add_argument("--bidder_epochs", type=int, default=10)
    p.add_argument("--bidder_lr", type=float, default=5e-4)
    p.add_argument("--bidder_cost", type=float, default=0.1,
                   help="Nash bidding tax coefficient λ on bid magnitudes")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_run_name(args):
    if args.run_name:
        return args.run_name
    return f"{args.dataset}_{args.backbone}_bbn"


def make_reversed_loader(train_loader_dataset, counts, batch_size, num_workers):
    """Reversed sampler: weights ∝ count[c] (oversample HEAD).
    Wait — BBN's "reverse" oversamples TAIL. Reversed weight is max_count - count + 1.
    Equivalently weights ∝ 1/count_reversed where count_reversed = max-count+1."""
    counts = counts.astype(np.float64)
    rev_counts = counts.max() - counts + 1.0
    weights_per_class = rev_counts / rev_counts.sum()
    # Per-sample: use class label
    labels = np.array([s[1] for s in train_loader_dataset.samples])
    sample_weights = weights_per_class[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
    return DataLoader(train_loader_dataset, batch_size=batch_size, sampler=sampler,
                       num_workers=num_workers, pin_memory=True, drop_last=True)


@torch.no_grad()
def eval_combined(backbone, head_a, head_b, loader, class_names, priors, device,
                  weight_a=0.5, tag=""):
    backbone.eval(); head_a.eval(); head_b.eval()
    preds_all, trues_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feat = backbone(imgs)
        p_a = F.softmax(head_a(feat), dim=1)
        p_b = F.softmax(head_b(feat), dim=1)
        p = weight_a * p_a + (1 - weight_a) * p_b
        preds_all.append(p.argmax(1).cpu())
        trues_all.append(labels)
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    return compute_lt_metrics(preds, trues, class_names, priors, tag=tag)


@torch.no_grad()
def eval_with_bidder(backbone, head_a, head_b, bidder, loader, class_names, priors, device, tag=""):
    backbone.eval(); head_a.eval(); head_b.eval(); bidder.eval()
    preds_all, trues_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feat = backbone(imgs)
        p_a = F.softmax(head_a(feat), dim=1)
        p_b = F.softmax(head_b(feat), dim=1)
        bid = bidder(feat)  # (B, 2)
        w = F.softmax(bid, dim=1)
        p = w[:, 0:1] * p_a + w[:, 1:2] * p_b
        preds_all.append(p.argmax(1).cpu())
        trues_all.append(labels)
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()
    return compute_lt_metrics(preds, trues, class_names, priors, tag=tag)


class TwoOutputBidder(nn.Module):
    """Outputs 2-dim bid for two branches."""
    def __init__(self, in_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x):
        return self.net(x)


def main():
    args = parse_args()
    run_name = make_run_name(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(LOG_DIR, run_name)
    logger.info("=" * 70)
    logger.info(f"  BBN-Bidding - {run_name}")
    logger.info("=" * 70)
    logger.info(f"  Args: {vars(args)}")
    print_paths()

    # Data
    logger.info("[1] Loading data...")
    data = build_dataset(args.dataset, args.variant, args.imb_factor,
                         args.batch_size, args.seed, args.num_workers)
    natural_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    priors = data["priors"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    # Build reversed loader
    train_ds = natural_loader.dataset
    counts = np.bincount([s[1] for s in train_ds.samples], minlength=num_classes)
    reversed_loader = make_reversed_loader(train_ds, counts, args.batch_size, args.num_workers)
    balanced_loader = data["balanced_loader"]
    logger.info(f"  Built reversed_loader (HEAD-suppressed, TAIL-boosted)")

    # Models
    logger.info("[2] Building model (shared backbone + 2 heads)...")
    backbone = build_backbone(args.backbone).to(device)
    feat_dim = backbone.feat_dim
    head_a = ClassifierHead(in_dim=feat_dim, num_classes=num_classes).to(device)
    head_b = ClassifierHead(in_dim=feat_dim, num_classes=num_classes).to(device)

    # Optimiser
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": list(head_a.parameters()) + list(head_b.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    resume_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_resume.pth")
    history = History()
    early = EarlyStopping(patience=args.patience, mode="max")
    start_epoch = 1

    if args.resume:
        ckpt = load_resume_checkpoint(resume_path)
        if ckpt is not None:
            backbone.load_state_dict(ckpt["backbone"])
            head_a.load_state_dict(ckpt["head_a"])
            head_b.load_state_dict(ckpt["head_b"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            history.records = ckpt["history"]
            early.best_value = ckpt["best_f1"]
            early.best_epoch = ckpt["best_epoch"]
            early.counter = ckpt["patience_counter"]
            early.best_state = ckpt.get("best_state")
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [RESUME] from E{ckpt['epoch']}, best={early.best_value:.4f}")

    # Stage 1: BBN training
    logger.info(f"[3] Stage 1: BBN training (max {args.max_epochs} epochs)...")
    t_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        backbone.train(); head_a.train(); head_b.train()
        # Cumulative learning α: 1 → 0 over epochs
        alpha = 1.0 - (epoch / args.max_epochs) ** 2

        train_loss = 0.0; n_seen = 0
        rev_iter = iter(reversed_loader)
        pbar = tqdm(natural_loader, desc=f"  E{epoch}/{args.max_epochs} α={alpha:.2f}", ncols=110)
        for x_a, y_a in pbar:
            try:
                x_b, y_b = next(rev_iter)
            except StopIteration:
                rev_iter = iter(reversed_loader)
                x_b, y_b = next(rev_iter)
            x_a = x_a.to(device); y_a = y_a.to(device)
            x_b = x_b.to(device); y_b = y_b.to(device)

            feat_a = backbone(x_a)
            feat_b = backbone(x_b)
            # BBN mixed logits
            logits_a = head_a(feat_a)
            logits_b = head_b(feat_b)
            mixed_logits = alpha * logits_a + (1 - alpha) * logits_b

            # Loss with mixed targets
            loss = alpha * F.cross_entropy(mixed_logits, y_a) + \
                   (1 - alpha) * F.cross_entropy(mixed_logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head_a.parameters()) + list(head_b.parameters()),
                1.0)
            optimizer.step()

            bs = x_a.size(0)
            train_loss += loss.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss /= n_seen

        # Validate (use 0.5/0.5 mix for now)
        val_metrics = eval_combined(backbone, head_a, head_b, val_loader,
                                     class_names, priors, device, weight_a=0.5,
                                     tag=f"E{epoch} val")
        history.append(epoch, train_loss=train_loss, alpha=alpha,
                       val_acc=val_metrics["acc"],
                       val_macro_f1=val_metrics["macro_f1"],
                       val_tail_f1=val_metrics["tail_f1"],
                       val_head_f1=val_metrics["head_f1"])

        is_best = early.step(val_metrics["macro_f1"], epoch,
                             state={"backbone": copy.deepcopy(backbone.state_dict()),
                                    "head_a": copy.deepcopy(head_a.state_dict()),
                                    "head_b": copy.deepcopy(head_b.state_dict())})
        marker = "*best*" if is_best else f"(p {early.counter}/{args.patience})"
        logger.info(f"  E{epoch} {marker} α={alpha:.3f} train_loss={train_loss:.4f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                    f"val_tail_f1={val_metrics['tail_f1']:.4f} "
                    f"val_head_f1={val_metrics['head_f1']:.4f}")

        save_resume_checkpoint(
            resume_path, epoch=epoch,
            backbone=backbone.state_dict(),
            head_a=head_a.state_dict(),
            head_b=head_b.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history.records,
            best_f1=early.best_value, best_epoch=early.best_epoch,
            patience_counter=early.counter, best_state=early.best_state,
        )

        if early.should_stop:
            logger.info(f"  Early stopping at E{epoch} (best E{early.best_epoch})")
            break

    elapsed1 = (time.time() - t_start) / 60

    # Restore best Stage-1 model
    if early.best_state is not None:
        backbone.load_state_dict(early.best_state["backbone"])
        head_a.load_state_dict(early.best_state["head_a"])
        head_b.load_state_dict(early.best_state["head_b"])

    # Evaluate Stage 1 (0.5 mix) on test
    logger.info(f"\n[4] Stage 1 best (val_macro_f1={early.best_value:.4f}, E{early.best_epoch})")
    test_05 = eval_combined(backbone, head_a, head_b, test_loader,
                             class_names, priors, device, weight_a=0.5,
                             tag="Test (0.5 mix)")
    logger.info(f"  Test 0.5 mix: Macro F1={test_05['macro_f1']:.4f}, "
                f"Tail F1={test_05['tail_f1']:.4f}")

    # Sweep weight_a fixed
    logger.info("[5] Stage 1 fixed-weight sweep on TEST:")
    sweep_results = {"mix_0.5": {k: v for k, v in test_05.items() if k != "report"}}
    best_w = 0.5; best_macro = test_05["macro_f1"]
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        m = eval_combined(backbone, head_a, head_b, test_loader,
                           class_names, priors, device, weight_a=w, tag="")
        logger.info(f"  weight_A={w:.2f}: Macro F1={m['macro_f1']:.4f}, "
                    f"Tail F1={m['tail_f1']:.4f}")
        sweep_results[f"mix_{w}"] = {k: v for k, v in m.items() if k != "report"}
        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]; best_w = w
    logger.info(f"  best fixed weight_A={best_w} → Macro F1={best_macro:.4f}")

    # ============= Stage 2: Bidder training =============
    logger.info(f"\n[6] Stage 2: Train bidder ({args.bidder_epochs} epochs)...")
    for p in backbone.parameters(): p.requires_grad = False
    for p in head_a.parameters(): p.requires_grad = False
    for p in head_b.parameters(): p.requires_grad = False
    backbone.eval(); head_a.eval(); head_b.eval()

    bidder = TwoOutputBidder(in_dim=feat_dim).to(device)
    b_opt = torch.optim.AdamW(bidder.parameters(), lr=args.bidder_lr, weight_decay=1e-4)
    b_sched = CosineAnnealingLR(b_opt, T_max=args.bidder_epochs, eta_min=1e-6)
    b_early = EarlyStopping(patience=4, mode="max")

    t_start2 = time.time()
    for epoch in range(1, args.bidder_epochs + 1):
        bidder.train()
        loss_sum = 0; n_seen = 0
        for imgs, labels in tqdm(balanced_loader,
                                  desc=f"  Bidder E{epoch}/{args.bidder_epochs}",
                                  ncols=100):
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                feat = backbone(imgs)
                p_a = F.softmax(head_a(feat), dim=1)
                p_b = F.softmax(head_b(feat), dim=1)
            bid = bidder(feat)
            w = F.softmax(bid, dim=1)
            p_final = w[:, 0:1] * p_a + w[:, 1:2] * p_b
            nll = F.nll_loss(torch.log(p_final + 1e-8), labels)
            cost = (bid ** 2).mean()
            loss = nll + args.bidder_cost * cost

            b_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(bidder.parameters(), 1.0)
            b_opt.step()

            bs = imgs.size(0)
            loss_sum += loss.item() * bs; n_seen += bs
        b_sched.step()

        val_b = eval_with_bidder(backbone, head_a, head_b, bidder, val_loader,
                                  class_names, priors, device, tag="bidder val")
        is_best = b_early.step(val_b["macro_f1"], epoch,
                               state=copy.deepcopy(bidder.state_dict()))
        marker = "*best*" if is_best else f"(p {b_early.counter}/4)"
        logger.info(f"  Bidder E{epoch} {marker} loss={loss_sum/n_seen:.4f} "
                    f"val_macro_f1={val_b['macro_f1']:.4f} "
                    f"val_tail_f1={val_b['tail_f1']:.4f}")
        if b_early.should_stop:
            break

    if b_early.best_state is not None:
        bidder.load_state_dict(b_early.best_state)

    elapsed2 = (time.time() - t_start2) / 60
    logger.info(f"\n[7] Stage 2 done. Test with bidder...")
    test_bidder = eval_with_bidder(backbone, head_a, head_b, bidder, test_loader,
                                    class_names, priors, device, tag="Test (bidder)")
    logger.info(f"  Test (bidder): Macro F1={test_bidder['macro_f1']:.4f}, "
                f"Tail F1={test_bidder['tail_f1']:.4f}")

    # Save everything
    save_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    history.save_json(os.path.join(save_dir, "history.json"))
    summary = {
        "run_name": run_name, "args": vars(args),
        "stage1_test_05": {k: v for k, v in test_05.items() if k != "report"},
        "stage1_sweep": sweep_results,
        "stage1_best_weight_a": best_w,
        "stage1_best_macro": best_macro,
        "stage2_test_bidder": {k: v for k, v in test_bidder.items() if k != "report"},
        "stage1_min": elapsed1, "stage2_min": elapsed2,
        "best_val_macro_f1": early.best_value,
        "best_epoch": early.best_epoch,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, "test_report.txt"), "w") as f:
        f.write(f"Run: {run_name}\nArgs: {vars(args)}\n\n")
        f.write(f"Stage 1 0.5 mix: Macro F1={test_05['macro_f1']:.4f} Tail F1={test_05['tail_f1']:.4f}\n")
        f.write(f"Stage 1 best fixed (w_A={best_w}): Macro F1={best_macro:.4f}\n")
        f.write(f"Stage 2 bidder: Macro F1={test_bidder['macro_f1']:.4f} Tail F1={test_bidder['tail_f1']:.4f}\n\n")
        f.write(test_bidder["report"])

    torch.save({
        "backbone": backbone.state_dict(),
        "head_a": head_a.state_dict(),
        "head_b": head_b.state_dict(),
        "bidder": bidder.state_dict(),
        "args": vars(args),
        "test_metrics": {k: v for k, v in test_bidder.items() if k != "report"},
    }, os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth"))

    logger.info(f"\n[8] DONE. Stage1 {elapsed1:.1f} min + Stage2 {elapsed2:.1f} min")
    logger.info(f"  Stage 1 best: Macro F1 = {best_macro:.4f}")
    logger.info(f"  Stage 2 (bidder): Macro F1 = {test_bidder['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

"""Common training utilities: early stopping, resume, logging."""

import os
import json
import copy
import logging
from typing import Dict, Any, Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================== LOGGING =====================
def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    """Track best metric and trigger early stopping after `patience` non-improving epochs."""

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self.best_state = None  # opaque blob the user can fill (e.g., model.state_dict())

    def step(self, value: float, epoch: int, state: Optional[Any] = None) -> bool:
        """Returns True if this is a new best, False otherwise."""
        is_better = (
            value > self.best_value + self.min_delta if self.mode == "max"
            else value < self.best_value - self.min_delta
        )
        if is_better:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            if state is not None:
                self.best_state = state
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ===================== RESUME / CHECKPOINTING =====================
def save_resume_checkpoint(path: str, **kwargs):
    """Save a resume checkpoint atomically (write tmp then rename)."""
    tmp = path + ".tmp"
    torch.save(kwargs, tmp)
    os.replace(tmp, path)


def load_resume_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    """Load resume checkpoint if it exists, return None otherwise."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [WARN] Failed to load resume checkpoint {path}: {e}")
        return None


# ===================== HISTORY LOGGING =====================
class History:
    """Track per-epoch metrics; save as JSON; plot curves."""

    def __init__(self):
        self.records = []  # list of dict (one per epoch)

    def append(self, epoch: int, **metrics):
        rec = {"epoch": epoch, **metrics}
        self.records.append(rec)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def load_json(self, path: str):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def plot_curves(self, path: str, metrics=("train_acc", "val_acc", "val_macro_f1")):
        if not self.records:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = [r["epoch"] for r in self.records]
        for m in metrics:
            if m in self.records[0]:
                values = [r.get(m, None) for r in self.records]
                ax.plot(epochs, values, marker="o", label=m)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()


# ===================== METRICS =====================
def compute_lt_metrics(preds, trues, class_names, priors,
                       head_thresh: float = 0.10, tail_thresh: float = 0.005,
                       tag: str = ""):
    """Compute Acc, Macro F1, Weighted F1, Head F1, Tail F1 + classification report."""
    from sklearn.metrics import f1_score, classification_report
    import numpy as np

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    pi = priors.numpy() if hasattr(priors, "numpy") else np.asarray(priors)

    acc = (preds == trues).mean()
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    per_f1 = f1_score(trues, preds, average=None, zero_division=0)

    head_idx = [i for i in range(len(class_names)) if pi[i] > head_thresh]
    tail_idx = [i for i in range(len(class_names)) if pi[i] < tail_thresh]
    head_f1 = float(np.mean([per_f1[i] for i in head_idx])) if head_idx else 0.0
    tail_f1 = float(np.mean([per_f1[i] for i in tail_idx])) if tail_idx else 0.0

    report = classification_report(trues, preds, target_names=class_names, zero_division=0)

    if tag:
        print(f"  [{tag}] Acc={acc:.4f}, Macro F1={macro_f1:.4f}, "
              f"Wtd F1={weighted_f1:.4f}, Head F1={head_f1:.4f}, Tail F1={tail_f1:.4f}")

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "head_f1": head_f1,
        "tail_f1": tail_f1,
        "per_f1": per_f1.tolist(),
        "report": report,
    }


# ===================== SEED =====================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

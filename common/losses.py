"""Loss functions for long-tail classification baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Standard CE / Weighted CE =====================
def standard_ce(class_priors=None, device=None):
    return nn.CrossEntropyLoss()


def weighted_ce(class_priors, device=None, num_classes=None):
    if num_classes is None:
        num_classes = len(class_priors)
    w = 1.0 / class_priors
    w = w / w.sum() * num_classes
    if device is not None:
        w = w.to(device)
    return nn.CrossEntropyLoss(weight=w)


# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    """Focal loss with optional alpha (per-class weights = inverse frequency)."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        pt = p[range(len(targets)), targets]
        focal_weight = (1.0 - pt).pow(self.gamma)
        ce = F.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        return loss.mean()


def focal_loss(class_priors, device, gamma: float = 2.0):
    alpha = 1.0 / class_priors.to(device)
    alpha = alpha / alpha.mean()
    return FocalLoss(gamma=gamma, alpha=alpha)


# ===================== Logit Adjustment =====================
class LogitAdjustmentLoss(nn.Module):
    """Logit Adjustment (Menon et al., ICLR 2021):
       loss = CE(logits + tau * log(pi), y)
    """
    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.register_buffer("log_priors", torch.log(class_priors + 1e-8))

    def forward(self, logits, targets):
        adjusted = logits + self.tau * self.log_priors
        return F.cross_entropy(adjusted, targets)


def logit_adjustment_loss(class_priors, device, tau: float = 1.0):
    return LogitAdjustmentLoss(class_priors.to(device), tau=tau)


# ===================== LDAM Loss =====================
class LDAMLoss(nn.Module):
    """LDAM (Learning with Deferred Re-balancing): label-aware margin loss.

    Cao et al., NeurIPS 2019.
    Margin per class: m_y = C / n_y^{1/4}
    """

    def __init__(self, class_priors: torch.Tensor, max_m: float = 0.5, s: float = 30.0,
                 weight: torch.Tensor = None):
        super().__init__()
        # Margin: inversely proportional to n_y^{1/4}
        m_list = 1.0 / torch.sqrt(torch.sqrt(class_priors + 1e-8))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = s
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)

        batch_m = self.m_list[targets].view(-1, 1)
        x_m = logits - batch_m * index.float()
        return F.cross_entropy(self.s * x_m, targets, weight=self.weight)


def ldam_drw_loss(class_priors, device, epoch: int = 0, drw_start: int = -1,
                  max_m: float = 0.5, s: float = 30.0, num_classes: int = None):
    """LDAM with Deferred Re-Weighting:
       - epochs < drw_start: LDAM only
       - epochs >= drw_start: LDAM + class-balanced weights

    Note: drw_start = -1 means no DRW (pure LDAM).
    """
    if num_classes is None:
        num_classes = len(class_priors)

    weight = None
    if drw_start >= 0 and epoch >= drw_start:
        # Effective number reweighting (Cui et al., CVPR 2019)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_priors * 1e6)  # treat as count
        weight = (1.0 - beta) / effective_num
        weight = weight / weight.sum() * num_classes
        weight = weight.to(device)

    return LDAMLoss(class_priors.to(device), max_m=max_m, s=s, weight=weight)


# ===================== GALA: Gradient-Aware Logit Adjustment =====================
class GALALoss(nn.Module):
    """GALA (Gradient-Aware Logit Adjustment, 2024).

    Approximation: track running gradient norm per class as a proxy for difficulty,
    then adjust logits by tau * log(pi) - eta * grad_norm_k.
    Simplified version for our paper - we use accumulated mis-classification rate.
    """

    def __init__(self, class_priors: torch.Tensor, num_classes: int,
                 tau: float = 1.0, eta: float = 0.5, momentum: float = 0.9):
        super().__init__()
        self.tau = tau
        self.eta = eta
        self.momentum = momentum
        self.register_buffer("log_priors", torch.log(class_priors + 1e-8))
        self.register_buffer("err_rate", torch.ones(num_classes) * 0.5)

    def forward(self, logits, targets):
        # Update running err rate (per-class miscls rate)
        with torch.no_grad():
            preds = logits.argmax(1)
            for k in targets.unique():
                mask = (targets == k)
                if mask.sum() > 0:
                    err = (preds[mask] != k).float().mean()
                    self.err_rate[k] = self.momentum * self.err_rate[k] + (1 - self.momentum) * err

        # Adjusted logits: prior shift + difficulty shift
        shift = self.tau * self.log_priors + self.eta * self.err_rate.log().clamp(min=-5)
        adjusted = logits + shift
        return F.cross_entropy(adjusted, targets)


def gala_loss(class_priors, device, num_classes, tau: float = 1.0, eta: float = 0.5):
    return GALALoss(class_priors.to(device), num_classes, tau=tau, eta=eta)


# ===================== Loss Builder =====================
def build_loss(method: str, class_priors, device, num_classes: int, **kwargs):
    """Factory function for all baseline losses."""
    method = method.lower()
    if method == "ce":
        return standard_ce()
    elif method == "weighted_ce" or method == "wce":
        return weighted_ce(class_priors, device, num_classes)
    elif method == "focal":
        return focal_loss(class_priors, device, gamma=kwargs.get("gamma", 2.0))
    elif method == "ldam":
        return ldam_drw_loss(class_priors, device, num_classes=num_classes,
                             max_m=kwargs.get("max_m", 0.5), s=kwargs.get("s", 30.0))
    elif method == "ldam_drw":
        # The DRW version needs to be re-instantiated per epoch
        return ldam_drw_loss(class_priors, device, num_classes=num_classes,
                             epoch=kwargs.get("epoch", 0),
                             drw_start=kwargs.get("drw_start", 80),
                             max_m=kwargs.get("max_m", 0.5), s=kwargs.get("s", 30.0))
    elif method == "logit_adjustment" or method == "la":
        return logit_adjustment_loss(class_priors, device, tau=kwargs.get("tau", 1.0))
    elif method == "gala":
        return gala_loss(class_priors, device, num_classes,
                         tau=kwargs.get("tau", 1.0), eta=kwargs.get("eta", 0.5))
    else:
        raise ValueError(f"Unknown loss method: {method}")

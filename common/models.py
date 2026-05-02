"""Backbone networks: Swin-T and ResNet50."""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights, Swin_T_Weights


class SwinTBackbone(nn.Module):
    feat_dim = 768

    def __init__(self):
        super().__init__()
        m = tv_models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Identity()
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)


class ResNet50Backbone(nn.Module):
    feat_dim = 2048

    def __init__(self):
        super().__init__()
        m = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        return self.backbone(x).flatten(1)


def build_backbone(name: str = "swin_t") -> nn.Module:
    if name == "swin_t":
        return SwinTBackbone()
    elif name == "resnet50":
        return ResNet50Backbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")


class ClassifierHead(nn.Module):
    """3-layer MLP classifier head."""

    def __init__(self, in_dim: int = 768, num_classes: int = 21,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BiddingNetwork(nn.Module):
    """Single expert's bidding network: features -> scalar bid."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

"""
Grad-CAM visualization for the 3 experts in our Bidding Game framework.

For a set of representative cell images (one per class), generate side-by-side:
  Original | Expert A CAM | Expert B CAM | Expert C CAM

Loads the trained main model checkpoints from `run_main.py`.

Usage:
    python -m experiments.run_gradcam --run_name mll_swin_t_game
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, MLL_DIR, PBC_DIR
from common.models import build_backbone, ClassifierHead
from common.data import safe_loader, get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True, help="e.g. mll_swin_t_game")
    p.add_argument("--dataset", default="mll", choices=["mll", "pbc"])
    p.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnet50"])
    p.add_argument("--num_classes", type=int, default=21)
    p.add_argument("--samples_per_class", type=int, default=2,
                   help="How many cells per class to visualize")
    return p.parse_args()


# ===================== Grad-CAM impl =====================
class GradCAM:
    """Generic Grad-CAM. Supports both ResNet (last conv block) and Swin-T (last stage)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward(retain_graph=True)

        # activations: depends on layer type
        # For Swin-T blocks: (B, H, W, C) - channels last
        # For ResNet: (B, C, H, W) - channels first
        act = self.activations
        grad = self.gradients

        # Detect layout
        if act.dim() == 4 and act.shape[-1] != act.shape[1]:
            # Likely channels-last (Swin-T): (B, H, W, C)
            weights = grad.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
            cam = (weights * act).sum(dim=-1)  # (B, H, W)
        else:
            # Channels-first (ResNet): (B, C, H, W)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1)  # (B, H, W)

        cam = F.relu(cam)
        # Normalize to [0, 1]
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()

    def close(self):
        for h in self.hooks:
            h.remove()


# ===================== Build expert wrappers =====================
class BackboneHeadWrapper(torch.nn.Module):
    """Combine backbone + a single expert head into one model for Grad-CAM."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        return self.head(self.backbone(x))


def find_target_layer(backbone):
    """Pick the last conv/stage layer for Grad-CAM."""
    if hasattr(backbone, "backbone"):
        m = backbone.backbone
    else:
        m = backbone

    # Swin-T: features is a Sequential of stages
    if hasattr(m, "features"):
        # Last stage
        return m.features[-1]

    # ResNet50 backbone wraps everything in Sequential
    if isinstance(m, torch.nn.Sequential):
        # Find last conv-like block (layer4 in ResNet)
        # m[7] is layer4 in ResNet50
        return m[-2]  # before avgpool

    raise ValueError("Could not auto-detect target layer for Grad-CAM")


# ===================== Main =====================
def main():
    args = parse_args()
    save_dir = os.path.join(RESULTS_DIR, args.run_name, "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    # Load backbone
    backbone = build_backbone(args.backbone).to(DEVICE)
    backbone.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"{args.run_name}_backbone.pth"),
        map_location=DEVICE, weights_only=True))
    backbone.eval()

    # Load 3 experts
    experts = []
    for i in range(3):
        h = ClassifierHead(in_dim=backbone.feat_dim, num_classes=args.num_classes).to(DEVICE)
        h.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, f"{args.run_name}_expert_{i}.pth"),
            map_location=DEVICE, weights_only=True))
        h.eval()
        experts.append(h)

    expert_names = ["Expert A (Head specialist)", "Expert B (Tail specialist)",
                    "Expert C (Extreme tail)"]
    target_layer = find_target_layer(backbone)

    # Pick representative images
    data_dir = MLL_DIR if args.dataset == "mll" else PBC_DIR
    class_dirs = sorted(os.listdir(data_dir))
    _, val_tf = get_transforms()

    print(f"Found {len(class_dirs)} classes in {data_dir}")

    for cls_name in class_dirs:
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        files = sorted([f for f in os.listdir(cls_path)
                        if f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))])
        if not files:
            continue
        chosen = files[:args.samples_per_class]

        for fname in chosen:
            img_path = os.path.join(cls_path, fname)
            img_pil = safe_loader(img_path)
            img_resized = img_pil.resize((224, 224))
            x = val_tf(img_pil).unsqueeze(0).to(DEVICE)
            class_idx = torch.tensor([class_dirs.index(cls_name)], device=DEVICE)

            # Generate CAM for each expert
            cams = []
            for h in experts:
                wrapper = BackboneHeadWrapper(backbone, h).to(DEVICE)
                cam_obj = GradCAM(wrapper, target_layer)
                cam = cam_obj(x, class_idx)[0]  # (H, W)
                cam_obj.close()
                cams.append(cam)

            # Plot: original + 3 CAMs
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_resized)
            axes[0].set_title(f"Original ({cls_name})")
            axes[0].axis("off")

            for i, (name, cam) in enumerate(zip(expert_names, cams)):
                cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))
                axes[i + 1].imshow(img_resized)
                axes[i + 1].imshow(cam_resized, cmap="jet", alpha=0.5)
                axes[i + 1].set_title(name, fontsize=10)
                axes[i + 1].axis("off")

            plt.tight_layout()
            out_path = os.path.join(save_dir, f"{cls_name}_{fname}.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_path}")

    print(f"\nGrad-CAM visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()

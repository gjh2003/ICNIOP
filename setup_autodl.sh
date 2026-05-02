#!/bin/bash
# AutoDL one-shot setup script.
#
# Usage on AutoDL:
#   chmod +x setup_autodl.sh
#   ./setup_autodl.sh

set -e

echo "=" * 60
echo "  AutoDL Environment Setup"
echo "=" * 60

# 1. GPU check
echo ""
echo "[1/4] Checking GPU..."
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('  CUDA OK:', torch.cuda.get_device_name(0)); print('  PyTorch:', torch.__version__); print('  CUDA version:', torch.version.cuda)"

# 2. Install dependencies
echo ""
echo "[2/4] Installing Python packages..."
pip install -q scikit-learn matplotlib seaborn scipy tqdm pillow

# 3. Download Swin-T pretrained weights
echo ""
echo "[3/4] Downloading Swin-T weights..."
mkdir -p /root/.cache/torch/hub/checkpoints
SWIN_PATH=/root/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth

if [ ! -f "$SWIN_PATH" ]; then
    # Enable AutoDL academic acceleration
    if [ -f /etc/network_turbo ]; then
        source /etc/network_turbo
    fi
    wget -q --show-progress -O "$SWIN_PATH" \
        https://download.pytorch.org/models/swin_t-704ceda3.pth
    echo "  Downloaded Swin-T weights to $SWIN_PATH"
else
    echo "  Swin-T weights already exist."
fi

# 4. Verify directory structure
echo ""
echo "[4/4] Verifying directories..."
echo "  Code root: $(pwd)"
echo "  Looking for BMC_dataset..."

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp}
echo "  DATA_ROOT = $DATA_ROOT"

if [ ! -d "$DATA_ROOT/BMC_dataset" ]; then
    echo "  [WARN] $DATA_ROOT/BMC_dataset not found!"
    echo "  Please upload your dataset to $DATA_ROOT/BMC_dataset/"
    echo "  Expected structure:"
    echo "    $DATA_ROOT/BMC_dataset/MLL/bone_marrow_cell_dataset/<class>/<image>.tif"
    echo "    $DATA_ROOT/BMC_dataset/PBC_dataset_normal_DIB/<class>/<image>.jpg"
else
    echo "  Found BMC_dataset/"
    if [ -d "$DATA_ROOT/BMC_dataset/MLL/bone_marrow_cell_dataset" ]; then
        N_MLL=$(find "$DATA_ROOT/BMC_dataset/MLL/bone_marrow_cell_dataset" -type f \( -name "*.tif" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
        echo "  MLL: $N_MLL images"
    fi
    if [ -d "$DATA_ROOT/BMC_dataset/PBC_dataset_normal_DIB" ]; then
        N_PBC=$(find "$DATA_ROOT/BMC_dataset/PBC_dataset_normal_DIB" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
        echo "  PBC: $N_PBC images"
    fi
fi

echo ""
echo "=" * 60
echo "  Setup complete!"
echo "=" * 60
echo ""
echo "Recommended environment variables:"
echo "  export DATA_ROOT=$DATA_ROOT"
echo "  export OUTPUT_ROOT=/root/autodl-tmp/code_gjh"
echo ""
echo "To run experiments, see run_all.sh"

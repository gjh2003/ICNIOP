"""Path configuration with environment variable overrides for AutoDL/local switching.

Usage:
    # On AutoDL
    export DATA_ROOT=/root/autodl-tmp
    export OUTPUT_ROOT=/root/autodl-tmp/code_gjh

    # On local
    (no env var needed, defaults to project root)
"""

import os

# Project root (where the code lives)
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory: where BMC_dataset/ lives. On AutoDL set DATA_ROOT=/root/autodl-tmp
DATA_ROOT = os.environ.get("DATA_ROOT", CODE_ROOT)

# Output directory: where results/ and checkpoints/ go. On AutoDL set to data disk.
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", CODE_ROOT)

# Specific paths
MLL_DIR = os.path.join(DATA_ROOT, "BMC_dataset", "MLL", "bone_marrow_cell_dataset")
PBC_DIR = os.path.join(DATA_ROOT, "BMC_dataset", "PBC_dataset_normal_DIB")

CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def print_paths():
    print(f"  CODE_ROOT     = {CODE_ROOT}")
    print(f"  DATA_ROOT     = {DATA_ROOT}")
    print(f"  OUTPUT_ROOT   = {OUTPUT_ROOT}")
    print(f"  MLL_DIR       = {MLL_DIR}")
    print(f"  PBC_DIR       = {PBC_DIR}")
    print(f"  CHECKPOINT    = {CHECKPOINT_DIR}")
    print(f"  RESULTS       = {RESULTS_DIR}")

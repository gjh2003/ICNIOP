#!/bin/bash
# Run all experiments for the paper.
# All commands use --resume so a restart picks up where it left off.
#
# Estimated total: ~60-70 hours on RTX 5090
# Estimated cost:  ~¥200 (¥3/h)
#
# Usage:
#   nohup ./run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log

set -e

# ============== Tier 0: Main + Core Baselines ==============
#
# IMPORTANT: After the architecture redesign (Expert A=P1, B=cRT, C=PostHocLA),
# Phase 1 is reused across ALL Phase 2/3 variants. The first run trains
# backbone+P1 head (~3-4h on 5090). Subsequent ablations should use --skip_p1
# to save 3-4h each.
#
# The baseline `run_baseline.py --method ce` produces THE SAME backbone training
# as run_main's Phase 1, so we run it FIRST and let run_main reuse its checkpoint.
# Actually no — they save under different names. We just train run_main first.

echo "[Tier 0.1] MLL Main Method (trains P1 backbone + builds 3 derived experts + bidding game)..."
python -m experiments.run_main --dataset mll --backbone swin_t --resume

echo "[Tier 0.2] MLL Baselines (each trains its own backbone independently for fair comparison)..."
python -m experiments.run_baseline --dataset mll --method ce --resume
python -m experiments.run_baseline --dataset mll --method weighted_ce --resume
python -m experiments.run_baseline --dataset mll --method focal --resume
python -m experiments.run_baseline --dataset mll --method ldam_drw --resume
python -m experiments.run_baseline --dataset mll --method logit_adjustment --resume
python -m experiments.run_ride --dataset mll --resume
python -m experiments.run_sade --dataset mll --resume

echo "[Tier 0.3] PBC-LT..."
python -m experiments.run_main --dataset pbc --variant lt --imb_factor 100 --resume
python -m experiments.run_baseline --dataset pbc --variant lt --imb_factor 100 --method ce --resume
python -m experiments.run_ride --dataset pbc --variant lt --imb_factor 100 --resume
python -m experiments.run_sade --dataset pbc --variant lt --imb_factor 100 --resume

# ============== Tier 1: GALA + Ablations ==============

echo "[Tier 1.1] GALA baseline..."
python -m experiments.run_baseline --dataset mll --method gala --resume

echo "[Tier 1.2] Ablation: Logit Adjustment tau sensitivity..."
python -m experiments.run_baseline --dataset mll --method logit_adjustment --la_tau 0.5 --resume
python -m experiments.run_baseline --dataset mll --method logit_adjustment --la_tau 2.0 --resume

echo "[Tier 1.3] Ablation: Asymmetric cost mechanism..."
# Symmetric costs (test if asymmetry is necessary)
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --expert_costs 0.1,0.1,0.1 --resume
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --expert_costs 0.5,0.5,0.5 --resume
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --expert_costs 1.0,1.0,1.0 --resume

# Reversed asymmetry (head expert most expensive)
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --expert_costs 1.0,0.5,0.1 --resume

# Learnable costs (let the model decide)
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --learnable_costs --cost_init 0.5,0.5,0.5 --resume

echo "[Tier 1.4] Ablation: Game loss weight alpha..."
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --p3_alpha 0.0 --resume
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --p3_alpha 0.1 --resume
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --p3_alpha 1.0 --resume
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --p3_alpha 2.0 --resume

# ============== Tier 2: ResNet50 + Visualization ==============

echo "[Tier 2.1] ResNet50 backbone ablation..."
python -m experiments.run_baseline --dataset mll --backbone resnet50 --method ce --resume
python -m experiments.run_ride --dataset mll --backbone resnet50 --resume
python -m experiments.run_sade --dataset mll --backbone resnet50 --resume
python -m experiments.run_main --dataset mll --backbone resnet50 --resume

echo "[Tier 2.2] Grad-CAM visualization..."
python -m experiments.run_gradcam --run_name mll_swin_t_game

# ============== Summary ==============

echo "[Final] Generating summary tables..."
python -m experiments.summarize_results

echo "All experiments complete!"
echo "See results/SUMMARY.md and results/SUMMARY.tex"

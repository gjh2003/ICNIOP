#!/bin/bash
# Migration script: clean up checkpoints from the OLD broken expert design
# and prepare for the new design (Expert A=P1, B=cRT, C=PostHocLA).
#
# Run this on AutoDL after `git pull` and BEFORE restarting experiments.
#
# Keeps:
#   - P1 backbone + head (already trained, ~3-4 hours of work)
#   - All baseline experiment results (run_baseline.py outputs)
#
# Removes:
#   - Old expert heads (will be regenerated under new design)
#   - Old bidders (must retrain with new experts)
#   - Old resume checkpoints for P2/P3 (incompatible with new code)

set -e

CKPT=${OUTPUT_ROOT:-/root/autodl-tmp/ICNIOP_output}/checkpoints
RUN=mll_swin_t_game

echo "Cleaning up incompatible checkpoints in $CKPT..."

# Show what we have
echo ""
echo "Existing files for $RUN:"
ls -lh $CKPT/${RUN}_*.pth 2>/dev/null || echo "  (none)"

echo ""
echo "Will KEEP:"
echo "  - ${RUN}_backbone.pth (P1 backbone, ~110MB)"
echo "  - ${RUN}_p1_head.pth (P1 head, ~2MB)"
echo "  - ${RUN}_p1_resume.pth (P1 resume, allows --resume to skip retraining)"
echo ""
echo "Will REMOVE (incompatible with new design):"
for f in \
    ${RUN}_expert_0.pth \
    ${RUN}_expert_1.pth \
    ${RUN}_expert_2.pth \
    ${RUN}_bidder_0.pth \
    ${RUN}_bidder_1.pth \
    ${RUN}_bidder_2.pth \
    ${RUN}_p2_ExpertA_resume.pth \
    ${RUN}_p2_ExpertB_resume.pth \
    ${RUN}_p2_ExpertC_resume.pth \
    ${RUN}_p3_resume.pth \
    ${RUN}_costs.pth; do
    if [ -f "$CKPT/$f" ]; then
        echo "  $f"
    fi
done

echo ""
read -p "Proceed? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 1
fi

# Remove
for f in \
    ${RUN}_expert_0.pth \
    ${RUN}_expert_1.pth \
    ${RUN}_expert_2.pth \
    ${RUN}_bidder_0.pth \
    ${RUN}_bidder_1.pth \
    ${RUN}_bidder_2.pth \
    ${RUN}_p2_ExpertA_resume.pth \
    ${RUN}_p2_ExpertB_resume.pth \
    ${RUN}_p2_ExpertC_resume.pth \
    ${RUN}_p3_resume.pth \
    ${RUN}_costs.pth; do
    rm -f "$CKPT/$f"
done

# Also clean old results for the main run (will be regenerated)
RESULTS=${OUTPUT_ROOT:-/root/autodl-tmp/ICNIOP_output}/results
echo ""
echo "Cleaning old results for $RUN..."
rm -rf "$RESULTS/$RUN"

echo ""
echo "Done. Now restart with:"
echo ""
echo "  cd /root/ICNIOP"
echo "  python -m experiments.run_main --dataset mll --backbone swin_t --skip_p1"
echo ""
echo "  (--skip_p1 reuses your already-trained P1 backbone + head)"
echo ""
echo "Or to run the full sweep (run_all.sh), first kill any old runs:"
echo "  pkill -f run_all.sh"
echo "  pkill -f run_main.py"
echo "  pkill -f run_baseline.py"
echo "Then:"
echo "  nohup ./run_all.sh > run_all.log 2>&1 &"

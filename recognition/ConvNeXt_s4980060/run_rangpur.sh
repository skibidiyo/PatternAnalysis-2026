#!/bin/bash
#SBATCH --job-name=convnext-adni
#SBATCH --partition=comp3710        # the course's A100 GPU nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train the ADNI AD vs CN classifier on the Rangpur HPC cluster.
#
#   sbatch run_rangpur.sh convnext-pretrained   # the headline model
#   sbatch run_rangpur.sh baseline              # the CNN baseline
#   sbatch run_rangpur.sh convnext              # ConvNeXt from scratch
#   sbatch run_rangpur.sh convnext-mini         # smaller ConvNeXt from scratch
#
# Every model must be trained before the comparison in the README can be made;
# they share a seed and therefore an identical patient-level split.
#
# convnext-pretrained needs the ImageNet weights cached before the job starts,
# because compute nodes have no outbound network access. Rangpur points
# TORCH_HOME at /cache, which is not guaranteed to be the same filesystem on
# the compute node, so both the pre-download and the job below pin it to $HOME.
# Run this once on the login node:
#
#   TORCH_HOME=$HOME/.cache/torch python -c \
#     "from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights; \
#      convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)"
#
# Author: s4980060

set -euo pipefail

MODEL="${1:-convnext-pretrained}"
DATA_ROOT="${DATA_ROOT:-/home/groups/comp3710/ADNI/AD_NC}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

# Fine-tuning converges in far fewer epochs and needs a much smaller learning
# rate than training from scratch; using the from-scratch recipe on pretrained
# weights destroys them in the first few steps.
case "$MODEL" in
    convnext-pretrained)
        EPOCHS="${EPOCHS:-12}"
        LR="${LR:-5e-5}"
        ;;
    *)
        EPOCHS="${EPOCHS:-30}"
        LR="${LR:-3e-4}"
        ;;
esac

mkdir -p logs outputs assets

echo "Job $SLURM_JOB_ID on $(hostname), model=$MODEL epochs=$EPOCHS lr=$LR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# A batch job starts from a clean environment, so conda must be activated here
# rather than relied on from the submitting shell.
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Pin the weight cache to home. The cluster default (/cache) may be node-local,
# in which case the weights downloaded on the login node would be invisible here
# and torchvision would try to re-download them without network access.
export TORCH_HOME="$HOME/.cache/torch"

# The data loader resolves real patient ids from meta_data_with_label.json,
# which it expects beside the AD_NC directory. Fail early with a clear message
# rather than part-way through training if the layout differs.
METADATA="$(dirname "$DATA_ROOT")/meta_data_with_label.json"
if [[ ! -f "$METADATA" ]]; then
    echo "ERROR: metadata not found at $METADATA" >&2
    echo "Patient-level splitting cannot be verified without it; pass the correct" >&2
    echo "path via --metadata in the python call below." >&2
    exit 1
fi

# Same idea for the pretrained weights: a missing cache would otherwise surface
# as a network timeout several minutes into the job.
if [[ "$MODEL" == "convnext-pretrained" ]]; then
    CACHE="$TORCH_HOME/hub/checkpoints"
    if ! ls "$CACHE"/convnext_tiny-*.pth >/dev/null 2>&1; then
        echo "ERROR: ImageNet weights not cached in $CACHE" >&2
        echo "Compute nodes cannot reach the internet. Pre-download them on the" >&2
        echo "login node using the command in this script's header." >&2
        exit 1
    fi
fi

python -u train.py \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --out-dir outputs \
    --assets-dir assets

python -u predict.py \
    --checkpoint "outputs/${MODEL}_best.pth" \
    --data-root "$DATA_ROOT" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --out-dir outputs \
    --assets-dir assets

echo "Done. Checkpoint and metrics in outputs/, figures in assets/"

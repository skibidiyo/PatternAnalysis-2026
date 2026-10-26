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
#   sbatch run_rangpur.sh convnext      # train ConvNeXt (default)
#   sbatch run_rangpur.sh baseline      # train the CNN baseline
#
# Both models must be trained before the comparison in the README can be made;
# they share a seed and therefore an identical patient-level split.
#
# Author: s4980060

set -euo pipefail

MODEL="${1:-convnext}"
DATA_ROOT="/home/groups/comp3710/ADNI/AD_NC"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

mkdir -p logs outputs assets

echo "Job $SLURM_JOB_ID on $(hostname), model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# A batch job starts from a clean environment, so conda must be activated here
# rather than relied on from the submitting shell.
source "$HOME/miniconda3/bin/activate"
conda activate torch

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

python -u train.py \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
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

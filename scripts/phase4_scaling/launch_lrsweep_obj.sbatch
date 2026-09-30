#!/bin/bash
#SBATCH --job-name=tgv_lrsweep
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/lrsweep_%A_%a.out
#SBATCH --error=logs/lrsweep_%A_%a.err
#
# Stage 2b: LR tuning for the 3 NON-MLM objectives on event-only data, seed 123.
# 3 objectives x 4 LRs = 12 tasks. MLM is the anchor (from the Stage-2a baseline).
# Pick the best LR per objective before the full seed x feature grid.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
OBJS=(autoregressive contrastive tversky)
LRS=(1e-4 3e-4 5e-4 1e-3)
i=$SLURM_ARRAY_TASK_ID
OBJ=${OBJS[$((i / 4))]}
LR=${LRS[$((i % 4))]}
export REPR_TAG=M_${OBJ}_lr${LR}_s123
export REPR_CAPACITY=M
export REPR_VARIANT=holdout
export REPR_SEED=123
export REPR_OBJECTIVE=$OBJ
export REPR_LR=$LR
export REPR_MAX_EPOCHS=12
export REPR_PATIENCE=6
export REPR_WARMUP_STEPS=2000
export REPR_CLIP=0.5
export REPR_BATCH=256
echo "[lrsweep] task=$i OBJ=$OBJ LR=$LR TAG=$REPR_TAG"
python scripts/phase4_scaling/train_repr_obj.py
echo "LRSWEEP_DONE_${OBJ}_${LR}"

#!/bin/bash
#SBATCH --job-name=tgv_lrtune
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --output=logs/lrtune_%A_%a.out
#SBATCH --error=logs/lrtune_%A_%a.err
#
# LR/schedule tuning for the under-optimized big models (L, XL), seed 123, holdout variant.
#
# WHY: the live trainer already does warmup->cosine decay, but the cosine horizon is
#   _tot = REPR_MAX_EPOCHS * steps_per_epoch with REPR_MAX_EPOCHS=60. The sweep runs
#   early-stopped (patience 8) by ~epoch 15, where cosine has only reached pr=0.25
#   -> LR still ~87% of peak the whole run. Result: XL plateaued at val 1.014 (WORSE
#   than S 1.003) and never realized its capacity.
#
# FIX (all env-driven, NO trainer edit):
#   - REPR_MAX_EPOCHS=30  -> cosine actually decays to its 0.1 floor within the run.
#   - REPR_PATIENCE=15    -> don't early-stop on the high-LR plateau before decay helps.
#   - sweep REPR_LR in {1e-3, 5e-4, 3e-4}: the 1e-3 cell isolates "did fixing the
#     horizon alone fix it"; the lower LRs test whether the peak was also too hot.
#
# Distinct TAGs -> separate checkpoints/logs; does NOT touch the existing sweep models.
# Compare afterward on repr_scaling_log_<TAG>.csv val curves, then transfer-eval the winners.
set -euo pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate

CAPS=(L XL)
LRS=(1e-3 5e-4 3e-4)
LRNAMES=(1e3 5e4 3e4)

i=${SLURM_ARRAY_TASK_ID}
c=$(( i / 3 ))
l=$(( i % 3 ))

CAP=${CAPS[$c]}
LR=${LRS[$l]}
LRNAME=${LRNAMES[$l]}
SEED=123

case $CAP in
  L)  BATCH=128 ;;
  XL) BATCH=64  ;;
esac

TAG=${CAP}_holdout_tune_lr${LRNAME}_s${SEED}

export REPR_TAG=$TAG
export REPR_CAPACITY=$CAP
export REPR_VARIANT=holdout
export REPR_SEED=$SEED
export REPR_SAMPLE_WEIGHT=conflict
export REPR_BUFFER_DEG=0.0
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=15
export REPR_BATCH=$BATCH
export REPR_LR=$LR
export REPR_MAX_ROWS=0
export REPR_WARMUP_STEPS=500
export REPR_CLIP=1.0

echo "[lrtune] task=$i TAG=$TAG CAP=$CAP LR=$LR BATCH=$BATCH max_ep=$REPR_MAX_EPOCHS patience=$REPR_PATIENCE host=$(hostname)"
python scripts/phase4_scaling/train_repr_scaling.py
echo "[lrtune] task=$i TAG=$TAG DONE"

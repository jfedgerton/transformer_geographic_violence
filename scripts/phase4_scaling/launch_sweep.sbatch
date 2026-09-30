#!/bin/bash
#SBATCH --job-name=tgv_scale
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err
#
# Phase 4 scaling-transfer pretraining grid.
#   variants x capacities x seeds = 2 x 4 x 3 = 24 array tasks (0-23).
#   variant 0 = holdout (global - LatAm/ME - 8 stratified countries) -> transfer target
#   variant 1 = africa  (Africa-bbox specialist baseline)
# Walltime is 48h/chunk; trainer checkpoints every epoch and resumes on requeue,
# so an unconverged L/XL run just gets resubmitted with the same TAG and continues.
set -euo pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate

VARIANTS=(holdout africa)
CAPS=(S M L XL)
SEEDS=(123 456 789)

i=${SLURM_ARRAY_TASK_ID}
v=$(( i / 12 ))
rem=$(( i % 12 ))
c=$(( rem / 3 ))
s=$(( rem % 3 ))

VARIANT=${VARIANTS[$v]}
CAP=${CAPS[$c]}
SEED=${SEEDS[$s]}
TAG=${CAP}_${VARIANT}_s${SEED}

case $CAP in
  S)  BATCH=256; LR=1e-3 ;;
  M)  BATCH=256; LR=1e-3 ;;
  L)  BATCH=128; LR=3e-4 ;;
  XL) BATCH=64;  LR=2e-4 ;;
esac

export REPR_TAG=$TAG
export REPR_CAPACITY=$CAP
export REPR_VARIANT=$VARIANT
export REPR_SEED=$SEED
export REPR_SAMPLE_WEIGHT=conflict
export REPR_BUFFER_DEG=0.0
export REPR_MAX_EPOCHS=60
export REPR_WARMUP_STEPS=500
export REPR_CLIP=1.0
export REPR_PATIENCE=8
export REPR_BATCH=$BATCH
export REPR_LR=$LR
export REPR_MAX_ROWS=0

echo "[launch] task=$i TAG=$TAG CAP=$CAP VARIANT=$VARIANT SEED=$SEED BATCH=$BATCH host=$(hostname)"
python scripts/phase4_scaling/train_repr_scaling.py
echo "[launch] task=$i TAG=$TAG DONE"

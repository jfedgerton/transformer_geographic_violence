#!/bin/bash
#SBATCH --job-name=tgv_objsmoke
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/objsmoke_%A_%a.out
#SBATCH --error=logs/objsmoke_%A_%a.err
#
# Smoke test: each of the 4 objectives, 1 epoch on a 20k-row subset, to verify
# train_repr_obj.py runs end-to-end and produces finite loss before the real sweep.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
OBJS=(mlm autoregressive contrastive tversky)
OBJ=${OBJS[$SLURM_ARRAY_TASK_ID]}
export REPR_TAG=smoke_${OBJ}
export REPR_CAPACITY=M
export REPR_VARIANT=holdout
export REPR_SEED=123
export REPR_OBJECTIVE=$OBJ
export REPR_MAX_ROWS=20000
export REPR_MAX_EPOCHS=1
export REPR_WARMUP_STEPS=50
export REPR_PATIENCE=1
echo "[smoke] OBJ=$OBJ"
python scripts/phase4_scaling/train_repr_obj.py
echo "SMOKE_DONE_${OBJ}"

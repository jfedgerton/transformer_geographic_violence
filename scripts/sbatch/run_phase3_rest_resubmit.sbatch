#!/bin/bash
#SBATCH --job-name=tgv_p3_rest
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_rest_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_rest_%A_%a.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
SEEDS=(456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export PHASE_MODE=global
export PYTHONUNBUFFERED=1
module load anaconda
source venv/bin/activate
echo "[$(date)] Phase 3 seed ${REPR_SEED} RESUBMIT (48h, unbuffered). Skipping prep; global artifacts already on disk."
echo "[$(date)] Step 06: MLM pre-training on GLOBAL data (seed=${REPR_SEED})"
python scripts/06_train_representation_model.py
echo "[$(date)] Step 07: Downstream UCDP target (seed=${REPR_SEED})"
python scripts/07_train_downstream_model.py
echo "[$(date)] DONE seed ${REPR_SEED}"

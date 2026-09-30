#!/bin/bash
#SBATCH --job-name=tgv_p3_d128
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --array=0-4
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_d128_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_d128_%A_%a.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
SEEDS=(123 456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export REPR_EMBED_DIM=128
export PHASE_MODE=global
export PYTHONUNBUFFERED=1
module load anaconda
source venv/bin/activate
echo "[$(date)] Phase3 D128 seed=${REPR_SEED} embed_dim=128; skip prep (global artifacts on disk)"
echo "[$(date)] Step06 MLM pretrain GLOBAL embed_dim=128"
python scripts/06_train_representation_model.py
echo "[$(date)] Step07 downstream UCDP"
python scripts/07_train_downstream_model.py
echo "[$(date)] DONE D128 seed=${REPR_SEED}"

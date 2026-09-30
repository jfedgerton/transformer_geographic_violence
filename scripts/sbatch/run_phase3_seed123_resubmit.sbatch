#!/bin/bash
#SBATCH --job-name=tgv_p3_s123
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_s123_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_s123_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
export REPR_SEED=123
export PHASE_MODE=global
export PYTHONUNBUFFERED=1
module load anaconda
source venv/bin/activate
echo "[$(date)] Phase 3 seed 123 RESUBMIT (48h walltime, unbuffered). Skipping prep; global artifacts already on disk."
echo "[$(date)] Step 06: MLM pre-training on GLOBAL data (seed=123)"
python scripts/06_train_representation_model.py
echo "[$(date)] Step 07: Downstream UCDP target (seed=123)"
python scripts/07_train_downstream_model.py
echo "[$(date)] DONE seed 123"

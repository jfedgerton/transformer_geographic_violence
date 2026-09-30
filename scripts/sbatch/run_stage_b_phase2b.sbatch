#!/bin/bash
#SBATCH --job-name=tgv_phase2b
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2b_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2b_%A_%a.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

SEEDS=(123 456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Phase 2b task ${SLURM_ARRAY_TASK_ID}, seed=${REPR_SEED}"
echo "[$(date)] Node: $(hostname)"

module load anaconda
source venv/bin/activate

# Sanity check: Phase 2a embeddings must exist for this seed
EMB_PATH=data_model/embeddings_all_seed${REPR_SEED}.parquet
if [ ! -f "$EMB_PATH" ]; then
  echo "ERROR: Phase 2a embeddings not found at $EMB_PATH"
  echo "Run Phase 2a first."
  exit 1
fi

echo "[$(date)] Step 06b: cross-attention training (seed=${REPR_SEED})"
python scripts/06b_train_crossattn.py

echo "[$(date)] Step 07b: downstream with crossattn embeddings (seed=${REPR_SEED})"
python scripts/07b_downstream_crossattn.py

echo "[$(date)] Phase 2b task ${SLURM_ARRAY_TASK_ID} DONE"

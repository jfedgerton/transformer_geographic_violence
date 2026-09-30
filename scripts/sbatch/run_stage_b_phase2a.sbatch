#!/bin/bash
#SBATCH --job-name=tgv_phase2a
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2a_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2a_%A_%a.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

# Map array index to seed
SEEDS=(123 456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Phase 2a task ${SLURM_ARRAY_TASK_ID}, seed=${REPR_SEED}"
echo "[$(date)] Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

module load anaconda
source venv/bin/activate

# Steps 01-05 are shared across seeds; run only on task 0 to avoid races
# (downstream tasks wait for these via a small wait loop)
SENTINEL=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/data_intermediate/.phase2a_prep_done

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "[$(date)] Task 0: running steps 01-05 (shared prep)"
  rm -f "$SENTINEL"
  python scripts/01_ingest_acled.py
  python scripts/01b_ingest_ucdp.py
  python scripts/01c_merge_sources.py
  python scripts/02_build_grid.py
  python scripts/03_aggregate_location_month.py
  python scripts/04_build_sequences.py
  python scripts/05_build_baseline_features.py
  touch "$SENTINEL"
  echo "[$(date)] Task 0: prep done, sentinel written"
else
  echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: waiting for sentinel..."
  for i in $(seq 1 60); do
    if [ -f "$SENTINEL" ]; then break; fi
    sleep 30
  done
  if [ ! -f "$SENTINEL" ]; then
    echo "ERROR: sentinel not found after 30 minutes" >&2
    exit 1
  fi
fi

echo "[$(date)] Step 06: MLM pre-training (seed=${REPR_SEED})"
python scripts/06_train_representation_model.py

echo "[$(date)] Step 07: Downstream training with UCDP target (seed=${REPR_SEED})"
python scripts/07_train_downstream_model.py

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} (seed=${REPR_SEED}) DONE"

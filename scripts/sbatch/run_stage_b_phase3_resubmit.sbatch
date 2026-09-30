#!/bin/bash
#SBATCH --job-name=tgv_phase3
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=1-4
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase3_%A_%a.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

SEEDS=(123 456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export PHASE_MODE=global

echo "[$(date)] Phase 3 task ${SLURM_ARRAY_TASK_ID}, seed=${REPR_SEED}, PHASE_MODE=global"
echo "[$(date)] Node: $(hostname)"

module load anaconda
source venv/bin/activate

SENTINEL=data_intermediate/.phase3_prep_done

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "[$(date)] Task 0: full pipeline prep (steps 01-05) for GLOBAL corpus"
  rm -f "$SENTINEL"
  python scripts/01_ingest_acled.py      # global ACLED
  python scripts/01b_ingest_ucdp.py      # global UCDP
  python scripts/01c_merge_sources.py
  python scripts/02_build_grid.py
  python scripts/03_aggregate_location_month.py
  python scripts/04_build_sequences.py
  python scripts/05_build_baseline_features.py
  touch "$SENTINEL"
  echo "[$(date)] Task 0: global prep done, sentinel written"
else
  echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: waiting for sentinel..."
  for i in $(seq 1 360); do
    if [ -f "$SENTINEL" ]; then break; fi
    sleep 30
  done
  if [ ! -f "$SENTINEL" ]; then
    echo "ERROR: sentinel not found after 180 minutes" >&2
    exit 1
  fi
fi

echo "[$(date)] Step 06: MLM pre-training on GLOBAL data (seed=${REPR_SEED})"
python scripts/06_train_representation_model.py

echo "[$(date)] Step 07: Downstream with UCDP target on Africa subset (seed=${REPR_SEED})"
# Note: 07 evaluates on the FULL test set including all countries the model trained on.
# To restrict evaluation to Africa for ViEWS comparison, the eval step needs to filter
# the test set by region; that filter happens post-hoc in 08b ensemble eval if needed.
python scripts/07_train_downstream_model.py

echo "[$(date)] Phase 3 task ${SLURM_ARRAY_TASK_ID} DONE"

#!/bin/bash
#SBATCH --job-name=tgv_p2a_recover
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/p2a_recover_%A_%a.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/p2a_recover_%A_%a.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

SEEDS=(123 456 789 1024 2025)
export REPR_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Phase 2a recovery task ${SLURM_ARRAY_TASK_ID}, seed=${REPR_SEED}"

module load anaconda
source venv/bin/activate

# Sanity: Phase 2a embeddings exist
EMB=data_model/embeddings_all_seed${REPR_SEED}.parquet
if [ ! -f "$EMB" ]; then
  echo "ERROR: missing $EMB"; exit 1
fi
# Sanity: fixed ucdp data has type_of_violence
python -c "import pandas as pd; df = pd.read_parquet('data_intermediate/ucdp_cleaned.parquet'); assert 'type_of_violence' in df.columns, 'type_of_violence still missing'; print('UCDP OK, rows:', len(df))"

echo "[$(date)] Running step 07 (downstream with UCDP target) for seed=${REPR_SEED}"
python scripts/07_train_downstream_model.py
echo "[$(date)] Recovery seed=${REPR_SEED} DONE"

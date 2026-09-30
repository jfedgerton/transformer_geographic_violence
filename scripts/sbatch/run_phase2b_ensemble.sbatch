#!/bin/bash
#SBATCH --job-name=tgv_p2b_ens
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2b_ensemble_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase2b_ensemble_%j.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
echo "[$(date)] Running 08c_ensemble_eval_crossattn.py"
python scripts/08c_ensemble_eval_crossattn.py
echo "[$(date)] Phase 2b ensemble eval complete"

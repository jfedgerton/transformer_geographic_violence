#!/bin/bash
#SBATCH --job-name=tgv_yardstick
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/yardstick_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/yardstick_%j.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
echo "[$(date)] Running co-occurrence yardstick"
python scripts/09_cooccurrence_yardstick.py
echo "[$(date)] Done"

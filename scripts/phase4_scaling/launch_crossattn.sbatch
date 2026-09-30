#!/bin/bash
#SBATCH --job-name=tgv_xattn
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xattn_%j.out
#SBATCH --error=logs/xattn_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
echo "[xattn] SMOKE=${SMOKE:-0}"
python scripts/phase4_scaling/prototype_crossattn.py
echo "XATTN_DONE"

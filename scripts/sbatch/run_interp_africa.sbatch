#!/bin/bash
#SBATCH --job-name=tgv_interp_af
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/interp_africa_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/interp_africa_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
python scripts/10_interpretability_africa.py

#!/bin/bash
#SBATCH --job-name=ens_totals
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ens_totals_%j.out
#SBATCH --error=logs/ens_totals_%j.err
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
md5sum scripts/forecast_ensemble/ens_alerts_totals.py
python scripts/forecast_ensemble/ens_alerts_totals.py

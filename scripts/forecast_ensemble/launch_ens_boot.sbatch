#!/bin/bash
#SBATCH --job-name=ens_boot
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ens_boot_%j.out
#SBATCH --error=logs/ens_boot_%j.err
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
md5sum scripts/forecast_ensemble/ens_boot.py
python scripts/forecast_ensemble/ens_boot.py

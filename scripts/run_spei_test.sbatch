#!/bin/bash
#SBATCH --job-name=spei_test
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/spei_test_%j.out
#SBATCH --error=logs/spei_test_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
python -c "import xarray" 2>/dev/null || pip install --quiet xarray netCDF4
python scripts/06_build_spei_grid.py

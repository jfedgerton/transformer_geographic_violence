#!/bin/bash
#SBATCH --job-name=tgv_brpanel
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/brpanel_%j.out
#SBATCH --error=logs/brpanel_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/build_brazil_panel.py
venv/bin/python scripts/crossdomain/build_brazil_panel.py
echo BRPANEL_DONE

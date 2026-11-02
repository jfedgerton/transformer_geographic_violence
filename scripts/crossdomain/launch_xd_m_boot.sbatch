#!/bin/bash
#SBATCH --job-name=tgv_xdMboot
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/xdMboot_%j.out
#SBATCH --error=logs/xdMboot_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/xd_m_boot.py scripts/crossdomain/xd_grid_brazil_v6.py
venv/bin/python scripts/crossdomain/xd_m_boot.py

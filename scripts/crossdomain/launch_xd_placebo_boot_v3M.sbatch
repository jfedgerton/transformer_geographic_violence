#!/bin/bash
#SBATCH --job-name=tgv_xpbM
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xpbM_%j.out
#SBATCH --error=logs/xpbM_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/xd_placebo_boot_v3M.py
venv/bin/python scripts/crossdomain/xd_placebo_boot_v3M.py

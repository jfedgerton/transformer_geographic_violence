#!/bin/bash
#SBATCH --job-name=tgv_window
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/window_%j.out
#SBATCH --error=logs/window_%j.err
# Ukraine/Sudan onset windows with cell-bootstrap intervals. sbatch --export=ALL,WIN_TARGET=incidence
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${WIN_TARGET:?pass WIN_TARGET=incidence or civilian}"
unset WIN_ROOT WIN_DRAWS WIN_SHARES WIN_B
WIN_TARGET=$WIN_TARGET venv/bin/python scripts/phase4_scaling/country_window.py

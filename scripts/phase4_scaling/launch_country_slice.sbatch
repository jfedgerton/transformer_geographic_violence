#!/bin/bash
#SBATCH --job-name=tgv_slice
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/slice_%j.out
#SBATCH --error=logs/slice_%j.err
# Ukraine/Sudan slices of the share grid. sbatch --export=ALL,SLICE_TARGET=incidence (or civilian, after that grid finishes)
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${SLICE_TARGET:?pass SLICE_TARGET=incidence or civilian}"
unset SLICE_ROOT SLICE_DRAWS SLICE_SHARES SLICE_ALLOW_MISSING
SLICE_TARGET=$SLICE_TARGET venv/bin/python scripts/phase4_scaling/country_slice.py

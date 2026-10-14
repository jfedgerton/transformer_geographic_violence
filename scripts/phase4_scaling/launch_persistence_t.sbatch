#!/bin/bash
#SBATCH --job-name=tgv_pers_t
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/pers_t_%j.out
#SBATCH --error=logs/pers_t_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
venv/bin/python scripts/phase4_scaling/persistence_month_t.py

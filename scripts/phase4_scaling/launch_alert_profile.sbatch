#!/bin/bash
#SBATCH --job-name=tgv_aprof
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/aprof_%j.out
#SBATCH --error=logs/aprof_%j.err
# Profile of the embedding's extra correct alerts (deaths, event types, countries). CPU only.
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
unset AP_ROOT
venv/bin/python scripts/phase4_scaling/alert_profile.py

#!/bin/bash
#SBATCH --job-name=tgv_alerts
#SBATCH --account=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/alerts_%j.out
#SBATCH --error=logs/alerts_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
venv/bin/python scripts/phase4_scaling/alert_counts.py

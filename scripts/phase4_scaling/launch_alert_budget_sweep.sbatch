#!/bin/bash
#SBATCH --job-name=tgv_budget
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/budget_%j.out
#SBATCH --error=logs/budget_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
unset SWEEP_ROOT
venv/bin/python scripts/phase4_scaling/alert_budget_sweep.py

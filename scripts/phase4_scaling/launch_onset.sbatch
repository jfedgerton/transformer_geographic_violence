#!/bin/bash
#SBATCH --job-name=tgv_onset
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=logs/onset_%A_%a.out
#SBATCH --error=logs/onset_%A_%a.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
CASES=(Ukraine Sudan)
export OMP_NUM_THREADS=8
ONSET_COUNTRY=${CASES[$SLURM_ARRAY_TASK_ID]} VOCAB_REMAP=1 venv/bin/python scripts/phase4_scaling/onset_transfer.py

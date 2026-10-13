#!/bin/bash
#SBATCH --array=0-1
#SBATCH --job-name=tgv_gridsmk
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gridsmk_%A_%a.out
#SBATCH --error=logs/gridsmk_%A_%a.err
# Smoke test of share_grid_conflict.py on Europe s123 (the one converged seed-123 encoder on disk).
# Task 0 = nested, task 1 = finetune at 1%. SMOKE=1: 20k test rows, 2 draws, 2 lrs, 3 epochs. Outputs end in _smoke.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(nested finetune)
export EVAL_REGION=europe EVAL_SEED=123 SMOKE=1 GRID_ARM=${ARMS[$SLURM_ARRAY_TASK_ID]} GRID_SHARES=0.01
python scripts/phase4_scaling/share_grid_conflict.py

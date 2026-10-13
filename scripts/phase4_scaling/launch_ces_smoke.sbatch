#!/bin/bash
#SBATCH --array=0-1
#SBATCH --job-name=ces_smk
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ces_smk_%A_%a.out
#SBATCH --error=logs/ces_smk_%A_%a.err
# Smoke grid on New England (DIV 0), seed 123, using the smoke encoder. Submit with --dependency=afterok:<smoke pretrain id>.
# Task 0 = nested, task 1 = finetune at 1%. Outputs end in _smoke.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export DIV=0 SEED=123 SMOKE=1
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then export GRID_ARM=nested; else export GRID_ARM=finetune GRID_SHARES=0.01; fi
python scripts/phase4_scaling/share_grid_ces.py

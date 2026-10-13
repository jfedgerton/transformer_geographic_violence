#!/bin/bash
#SBATCH --array=0-26
#SBATCH --job-name=ces_pre
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ces_pre_%A_%a.out
#SBATCH --error=logs/ces_pre_%A_%a.err
# 9 held-out Census divisions x 3 seeds. Task i: DIV = i % 9, SEED = (123,456,789)[i / 9].
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
SDS=(123 456 789); i=$SLURM_ARRAY_TASK_ID
export DIV=$((i % 9)) SEED=${SDS[$((i / 9))]} SMOKE=0 EPOCHS=30 PATIENCE=5
echo "[task] $i DIV=$DIV SEED=$SEED"
python scripts/phase4_scaling/pretrain_ces.py

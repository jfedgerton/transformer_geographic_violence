#!/bin/bash
#SBATCH --job-name=tgv_tmet
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/tmet_%A_%a.out
#SBATCH --error=logs/tmet_%A_%a.err
#
# 3-metric transfer eval (incidence / onset-lag1 / severity-CRPS) on the M-CONTROL
# encoder, one job per seed. Writes outputs/transfer_metrics_ctrl_s{SEED}.csv.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate

SEEDS=(123 456 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export EVAL_CAP=M
export EVAL_SEED=$SEED
export EVAL_CKPT=$D/data_model/repr_model_M_holdout_pilot_lr5e4_s${SEED}.pt
export EVAL_OUT=$D/outputs/transfer_metrics_ctrl_s${SEED}.csv

echo "[tmet] seed=$SEED ckpt=$EVAL_CKPT host=$(hostname)"
python scripts/phase4_scaling/transfer_metrics.py
echo "[tmet] seed=$SEED DONE"

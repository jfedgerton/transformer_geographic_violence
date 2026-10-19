#!/bin/bash
# Forecasting ensemble, FINAL four-head launcher: one array task = one all-region pretrained encoder (5 seeds), all four targets.
# Run:    sbatch --array=0-4 --export=ALL,ENS_PRETRAIN_CUTOFF_OK=1 launch_ens_multitask.sbatch
# ENS_PRETRAIN_CUTOFF_OK=1 only after the pretraining data cutoff (<= 2020-12) has been verified for every encoder.
#SBATCH --job-name=ens_mt
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ens_mt_%A_%a.out
#SBATCH --error=logs/ens_mt_%A_%a.err
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
SEEDS=(123 456 789 101112 131415)
i=${SLURM_ARRAY_TASK_ID}
export ENS_ENC_SEED=${SEEDS[$i]}
export ENS_WORKERS=6
echo "[task $i] four-head encoder=L_contrastive_e_s${ENS_ENC_SEED}"
python scripts/forecast_ensemble/ens_multitask.py

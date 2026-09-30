#!/bin/bash
#SBATCH --job-name=tgv_eval5
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/eval5_%A_%a.out
#SBATCH --error=logs/eval5_%A_%a.err
#
# 5-seed L.e transfer eval: run transfer_eval_full.py for the 2 NEW seeds (101112,131415).
# Old 3 (123/456/789) already have outputs/eval_full_L_contrastive_e_s{seed}.csv from the capacity curve.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
SEEDS=(101112 131415)
export EVAL_TAG=L_contrastive_e_s${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "[eval5] EVAL_TAG=$EVAL_TAG"
python scripts/phase4_scaling/transfer_eval_full.py
echo "EVAL5_DONE_${EVAL_TAG}"

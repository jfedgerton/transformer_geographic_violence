#!/bin/bash
#SBATCH --job-name=tgv_eval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#
# #13: three-assessment transfer eval, one task per grid checkpoint (60).
# Same index decode as the grid: loss=i//15, feat=(i%15)//3, seed=(i%15)%3.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
LOSSES=(mlm autoregressive contrastive tversky)
FEATS=(a b c e f)
SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
li=$((i / 15)); j=$((i % 15)); fi=$((j / 3)); si=$((j % 3))
export EVAL_TAG=M_${LOSSES[$li]}_${FEATS[$fi]}_s${SEEDS[$si]}
echo "[eval] i=$i EVAL_TAG=$EVAL_TAG"
python scripts/phase4_scaling/transfer_eval_full.py
echo "EVAL_DONE_${EVAL_TAG}"

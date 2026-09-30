#!/bin/bash
#SBATCH --job-name=tgv_capeval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/capeval_%A_%a.out
#SBATCH --error=logs/capeval_%A_%a.err
#
# Capacity-curve transfer eval: contrastive x {S,M,L,XL} x {a,e} x 3 seeds = 24 frozen evals.
# Uses the capacity-aware transfer_eval_full.py (CAP derived from EVAL_TAG prefix).
#   index: cap=i//6 ; rem=i%6 ; feat=rem//3 ; seed=rem%3
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
CAPS=(S M L XL); FEATS=(a e); SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
ci=$((i / 6)); r=$((i % 6)); fi=$((r / 3)); si=$((r % 3))
export EVAL_TAG=${CAPS[$ci]}_contrastive_${FEATS[$fi]}_s${SEEDS[$si]}
echo "[capeval] i=$i EVAL_TAG=$EVAL_TAG"
python scripts/phase4_scaling/transfer_eval_full.py
echo "CAPEVAL_DONE_${EVAL_TAG}"

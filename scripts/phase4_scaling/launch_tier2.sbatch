#!/bin/bash
#SBATCH --job-name=tgv_tier2
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/tier2_%A_%a.out
#SBATCH --error=logs/tier2_%A_%a.err
#
# Tier-2 cross-attn decider: 4 modes x 3 seeds, warm-start L-contrastive-e per seed, global holdout.
#   i: mode=i/3 (nospatial/geo/emb/both), seed=i%3 (123/456/789)
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
MODES=(nospatial geo emb both)
SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
mi=$((i / 3)); si=$((i % 3))
export REPR2_MODE=${MODES[$mi]}
export REPR2_CAP=L
export REPR2_FEAT=e
export REPR2_SEED=${SEEDS[$si]}
export REGION=global
export SMOKE=0
export CKPT=$D/data_model/repr_model_L_contrastive_e_s${SEEDS[$si]}.pt
echo "[tier2] i=$i MODE=$REPR2_MODE SEED=$REPR2_SEED ckpt=$(basename $CKPT)"
python scripts/phase4_scaling/tier2_crossattn_finetune.py
echo "TIER2_DONE_${REPR2_MODE}_s${REPR2_SEED}"

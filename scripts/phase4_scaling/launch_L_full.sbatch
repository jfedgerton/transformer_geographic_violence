#!/bin/bash
#SBATCH --job-name=tgv_Lfull
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/Lfull_%A_%a.out
#SBATCH --error=logs/Lfull_%A_%a.err
#
# L full-data real-seed runs at the LOCKED LR 3e-4 (chosen by full-data probe; = width-scaled value).
# a-seed123 is covered by the L@3e-4 probe (copy that ckpt to canonical name when it converges).
#   i: 0=a/456 1=a/789 2=e/123 3=e/456 4=e/789
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
FEATS=(a a e e e)
SEEDS=(456 789 123 456 789)
i=$SLURM_ARRAY_TASK_ID
FEAT=${FEATS[$i]}; SEED=${SEEDS[$i]}
export REPR_TAG=L_contrastive_${FEAT}_s${SEED}
export REPR_CAPACITY=L
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=$FEAT
export REPR_SEED=$SEED
export REPR_LR=3e-4
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[Lfull] i=$i TAG=$REPR_TAG FEAT=$FEAT SEED=$SEED LR=3e-4"
python scripts/phase4_scaling/train_repr_obj.py
echo "LFULL_DONE_${REPR_TAG}"

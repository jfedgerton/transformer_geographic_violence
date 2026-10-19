#!/bin/bash
#SBATCH --job-name=tgv_XLfull
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/XLfull_%A_%a.out
#SBATCH --error=logs/XLfull_%A_%a.err
#
# XL full-data real-seed runs at LOCKED LR 1e-4 (subsample probe: 5e-5=2.576 ~ 1e-4=2.579 << 2e-4=2.635;
# 1e-4 = width-scaled value). 7-day wall (XL ~6-10h/epoch; partition MaxTime=14d). Full data, to convergence.
#   i: 0=a/123 1=a/456 2=a/789 3=e/123 4=e/456 5=e/789
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
FEATS=(a a a e e e)
SEEDS=(123 456 789 123 456 789)
i=$SLURM_ARRAY_TASK_ID
FEAT=${FEATS[$i]}; SEED=${SEEDS[$i]}
export REPR_TAG=XL_contrastive_${FEAT}_s${SEED}
export REPR_CAPACITY=XL
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=$FEAT
export REPR_SEED=$SEED
export REPR_LR=1e-4
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[XLfull] i=$i TAG=$REPR_TAG FEAT=$FEAT SEED=$SEED LR=1e-4"
export REPR_VARIANT=holdout  # 2026-10-19: as originally run (was the script default); required now that the default is removed
python scripts/phase4_scaling/train_repr_obj.py
echo "XLFULL_DONE_${REPR_TAG}"

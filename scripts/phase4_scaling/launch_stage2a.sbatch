#!/bin/bash
#SBATCH --job-name=tgv_stage2a
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/stage2a_%A_%a.out
#SBATCH --error=logs/stage2a_%A_%a.err
#
# Stage 2a: the 5 feature sets under the trusted MLM loss x 3 seeds = 15 runs.
# Establishes the feature-set baseline on the known-good objective.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
SETS=(a b c e f)
SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
FS=${SETS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}
export REPR_TAG=M_mlm_${FS}_s${SEED}
export REPR_CAPACITY=M REPR_VARIANT=holdout REPR_SEED=$SEED REPR_OBJECTIVE=mlm
export REPR_FEATSET=$FS
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_LR=5e-4
export REPR_MAX_EPOCHS=30 REPR_PATIENCE=15 REPR_WARMUP_STEPS=2000 REPR_CLIP=0.5 REPR_BATCH=256
echo "[stage2a] i=$i FEATSET=$FS SEED=$SEED TAG=$REPR_TAG"
python scripts/phase4_scaling/train_repr_obj.py
echo "STAGE2A_DONE_${FS}_${SEED}"

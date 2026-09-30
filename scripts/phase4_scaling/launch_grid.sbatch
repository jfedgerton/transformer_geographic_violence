#!/bin/bash
#SBATCH --job-name=tgv_grid
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#
# THE definitive M grid: loss(4) x feature(5) x seed(3) = 60 runs.
# Country-level 26-country holdout, patience=5, locked per-loss LRs.
# MLM is the control ROW (no separate baseline stage). Index decode:
#   loss = i//15 ; within-loss j = i%15 ; feature = j//3 ; seed = j%3
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
LOSSES=(mlm autoregressive contrastive tversky)
LRS=(5e-4 1e-4 5e-4 1e-4)          # parallel to LOSSES (locked from Stage-2b sweep)
FEATS=(a b c e f)
SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
li=$((i / 15)); j=$((i % 15)); fi=$((j / 3)); si=$((j % 3))
OBJ=${LOSSES[$li]}; LR=${LRS[$li]}; FS=${FEATS[$fi]}; SEED=${SEEDS[$si]}
export REPR_TAG=M_${OBJ}_${FS}_s${SEED}
export REPR_CAPACITY=M
export REPR_VARIANT=holdout
export REPR_SEED=$SEED
export REPR_OBJECTIVE=$OBJ
export REPR_FEATSET=$FS
export REPR_LR=$LR
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_WARMUP_STEPS=2000
export REPR_CLIP=0.5
export REPR_BATCH=256
echo "[grid] i=$i OBJ=$OBJ FEATSET=$FS SEED=$SEED LR=$LR TAG=$REPR_TAG"
python scripts/phase4_scaling/train_repr_obj.py
echo "GRID_DONE_${OBJ}_${FS}_${SEED}"

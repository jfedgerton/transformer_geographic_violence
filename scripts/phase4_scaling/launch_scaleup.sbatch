#!/bin/bash
#SBATCH --job-name=tgv_scale
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err
#
# Capacity scale-up: contrastive x {a,e} x {S,L,XL} x 3 seeds = 18 pretrains.
# (M already trained in the 60-run grid -> reuse those checkpoints.)
# Per-capacity LR ladder: S 1e-3 / M 5e-4 / L 3e-4 / XL 1e-4.
# Index decode: cap=i//6 ; rem=i%6 ; feat=rem//3 ; seed=rem%3
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
CAPS=(S L XL); FEATS=(a e); SEEDS=(123 456 789)
declare -A LRMAP=( [S]=1e-3 [L]=3e-4 [XL]=1e-4 )
i=$SLURM_ARRAY_TASK_ID
ci=$((i / 6)); r=$((i % 6)); fi=$((r / 3)); si=$((r % 3))
CAP=${CAPS[$ci]}; FEAT=${FEATS[$fi]}; SEED=${SEEDS[$si]}; LR=${LRMAP[$CAP]}
export REPR_TAG=${CAP}_contrastive_${FEAT}_s${SEED}
export REPR_CAPACITY=$CAP
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=$FEAT
export REPR_SEED=$SEED
export REPR_LR=$LR
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[scale] i=$i TAG=$REPR_TAG CAP=$CAP FEAT=$FEAT SEED=$SEED LR=$LR"
export REPR_VARIANT=holdout  # 2026-10-19: as originally run (was the script default); required now that the default is removed
python scripts/phase4_scaling/train_repr_obj.py
echo "SCALE_DONE_${REPR_TAG}"

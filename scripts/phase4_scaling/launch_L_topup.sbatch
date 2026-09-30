#!/bin/bash
#SBATCH --job-name=tgv_Ltopup
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/Ltopup_%A_%a.out
#SBATCH --error=logs/Ltopup_%A_%a.err
#
# L*contrastive*e 5-seed top-up: add seeds 101112, 131415 (existing 123/456/789).
# LR 3e-4 (locked width-scaled L value). Full data, to convergence.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
SEEDS=(101112 131415)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$i]}
export REPR_TAG=L_contrastive_e_s${SEED}
export REPR_CAPACITY=L
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=e
export REPR_SEED=$SEED
export REPR_LR=3e-4
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[Ltopup] i=$i TAG=$REPR_TAG SEED=$SEED LR=3e-4"
python scripts/phase4_scaling/train_repr_obj.py
echo "LTOPUP_DONE_${REPR_TAG}"

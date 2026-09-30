#!/bin/bash
#SBATCH --job-name=tgv_xlsub
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/xlsub_%A_%a.out
#SBATCH --error=logs/xlsub_%A_%a.err
#
# XL LR probe on a SUBSAMPLE (REPR_MAX_ROWS) to pick the LR cheaply (ranking stabilizes early).
# LR selection only -> final XL is full-data at the chosen LR. i: 0,1,2 = XL @ {5e-5,1e-4,2e-4}
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
LRS=(5e-5 1e-4 2e-4)
i=$SLURM_ARRAY_TASK_ID
LR=${LRS[$i]}
export REPR_TAG=XL_contrastive_a_s123_lr${LR}_sub
export REPR_CAPACITY=XL
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=a
export REPR_SEED=123
export REPR_LR=$LR
export REPR_MAX_ROWS=1000000
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[xlsub] i=$i TAG=$REPR_TAG LR=$LR MAX_ROWS=1000000"
python scripts/phase4_scaling/train_repr_obj.py
echo "XLSUB_DONE_${REPR_TAG}"

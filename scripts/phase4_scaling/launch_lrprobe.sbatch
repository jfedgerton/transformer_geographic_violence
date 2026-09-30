#!/bin/bash
#SBATCH --job-name=tgv_lrprobe
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/lrprobe_%A_%a.out
#SBATCH --error=logs/lrprobe_%A_%a.err
#
# LR probe at L and XL (contrastive, feature a, seed 123, to early-convergence).
# Winning-LR run is reused as the seed-123 run for that capacity (tag has _lr; rename after).
#   i: 0,1,2 = L @ {2e-4,3e-4,5e-4} ; 3,4,5 = XL @ {5e-5,1e-4,2e-4}
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
CAPS=(L L L XL XL XL)
LRS=(2e-4 3e-4 5e-4 5e-5 1e-4 2e-4)
i=$SLURM_ARRAY_TASK_ID
CAP=${CAPS[$i]}; LR=${LRS[$i]}
export REPR_TAG=${CAP}_contrastive_a_s123_lr${LR}
export REPR_CAPACITY=$CAP
export REPR_OBJECTIVE=contrastive
export REPR_FEATSET=a
export REPR_SEED=123
export REPR_LR=$LR
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_EPOCHS=30
export REPR_PATIENCE=5
export REPR_MIN_DELTA=1e-3
echo "[lrprobe] i=$i TAG=$REPR_TAG CAP=$CAP LR=$LR"
python scripts/phase4_scaling/train_repr_obj.py
echo "LRPROBE_DONE_${REPR_TAG}"

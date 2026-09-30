#!/bin/bash
#SBATCH --job-name=tgv_mvpaf
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/mvpaf_%A_%a.out
#SBATCH --error=logs/mvpaf_%A_%a.err
#
# MVP decider (AFRICA, full data, 8 epochs): 3 modes x 3 seeds, warm-start L-contrastive-e per seed.
#   i: mode=i/3 (nospatial/geo/learned), seed=i%3 (123/456/789)
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
MODES=(nospatial geo learned)
KS=(20 20 64)
HOPSS=(1 1 2)
SEEDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
mi=$((i / 3)); si=$((i % 3))
export REPR2_MODE=${MODES[$mi]}
export REPR2_K=${KS[$mi]}
export REPR2_HOPS=${HOPSS[$mi]}
export REPR2_CAP=L REPR2_FEAT=e REGION=africa SMOKE=0
export REPR2_SEED=${SEEDS[$si]}
export CKPT=$D/data_model/repr_model_L_contrastive_e_s${SEEDS[$si]}.pt
echo "[mvpaf] i=$i MODE=$REPR2_MODE K=$REPR2_K HOPS=$REPR2_HOPS SEED=$REPR2_SEED"
python scripts/phase4_scaling/tier2_mvp_finetune.py
echo "MVPAF_DONE_${REPR2_MODE}_s${REPR2_SEED}"

#!/bin/bash
#SBATCH --job-name=tgv_fs_all
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/fullsrc_all_%A_%a.out
#SBATCH --error=logs/fullsrc_all_%A_%a.err
# 2026-10-19 all-region L encoders (forecasting ensemble, appendix), full source data (no country holdout, no LORO).
# Same settings as launch_fullsrc_L.sbatch. Task i: seed = (123,456,789)[i].
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
md5sum scripts/phase4_scaling/train_repr_obj.py
SDS=(123 456 789); SD=${SDS[$SLURM_ARRAY_TASK_ID]}
unset REPR_LORO
export REPR_VARIANT=full
export REPR_TAG=L_contrastive_e_fullsrc_all_s${SD}
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_SEED=$SD REPR_LR=3e-4 REPR_MAX_EPOCHS=30 REPR_PATIENCE=15
test ! -e data_model/repr_model_${REPR_TAG}.pt || { echo "checkpoint exists: $REPR_TAG"; exit 1; }
python scripts/phase4_scaling/train_repr_obj.py
echo FULLSRC_ALL_DONE_$REPR_TAG

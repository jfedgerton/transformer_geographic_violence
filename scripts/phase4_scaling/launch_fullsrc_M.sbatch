#!/bin/bash
#SBATCH --job-name=tgv_fs_M
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/fullsrc_M_%A_%a.out
#SBATCH --error=logs/fullsrc_M_%A_%a.err
# 2026-10-19 full-source LORO rerun, capacity M. Only the target region is excluded (REPR_VARIANT=full).
# Settings otherwise identical to launch_loro_seeds.sbatch / launch_loro_converge.sbatch:
# contrastive, featset e, LR 5e-4 (locked per-size LR), patience 15, max 30 epochs, script-default panel and vocab.
# Task i: region = i % 6, seed = (123,456,789)[i / 6]. Tasks 0-5 = seed 123 (size check).
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
md5sum scripts/phase4_scaling/train_repr_obj.py
regions=(africa mena europe americas south_asia asia_pacific)
SDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
RG=${regions[$((i % 6))]}; SD=${SDS[$((i / 6))]}
export REPR_LORO=$RG REPR_VARIANT=full
export REPR_TAG=M_contrastive_e_fullsrc_loro_${RG}_s${SD}
export REPR_CAPACITY=M REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_SEED=$SD REPR_LR=5e-4 REPR_MAX_EPOCHS=30 REPR_PATIENCE=15
test ! -e data_model/repr_model_${REPR_TAG}.pt || { echo "checkpoint exists: $REPR_TAG"; exit 1; }
echo "[fullsrc] i=$i TAG=$REPR_TAG RG=$RG SEED=$SD CAP=M LR=5e-4"
python scripts/phase4_scaling/train_repr_obj.py
echo FULLSRC_DONE_$REPR_TAG

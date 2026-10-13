#!/bin/bash
#SBATCH --array=0-14
#SBATCH --job-name=tgv_loro_cv
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-12:00:00
#SBATCH --output=logs/loro_cv_%A_%a.out
#SBATCH --error=logs/loro_cv_%A_%a.err
# Rerun of the 15 LORO encoders that hit the 48h limit (jobs 54257996, 56179446) so each runs to early stop
# (patience 15) or the 30-epoch cap. Settings identical to launch_loro_seeds.sbatch; only walltime changes.
# Time-limited checkpoints archived in data_model/archive_timelimit_48h_2026-10-10/.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PAIRS=(africa:123 mena:123 americas:123 south_asia:123 asia_pacific:123 africa:456 mena:456 europe:456 americas:456 south_asia:456 asia_pacific:456 mena:789 europe:789 south_asia:789 asia_pacific:789)
P=${PAIRS[$SLURM_ARRAY_TASK_ID]}; RG=${P%%:*}; SD=${P##*:}
export REPR_LORO=$RG
export REPR_TAG=L_contrastive_e_loro_${RG}_s${SD}
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_VARIANT=holdout REPR_SEED=${SD} REPR_LR=3e-4 REPR_MAX_EPOCHS=30 REPR_PATIENCE=15
test ! -e data_model/repr_model_${REPR_TAG}.pt || { echo "refusing to overwrite data_model/repr_model_${REPR_TAG}.pt"; exit 1; }
echo "[cfg] task=$SLURM_ARRAY_TASK_ID TAG=$REPR_TAG"
python scripts/phase4_scaling/train_repr_obj.py
echo LORO_CV_DONE_${RG}_s${SD}

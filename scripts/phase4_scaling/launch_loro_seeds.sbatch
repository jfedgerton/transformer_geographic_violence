#!/bin/bash
#SBATCH --array=0-11
#SBATCH --job-name=tgv_loro_sd
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/loro_sd_%A_%a.out
#SBATCH --error=logs/loro_sd_%A_%a.err
# Same settings as launch_loro.sbatch (seed 123 encoders); only REPR_SEED and the tag change.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
seeds=(456 789)
RG=${regions[$((SLURM_ARRAY_TASK_ID % 6))]}
SD=${seeds[$((SLURM_ARRAY_TASK_ID / 6))]}
export REPR_LORO=$RG
export REPR_TAG=L_contrastive_e_loro_${RG}_s${SD}
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_VARIANT=holdout REPR_SEED=${SD} REPR_LR=3e-4
test ! -e data_model/repr_model_${REPR_TAG}.pt || { echo "refusing to overwrite data_model/repr_model_${REPR_TAG}.pt"; exit 1; }
python scripts/phase4_scaling/train_repr_obj.py
echo LORO_SD_DONE_${RG}_s${SD}

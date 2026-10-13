#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_src
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/src_%A_%a.out
#SBATCH --error=logs/src_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export REPR_ONLY=$RG
export REPR_TAG=L_contrastive_e_only_${RG}_s123
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_VARIANT=holdout REPR_SEED=123 REPR_LR=3e-4
python scripts/phase4_scaling/train_repr_obj2.py
echo SRC_DONE_$RG

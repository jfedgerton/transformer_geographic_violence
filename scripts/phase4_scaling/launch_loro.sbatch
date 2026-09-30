#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_loro
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/loro_%A_%a.out
#SBATCH --error=logs/loro_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export REPR_LORO=$RG
export REPR_TAG=L_contrastive_e_loro_${RG}_s123
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_VARIANT=holdout REPR_SEED=123 REPR_LR=3e-4
python scripts/phase4_scaling/train_repr_obj.py
echo LORO_DONE_$RG

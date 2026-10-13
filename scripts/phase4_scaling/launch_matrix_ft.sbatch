#!/bin/bash
#SBATCH --array=0-35
#SBATCH --job-name=tgv_mtx
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mtx_%A_%a.out
#SBATCH --error=logs/mtx_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
SRC=${regions[$((SLURM_ARRAY_TASK_ID / 6))]}
TGT=${regions[$((SLURM_ARRAY_TASK_ID % 6))]}
# 36 cells: off-diagonal = cross-region transfer; diagonal = within-region ceiling (source saw the target region)
export EVAL_REGION=${TGT} EVAL_SEED=123 EVAL_TAG=L_contrastive_e_only_${SRC}_s123
export EVAL_ARMS=ft_pretrained EVAL_FRACS=0.01,0.05 EVAL_SUFFIX=_to_${TGT}
python scripts/phase4_scaling/transfer_eval_region_ft2.py
echo MTX_DONE_${SRC}_to_${TGT}

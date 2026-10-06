#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_rft
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rft_%A_%a.out
#SBATCH --error=logs/rft_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export EVAL_REGION=${RG} EVAL_SEED=123
python scripts/phase4_scaling/transfer_eval_region_ft.py
echo RFT_DONE_$RG

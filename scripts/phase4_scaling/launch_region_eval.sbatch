#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_reval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reval_%A_%a.out
#SBATCH --error=logs/reval_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export EVAL_TAG=L_contrastive_e_loro_${RG}_s123
export EVAL_REGION=${RG}
python scripts/phase4_scaling/transfer_eval_region.py
echo REVAL_DONE_$RG

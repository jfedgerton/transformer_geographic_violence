#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_nest
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nest_%A_%a.out
#SBATCH --error=logs/nest_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export EVAL_REGION=${RG} EVAL_SEED=123
python scripts/phase4_scaling/nested_region_frac.py
echo NEST_DONE_$RG

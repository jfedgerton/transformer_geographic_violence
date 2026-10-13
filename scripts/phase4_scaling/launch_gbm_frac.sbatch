#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=tgv_gbmf
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gbmf_%A_%a.out
#SBATCH --error=logs/gbmf_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
regions=(africa mena europe americas south_asia asia_pacific)
RG=${regions[$SLURM_ARRAY_TASK_ID]}
export EVAL_REGION=${RG}
python scripts/phase4_scaling/gbm_region_frac.py
echo GBMF_DONE_$RG

#!/bin/bash
#SBATCH --array=2,3,4,5,6,7,10,11,15
#SBATCH --job-name=tgv_cft2
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/cft2_%A_%a.out
#SBATCH --error=logs/cft2_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
arms=(warm_ft scratch_ft warm_frozen scratch_frozen)
fracs=(0.01 0.05 0.25 1.0)
i=$SLURM_ARRAY_TASK_ID
export ARM=${arms[$((i/4))]} FRAC=${fracs[$((i%4))]} BATCH=256 EPOCHS=18
export CRIME_OUT=outputs/crime_ft_${ARM}_f${FRAC}.csv
python scripts/phase4_scaling/crime_finetune.py
echo CFT_DONE

#!/bin/bash
#SBATCH --array=0-15
#SBATCH --job-name=tgv_homift
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/homift_%A_%a.out
#SBATCH --error=logs/homift_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CRIME_TAG=grid1km_homicide_wide
arms=(warm_ft scratch_ft warm_frozen scratch_frozen)
fracs=(0.1 0.25 0.5 1.0)
i=$SLURM_ARRAY_TASK_ID
export ARM=${arms[$((i/4))]} FRAC=${fracs[$((i%4))]} BATCH=256 EPOCHS=25
export CRIME_OUT=outputs/homi_ft_${ARM}_f${FRAC}.csv
python scripts/phase4_scaling/crime_finetune.py
echo HOMIFT_DONE

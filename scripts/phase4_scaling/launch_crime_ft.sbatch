#!/bin/bash
#SBATCH --array=0-15
#SBATCH --job-name=tgv_cft
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/cft_%A_%a.out
#SBATCH --error=logs/cft_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
arms=(warm_ft scratch_ft warm_frozen scratch_frozen)
fracs=(0.01 0.05 0.25 1.0)
i=$SLURM_ARRAY_TASK_ID
export ARM=${arms[$((i/4))]} FRAC=${fracs[$((i%4))]} BATCH=512 EPOCHS=25
python scripts/phase4_scaling/crime_finetune.py
echo CFT_DONE

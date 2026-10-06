#!/bin/bash
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH -o slurm-dft-%A_%a.out
module load anaconda
source venv/bin/activate
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
arms=(warm_ft scratch_ft warm_frozen scratch_frozen baseline)
export ARM=${arms[$SLURM_ARRAY_TASK_ID]}
export FRAC=1.0
export SEED=123
echo "[launch] ARM=$ARM FRAC=$FRAC SEED=$SEED"
python scripts/phase4_scaling/dengue_finetune.py

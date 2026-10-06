#!/bin/bash
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-0
#SBATCH -o slurm-dlno-%A_%a.out
module load anaconda
source venv/bin/activate
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
export ARM=warm_lnopen
export FRAC=1.0
export SEED=123
export DENGUE_OUT=outputs/dengue_lnopen_results.csv
echo "[launch] ARM=$ARM FRAC=$FRAC SEED=$SEED OUT=$DENGUE_OUT"
python scripts/phase4_scaling/dengue_finetune_lnopen.py

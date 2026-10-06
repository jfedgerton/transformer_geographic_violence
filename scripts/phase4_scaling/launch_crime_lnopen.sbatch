#!/bin/bash
#SBATCH --array=0-0
#SBATCH --job-name=tgv_clno
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/clno_%A_%a.out
#SBATCH --error=logs/clno_%A_%a.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARM=warm_lnopen FRAC=1.0 SEED=123 BATCH=256 EPOCHS=15
export CRIME_OUT=outputs/crime_ft_warm_lnopen_f1.0.csv
python scripts/phase4_scaling/crime_finetune_lnopen.py
echo CLNO_DONE

#!/bin/bash
#SBATCH --job-name=ces_smkpre
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ces_smkpre_%j.out
#SBATCH --error=logs/ces_smkpre_%j.err
# Smoke pretraining, New England held out (DIV 0), seed 123, 20k respondents, 2 epochs -> *_smoke.pt
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export DIV=0 SEED=123 SMOKE=1
python scripts/phase4_scaling/pretrain_ces.py

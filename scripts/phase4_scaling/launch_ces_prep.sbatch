#!/bin/bash
#SBATCH --job-name=ces_prep
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ces_prep_%j.out
#SBATCH --error=logs/ces_prep_%j.err
# CPU only: builds data_intermediate/ces_tokens.parquet and ces_vocab.json from the CES cumulative file.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
python scripts/phase4_scaling/prep_ces.py

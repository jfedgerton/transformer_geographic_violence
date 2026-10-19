#!/bin/bash
#SBATCH --job-name=cellpers
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cellpers_%j.out
#SBATCH --error=logs/cellpers_%j.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
python scripts/phase4_scaling/cell_alerts_persistence.py

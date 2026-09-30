#!/bin/bash
#SBATCH --job-name=tgv_crsmk
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/crsmk_%j.out
#SBATCH --error=logs/crsmk_%j.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export CRIME_SPATIAL=grid150m CRIME_TARGET=violent CRIME_PANEL=wide SMOKE=1
python scripts/phase4_scaling/crime_ingest.py
echo CRSMK_DONE

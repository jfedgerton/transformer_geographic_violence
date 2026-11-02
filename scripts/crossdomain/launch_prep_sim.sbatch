#!/bin/bash
#SBATCH --job-name=tgv_simprep
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/simprep_%j.out
#SBATCH --error=logs/simprep_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/prep_sim_homicide.py
venv_xd/bin/python scripts/crossdomain/prep_sim_homicide.py
echo SIMPREP_DONE

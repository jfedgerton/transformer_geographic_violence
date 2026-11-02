#!/bin/bash
#SBATCH --job-name=tgv_trafprep
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/trafprep_%j.out
#SBATCH --error=logs/trafprep_%j.err
# PLACEBO data: SIM land-transport deaths (V01-V89) -> event file -> PRIO cell-month panel (same construction as homicide).
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/prep_sim_traffic.py scripts/crossdomain/build_brazil_traffic_panel.py
venv_xd/bin/python scripts/crossdomain/prep_sim_traffic.py
echo TRAFPREP_DONE
venv/bin/python scripts/crossdomain/build_brazil_traffic_panel.py
echo TRAFPANEL_DONE

#!/bin/bash
#SBATCH --job-name=tgv_plprep2
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/plprep2_%j.out
#SBATCH --error=logs/plprep2_%j.err
# Extra placebos (suicide, drowning, dengue): SIM extraction -> PRIO cell-month panels (homicide token construction)
# -> structural profile of all five causes (xd_profile_causes.csv). CPU only; no model runs.
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
md5sum scripts/crossdomain/prep_sim_placebos_v2.py scripts/crossdomain/build_brazil_placebo_panel.py scripts/crossdomain/xd_profile_causes.py
venv_xd/bin/python scripts/crossdomain/prep_sim_placebos_v2.py
echo PLPREP_DONE
for c in suicide drowning dengue; do XD_PLACEBO=$c venv/bin/python scripts/crossdomain/build_brazil_placebo_panel.py; echo "PANEL_DONE $c"; done
venv/bin/python scripts/crossdomain/xd_profile_causes.py

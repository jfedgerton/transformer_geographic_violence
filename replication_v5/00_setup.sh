# 00_setup.sh -- raw inputs, environments and a provenance record. Submits no jobs.
NE_ZIP=data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip
NE_MD5=a33f3bf62d90e1916ebc16fe38959165
pre_00() {
  need scripts/00_master.py   # CONFIG and ensure_dirs(), imported by every stage-01 script
  need venv/bin/python venv_xd/bin/python env/requirements_frozen_2026-10-19.txt env/versions_2026-10-19.txt
  need data_raw/acled_full.csv data_raw/ucdp/GEDEvent_v25_1.csv data_raw/br_centroids/municipios.csv
  ls data_raw/priogrid/static/*.json >/dev/null 2>&1 || die "no PRIO-GRID static json in data_raw/priogrid/static/"
  ls data_raw/sim_brazil/dores/DOBR*.dbc >/dev/null 2>&1 || die "no SIM DOBR*.dbc in data_raw/sim_brazil/dores/ (run $X/fetch_sim_dores.sh on the login node)"
  need data_raw/ces/cumulative_2006-2025.feather data_raw/crimecount/code_crime_core_all.parquet
  need $NE_ZIP; [ "$(md5sum < $NE_ZIP | cut -d' ' -f1)" = "$NE_MD5" ] || die "$NE_ZIP md5 differs from $NE_MD5 (Natural Earth 50 m admin 0 countries)"
  command -v sbatch >/dev/null || die "sbatch not found"
}
run_00() {
  sbatch --version | tee -a "$LOG"
  md5sum replication_v5/*.sh $V/* >> "$LOG"
  (venv/bin/python -m pip freeze 2>&1 || echo PIP_FREEZE_FAILED) > logs/pip_freeze_run_all_v5.txt
  if diff -q logs/pip_freeze_run_all_v5.txt env/requirements_frozen_2026-10-19.txt >/dev/null; then
    echo "pip freeze matches env/requirements_frozen_2026-10-19.txt" | tee -a "$LOG"
  else
    echo "WARNING: pip freeze differs from env/requirements_frozen_2026-10-19.txt" | tee -a "$LOG"
    diff logs/pip_freeze_run_all_v5.txt env/requirements_frozen_2026-10-19.txt >> "$LOG" || true
  fi
  cmp -s scripts/05_build_baseline_features.py $P/05_build_baseline_features.py || echo "WARNING: scripts/ and $P copies of 05_build_baseline_features.py differ (launchers run scripts/)" | tee -a "$LOG"
  cmp -s scripts/build_sequences_super.py $P/build_sequences_super.py || echo "WARNING: scripts/ and $P copies of build_sequences_super.py differ (launchers run scripts/)" | tee -a "$LOG"
}

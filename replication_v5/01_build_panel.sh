# 01_build_panel.sh -- ACLED + UCDP ingest and merge, PRIO-GRID cell-month panel, event sequences, baseline features,
# PRIO-GRID statics, fatality target (input to the Brazil panels) and the per-region seen-token lists.
# Launchers are the ones that built the reported panel (run_all.sh v4 stage 1, unchanged).
D_BASE=""; D_SUPER=""; D_PRIO=""; D_PGM=""; D_SEEN=""; D_DATA=""
pre_01() {  # 01-05, featbuild, PRIO static and pgm target overwrite silently, so every output must be absent
  absent $DI/ucdp_cleaned.parquet $DI/events_unified.parquet $DI/acled_with_grid.parquet $DI/grid_month_panel.parquet $DI/splits.json \
         $DI/sequences.parquet $DI/sequences_super.parquet $DI/vocab.json $DI/vocab_super.json \
         $DI/baseline_features.parquet $DI/baseline_features_super.parquet $DI/priogrid_static.parquet $DI/pgm_fatalities_target.parquet \
         outputs/vocab_unseen_counts_fullsrc.csv
  for rg in "${RGS[@]}"; do absent $DI/seen_tokens_fullsrc_loro_$rg.json; done
}
run_01() {
  D_BASE=$(submit data_base --partition=standard scripts/rebuild_pipeline.sbatch)                 # 01 ingest .. 05 base features
  D_SUPER=$(submit data_super --partition=standard $(dep $D_BASE) $P/launch_featbuild.sbatch)     # 9-field sequences, *_super files
  D_PRIO=$(submit data_prio --account=open --partition=open --cpus-per-task=2 --mem=32G --time=02:00:00 --output=logs/prio_static_%j.out \
           --wrap "bash -lc 'cd $ROOT && module load anaconda && venv/bin/python scripts/build_priogrid_static.py'")
  D_PGM=$(submit data_pgm --account=open --partition=open --cpus-per-task=2 --mem=48G --time=02:00:00 --output=logs/pgm_target_%j.out $(dep $D_BASE) \
          --wrap "bash -lc 'cd $ROOT && module load anaconda && venv/bin/python scripts/build_pgm_fatalities_target.py'")
  D_SEEN=$(submit data_seen $(dep $(ids $D_BASE $D_SUPER)) $P/launch_seen_tokens_fullsrc.sbatch)
  D_DATA=$(ids $D_BASE $D_SUPER $D_PRIO $D_PGM $D_SEEN)
}

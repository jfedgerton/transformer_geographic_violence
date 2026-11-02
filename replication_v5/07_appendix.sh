# 07_appendix.sh -- appendix analyses that are not part of the main LORO grid.
#   Ukraine (Table 7, App D-F, Figs 11-15), M encoder, BOTH fine-tuning arms: top2 (as first reported) and full (the
#     main-text arm). Per arm: country slice + 2022 window (Table 7), map exports at 0.25% and 10% of Europe's training
#     data (maps rendered by the launcher), and the equal-budget alert-gain figure.
#   Other domains (Table 8, App G, App Y): CES turnout (9 divisions), US city crime (5 cities), Brazil dengue
#     (5 macro-regions); each pretrains its own encoder and runs nested + fine-tune grids; App Y TP/FP counts.
# Input not built here: data_intermediate/dengue_panel_raw.csv.gz (from the dengue panel R scripts, scripts/08-11_*.R).
D_UKR=""; D_DOM=""
pre_07() {
  local A sd sh
  if ! on 04; then for A in top2 full; do for sd in "${SDS[@]}"; do for sh in "${SHG[@]}"; do   # Europe hold-out: 3 seeds x 7 shares
    need outputs/share_grid_fullsrc/preds_M_finetune_europe_s${sd}_f${sh}_${A}.npz; done; done; done
    for sd in "${SDS[@]}"; do need outputs/share_grid_fullsrc/metrics_M_nested_europe_s${sd}.csv; done; fi
  for A in top2 full; do
    absent outputs/country_slice_incidence_M_fullsrc_${A}.csv outputs/country_window_incidence_M_fullsrc_${A}.csv outputs/country_window_context_incidence_M_fullsrc_${A}.csv
    absent outputs/ukraine_maps_M_f0025/ukraine_cells_f0.0025_M_fullsrc_${A}.csv outputs/ukraine_maps_M_f01/ukraine_cells_f0.1_M_fullsrc_${A}.csv
    absent outputs/ukraine_alert_gain_M_${A}/fig_ukraine_alert_gain_M_fullsrc_${A}.png
  done
  need data_raw/shapefiles/ne_10m_admin_0_ukr/ne_10m_admin_0_countries_ukr.shp $DI/dengue_panel_raw.csv.gz
  absent outputs/audit_cells_in.csv outputs/audit_cell_country.csv outputs/sudan_cells_keep.csv outputs/sudan_exclude_cells.csv   # Sudan chain (not reported; read at import by the Ukraine scripts)
  for A in top2 full; do absent outputs/ukraine_maps_M_f0025/ukraine_share_selection_M_fullsrc_${A}.csv outputs/ukraine_maps_M_f01/ukraine_share_selection_M_fullsrc_${A}.csv outputs/ukraine_alert_gain_M_${A}/ukraine_alert_gain_cells_M_fullsrc_${A}.csv; done
  absent $DI/ces_tokens.parquet $DI/ces_vocab.json
  absent_glob 'outputs/share_grid/metrics_ces_*' 'outputs/share_grid/metrics_crime_*' 'outputs/share_grid/metrics_dengue_*' \
              'data_model/repr_model_L_contrastive_ces*' 'data_model/repr_model_L_contrastive_crime*' 'data_model/repr_model_L_contrastive_dengue*' \
              "$DI/crime_vocab_*.json" "$DI/dengue_vocab_*.json"
}
run_07() {
  local A D_SL D_WI D_LO D_HI
  local DG=$(ids ${FT2[M]:-} ${FTF[M]:-} ${NEST[M]:-})
  # country_slice/window_fullsrc.py read outputs/sudan_cells_keep.csv at import; it comes from this 3-step chain
  # (cell-to-country audit under Natural Earth 50 m borders), run exactly as on 2026-10-27
  local D_SUD   # declared apart from the assignment so a failed submission stops the driver (set -e)
  D_SUD=$(wrapcpu sudan_cells_keep 16G 00:30:00 "$D_DATA" "venv/bin/python $P/audit_cells_export.py; Rscript $P/audit_cell_country.R; venv/bin/python $P/sudan_cells_fix.py")
  DG=$(ids $DG $D_SUD)
  for A in top2 full; do
    D_SL=$(submit ukr_slice_$A $(dep $DG) --export=ALL,SIZE=M,SLICE_TARGET=incidence,SLICE_FT=$A $P/launch_country_slice_fullsrc.sbatch)
    D_WI=$(submit ukr_window_$A --account=open --partition=open --requeue $(dep $DG) --export=ALL,SIZE=M,WIN_TARGET=incidence,WIN_FT=$A $P/launch_country_window_fullsrc.sbatch)
    D_LO=$(submit ukr_map_f0025_$A $(dep $DG) --export=ALL,SIZE=M,UKR_SHARE=0.0025,UKR_FT=$A,OUTSFX=_f0025 $P/launch_ukraine_map_fullsrc.sbatch)
    D_HI=$(submit ukr_map_f01_$A $(dep $DG) --export=ALL,SIZE=M,UKR_SHARE=0.1,UKR_FT=$A,OUTSFX=_f01 $P/launch_ukraine_map_fullsrc.sbatch)
    wrapcpu ukr_gain_$A 48G 02:00:00 "$(ids $D_LO $D_HI)" \
      "source venv/bin/activate; mkdir -p outputs/ukraine_alert_gain_M_$A; UKR_ARM=$A Rscript $V/fig_ukraine_alert_gain_arm.R outputs/ukraine_maps_M_f0025/ukraine_cells_f0.0025_M_fullsrc_$A.csv outputs/ukraine_maps_M_f01/ukraine_cells_f0.1_M_fullsrc_$A.csv outputs/ukraine_alert_gain_M_$A" >/dev/null
    D_UKR=$(ids $D_UKR $D_SL $D_WI)
  done
  # ---- CES turnout ----
  local C1 C2 C3 C4 K1 K2 K3 K4 K5 K6 K7
  C1=$(submit ces_prep $P/launch_ces_prep.sbatch)
  C2=$(submit ces_pretrain $(dep $C1) $P/launch_ces_pretrain.sbatch)
  C3=$(submit ces_nested $(dep $C2) --array=0-26 --export=ALL,GRID_ARM=nested,VOCAB_REMAP=1 $P/launch_ces_grid.sbatch)
  C4=$(submit ces_finetune $(dep $C2) --array=0-188 --export=ALL,GRID_ARM=finetune,VOCAB_REMAP=1 $P/launch_ces_grid.sbatch)
  # ---- crime and dengue ----
  K1=$(wrapcpu crime_ingest 48G 02:00:00 "" "venv/bin/python $P/crime_ingest.py")   # no full-run launcher exists; launch_crime_smoke runs the same script
  K2=$(submit dengue_ingest scripts/run_dengue_ingest.sbatch)
  K3=$(submit dom_pretrain $(dep $(ids $K1 $K2)) $P/launch_domain_pretrain_seeds.sbatch)
  K4=$(submit crime_nested $(dep $K3) --array=0-14 --export=ALL,DOMAIN=crime,GRID_ARM=nested $P/launch_grid_domain.sbatch)
  K5=$(submit crime_finetune $(dep $K3) --array=0-104 --export=ALL,DOMAIN=crime,GRID_ARM=finetune $P/launch_grid_domain.sbatch)
  K6=$(submit dengue_nested $(dep $K3) --array=0-14 --export=ALL,DOMAIN=dengue,GRID_ARM=nested $P/launch_grid_domain.sbatch)
  K7=$(submit dengue_finetune $(dep $K3) --array=0-104 --export=ALL,DOMAIN=dengue,GRID_ARM=finetune $P/launch_grid_domain.sbatch)
  D_DOM=$(ids $C3 $C4 $K4 $K5 $K6 $K7)
  wrapcpu domain_tp_fp 16G 00:30:00 "$(ids $C4 $K5 $K7)" "venv/bin/python scripts/domain_tp_fp_f0025.py > logs/domain_tp_fp_f0025_v5.txt" >/dev/null
}

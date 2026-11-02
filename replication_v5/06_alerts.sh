# 06_alerts.sh -- equal-budget alert counts and cell maps for the M encoder.
#   alert chain: counts vs persistence and nested (Sec 4.1 4,762 / 1,695 / 710 / 197; App C 2,018; App J), per-cell files,
#                cell maps (Fig 4, Fig 19) via figures_tables/fig_cell_alerts.R with the Natural Earth 50 m zip
#   fine-tune counts, full arm (Sec 4.2: 2,204 at 0.25%); validation gate inside the script reproduces the nested counts
#   country alert rates (App J: 65 countries with >=100 event cell-months; NE 50 m borders -> _ne50m file)
D_ALERT=""
pre_06() {
  if ! on 03; then local i; for i in $(seq 0 17); do need outputs/preds_region_$(tag M $i).npz; done; fi
  if ! on 04; then local i; for i in $(seq 0 125); do need outputs/share_grid_fullsrc/preds_M_finetune_${RGS[$((i % 6))]}_s${SDS[$(((i / 6) % 3))]}_f${SHG[$((i / 18))]}_full.npz; done; fi
  absent outputs/alert_counts_persistence_M_fullsrc.csv outputs/cell_alerts_incidence_M_fullsrc.csv.gz outputs/cell_alerts_persistence_M_fullsrc.csv.gz
  absent outputs/cell_alerts_fig_M_fullsrc outputs/alert_counts_finetune_M_fullsrc_full.csv outputs/country_alert_rates_M_fullsrc_ne50m.csv
}
run_06() {
  D_ALERT=$(submit alerts_M $(dep $(ids ${EVAL[M]:-} ${NEST[M]:-})) --export=ALL,SIZE=M,NE_ZIP=$ROOT/$NE_ZIP $P/launch_alert_chain_fullsrc.sbatch)
  # alert_counts_finetune_fullsrc_full.py has no launcher (it ran as job 56907106); its gate needs alert_counts_nested_M_fullsrc.csv
  wrapcpu alert_ft_full_M 32G 01:00:00 "$(ids $D_ALERT ${FTF[M]:-})" "venv/bin/python $P/alert_counts_finetune_fullsrc_full.py" >/dev/null
  wrapcpu country_rates_M 16G 00:30:00 "$D_ALERT" \
    "Rscript $P/country_alert_rates.R outputs/cell_alerts_incidence_M_fullsrc.csv.gz $NE_ZIP outputs/country_alert_rates_M_fullsrc_ne50m.csv" >/dev/null
}

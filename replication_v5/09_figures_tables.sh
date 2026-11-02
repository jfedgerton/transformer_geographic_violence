# 09_figures_tables.sh -- every figure and LaTeX table reported in the main text and appendix, from the result files.
# CPU only (open account). Output: figures_tables/v5/ (figures) and figures_tables/tables_v5/ (tab_<label>.tex).
# Figures built in earlier stages: Fig 19 (stage 06 alert chain, fig_cell_alerts.R), Figs 11-15 (stage 07 Ukraine).
# The alert-chain launcher exits 0 even when it skips the Fig 19 step (missing R package), so the figures job below
# fails loudly if that png is absent.
# Hand-made, not generated: Figs 1, 2, 7 (TikZ) and Tables 1 (design choices) and 9 (ICD-10 mapping).
FO=figures_tables/v5
pre_09() {
  absent $FO figures_tables/tables_v5
  if ! on 03; then need outputs/persistence_month_t_M_fullsrc.csv; fi
  if ! on 05; then need outputs/bootstrap_ci_M_fullsrc_block5.csv outputs/bootstrap_ci_M_fullsrc_block5_shares.csv outputs/bootstrap_ft_fullsrc_M_full_block5.csv; fi
  if ! on 06; then need outputs/cell_alerts_persistence_M_fullsrc.csv.gz outputs/cell_alerts_incidence_M_fullsrc.csv.gz outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png; fi
  if ! on 08; then need outputs/crossdomain/xd_placebo_boot_v3M.csv outputs/crossdomain/xd_m_boot.csv outputs/crossdomain/xd_s_boot.csv; fi
}
run_09() {
  local D=$(ids $D_BOOT $D_ALERT $D_UKR $D_DOM $D_BRBOOT ${D_PERS:-})
  wrapcpu figures 16G 01:00:00 "$D" \
    "test -f outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png || { echo FIG19_MISSING_stage06_skipped_fig_cell_alerts.R; exit 1; }; mkdir -p $FO; \
     Rscript $P/fig_loro_gains_v2.R outputs/bootstrap_ci_M_fullsrc_block5.csv $FO/fig3_loro_gains; \
     mkdir -p $FO/fig4_cell_maps; Rscript $V/fig_cell_maps.R outputs/cell_alerts_persistence_M_fullsrc.csv.gz outputs/cell_alerts_incidence_M_fullsrc.csv.gz $NE_ZIP $FO/fig4_cell_maps; \
     mkdir -p $FO/fig5_finetune_gains; Rscript $P/fig_finetune_gains.R outputs/bootstrap_ft_fullsrc_M_full_block5.csv $FO/fig5_finetune_gains; \
     mkdir -p $FO/fig6_brazil; Rscript $P/fig_placebo_gradient_a_only_nolabel.R outputs/crossdomain/xd_placebo_boot_v3M.csv $FO/fig6_brazil; \
     mkdir -p $FO/fig9_random_encoder $FO/fig10_share_curve $FO/fig17_share_regions $FO/fig18_persistence_gap $FO/fig16_brazil_random; \
     Rscript $V/fig_random_encoder.R $FO/fig9_random_encoder $ROOT; \
     Rscript $V/fig_share_curve.R $FO/fig10_share_curve $ROOT; \
     Rscript $P/fig_placebo_gradient.R outputs/crossdomain/xd_placebo_boot_v3M.csv $FO/fig16_brazil_random; \
     Rscript $V/fig_share_regions.R $FO/fig17_share_regions $ROOT; \
     Rscript $V/fig_persistence_gap.R $FO/fig18_persistence_gap $ROOT" >/dev/null
  wrapcpu tables 8G 00:30:00 "$D" "TABLES_ROOT=$ROOT venv/bin/python $V/make_tables.py --ukraine-full" >/dev/null
}

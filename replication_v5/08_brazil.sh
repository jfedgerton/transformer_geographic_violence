# 08_brazil.sh -- cross-domain test on Brazilian mortality records (SIM, DATASUS), encoders with the Americas held out.
#   panels   : homicide, traffic, suicide and drowning cell-month panels (App G.2, Table 9)
#   M        : homicide headline (C0 fine-tune 1%, C1 nested) + nested runs for traffic/suicide/drowning (Fig 6, Fig 16,
#              Table 10, App B Brazil numbers, T.3, T.4); bootstraps xd_m_boot.csv and xd_placebo_boot_v3M.csv
#   S        : homicide headline (Table 20, T.4); bootstrap xd_s_boot.csv
#   L        : Table 20 / T.4 L column. These rows come from the earliest grid versions, exactly as run:
#              C0_L = launch_xd_brazil.sbatch ft_rich tasks 1,5,9 (americas, 1%, xd_grid_brazil.py, md5 4f412c3c...)
#              C1_L = launch_xd_ladder.sbatch c_nested tasks 0-2 (americas, paper base, xd_grid_brazil_v2.py, md5 35789544...,
#                     verified in logs/xdlad_56708118_{0,1,2}.out)
#              xd_m_boot.py reads both, so the L runs must finish before it.
# Not run (not reported): thinned-homicide panels and runs, c4/geo placebo jobsets, the S placebo grid and its bootstrap.
# xd_placebo_boot_v3M.py writes INCOMPLETE rows for those comparisons; Fig 6 and Fig 16 use only the complete
# nested_americas_rich_bpaper_capM rows.
D_BR=""; D_BRBOOT=""
pre_08() {
  if ! on 01; then need $DI/vocab_super.json $DI/pgm_fatalities_target.parquet; fi
  absent $DI/xd_brazil_panel.parquet $DI/xd_brazil_traffic_panel.parquet $DI/xd_brazil_suicide_panel.parquet $DI/xd_brazil_drowning_panel.parquet
  local S sd; if ! on 02; then for S in M S L; do for sd in "${SDS[@]}"; do need data_model/repr_model_${S}_contrastive_e_fullsrc_loro_americas_s$sd.pt; done; done; fi
  absent outputs/crossdomain/xd_m_boot.csv outputs/crossdomain/xd_s_boot.csv outputs/crossdomain/xd_placebo_boot_v3M.csv
  absent $DI/xd_brazil_dengue_panel.parquet
  absent_glob "$DI/sim_*_events.parquet" 'outputs/crossdomain/*_capM*' 'outputs/crossdomain/*_capS*'
  for sd in "${SDS[@]}"; do absent outputs/crossdomain/preds_brazil_finetune_americas_s${sd}_rich_f0.01.npz outputs/crossdomain/preds_brazil_nested_americas_s${sd}_rich_bpaper.npz; done
}
run_08() {
  local DD B_HOM B_PAN B_TRF B_PLA D H_M H_S PL d J L0 L1
  DD=$(ids $D_SUPER $D_PGM)   # panels need vocab_super.json and pgm_fatalities_target.parquet
  B_HOM=$(submit br_sim_hom --partition=open $X/launch_prep_sim.sbatch)
  B_PAN=$(submit br_panel_hom --partition=open $(dep $(ids $B_HOM $DD)) $X/launch_brazil_panel.sbatch)
  B_TRF=$(submit br_traffic --partition=standard $(dep $DD) $X/launch_sim_traffic.sbatch)
  B_PLA=$(submit br_placebos --partition=standard $(dep $DD) $X/launch_xd_placebos_prep_v2.sbatch)
  D_BR=$(ids $B_PAN $B_TRF $B_PLA)
  # ---- M ----
  D=$(dep $(ids ${PRE[M]:-} $D_DATA $D_BR))
  H_M=$(submit xd_head_M $D --array=0-5 $X/launch_xd_m.sbatch)
  PL=""; for d in traffic suicide drowning; do
    J=$(submit xd_pl_nested_${d}_M $D --array=0-2 --export=ALL,XD_JOBSET=pl_nested,XD_DOMAIN=$d $X/launch_xd_placeboM.sbatch); PL=$(ids $PL $J)
  done
  # ---- L (Table 20 L column) ----
  D=$(dep $(ids ${PRE[L]:-} $D_DATA $D_BR))
  L0=$(submit xd_L_c0 $D --array=1,5,9 --export=ALL,XD_JOBSET=ft_rich $X/launch_xd_brazil.sbatch)
  L1=$(submit xd_L_c1 $D --array=0-2 --export=ALL,XD_JOBSET=c_nested $X/launch_xd_ladder.sbatch)
  # ---- S ----
  D=$(dep $(ids ${PRE[S]:-} $D_DATA $D_BR))
  H_S=$(submit xd_head_S $D --array=0-5 $X/launch_xd_s.sbatch)
  # ---- bootstraps (M boot launchers say basic -> standard; S says standard) ----
  J=$(submit xd_boot_M --partition=standard $(dep $(ids $H_M $L0 $L1)) $X/launch_xd_m_boot.sbatch); D_BRBOOT=$(ids $D_BRBOOT $J)
  J=$(submit xd_plboot_M --partition=standard $(dep $(ids $H_M $PL)) $X/launch_xd_placebo_boot_v3M.sbatch); D_BRBOOT=$(ids $D_BRBOOT $J)
  J=$(submit xd_boot_S $(dep $H_S) $X/launch_xd_s_boot.sbatch); D_BRBOOT=$(ids $D_BRBOOT $J)
}

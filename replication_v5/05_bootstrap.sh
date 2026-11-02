# 05_bootstrap.sh -- paired bootstrap intervals (1,000 replicates; 5x5-degree blocks and single cells; seed fixed in the
# scripts): persistence + nested (Fig 3, Table 11, Fig 18), random encoder (Table 11), frozen shares (M only; Table 12, Fig 10/17),
# fine-tune vs scratch for both arms (Fig 5, Tables 18, 19, 22, 23). Runs for S, M and L (size ladder, Tables 17-19).
pre_05() {
  local S i FT
  for S in $SIZES; do
    if ! on 03; then for i in $(seq 0 17); do need outputs/eval_region_$(tag $S $i).csv outputs/preds_region_$(tag $S $i).npz outputs/preds_region_$(tag $S $i)_rand.npz; done; fi
    if ! on 04; then
      if [ $S = M ]; then for i in $(seq 0 17); do need outputs/share_grid_fullsrc/metrics_M_nested_${RGS[$((i % 6))]}_s${SDS[$((i / 6))]}.csv; done; fi
      for FT in top2 full; do for i in $(seq 0 125); do
        need outputs/share_grid_fullsrc/metrics_${S}_finetune_${RGS[$((i % 6))]}_s${SDS[$(((i / 6) % 3))]}_f${SHG[$((i / 18))]}_${FT}.csv
      done; done
    fi
    absent outputs/bootstrap_ci_${S}_fullsrc_cell_smoke.csv   # the launcher's smoke run would stop the job late
    absent outputs/bootstrap_ci_${S}_fullsrc_block5.csv outputs/bootstrap_ci_${S}_fullsrc_cell.csv outputs/bootstrap_ci_${S}_fullsrc_block5_rand.csv outputs/bootstrap_ci_${S}_fullsrc_cell_rand.csv
    for FT in top2 full; do absent outputs/bootstrap_ft_fullsrc_${S}_${FT}_block5.csv outputs/bootstrap_ft_fullsrc_${S}_${FT}_cell.csv; done
  done
  absent outputs/bootstrap_ci_M_fullsrc_block5_shares.csv outputs/bootstrap_ci_M_fullsrc_cell_shares.csv
}
D_BOOT=""
run_05() { local S J; for S in $SIZES; do
  J=$(submit boot_pn_$S $(dep ${EVAL[$S]:-}) --export=ALL,BOOT_SIZE=$S $P/launch_bootstrap_ci_fullsrc.sbatch); D_BOOT=$(ids $D_BOOT $J)
  J=$(submit boot_rand_$S --partition=standard $(dep ${RAND[$S]:-}) --export=ALL,BOOT_SIZE=$S $P/launch_bootstrap_rand_fullsrc.sbatch); D_BOOT=$(ids $D_BOOT $J)
  if [ $S = M ]; then J=$(submit boot_shares_M --partition=standard $(dep $(ids ${EVAL[M]:-} ${NEST[M]:-})) --export=ALL,BOOT_SIZE=M $P/launch_bootstrap_shares_fullsrc.sbatch); D_BOOT=$(ids $D_BOOT $J); fi
  J=$(submit boot_ft_top2_$S $(dep ${FT2[$S]:-}) --export=ALL,BOOT_SIZE=$S,BOOT_FT=top2 $P/launch_bootstrap_ft_fullsrc.sbatch); D_BOOT=$(ids $D_BOOT $J)
  J=$(submit boot_ft_full_$S $(dep ${FTF[$S]:-}) --export=ALL,BOOT_SIZE=$S,BOOT_FT=full $P/launch_bootstrap_ft_fullsrc.sbatch); D_BOOT=$(ids $D_BOOT $J)
done; }

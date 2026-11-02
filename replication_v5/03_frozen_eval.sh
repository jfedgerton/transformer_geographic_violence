# 03_frozen_eval.sh -- frozen-embedding evaluation on each held-out region: persistence, base XGBoost, base + pretrained
# embedding (Fig 3, Sec 4.1, Tables 2, 13, 15, T17), the random-weight encoder control (Tables 4, 5, Fig 9), and
# month-t persistence (M only; Table 2). The eval files also carry the civilian (Table 13) and onset (Table 15) targets.
declare -A EVAL=() RAND=()
D_PERS=""   # month-t persistence job; make_tables.py reads its csv (stage 09 depends on it)
pre_03() {
  if ! on 01; then check_data; fi
  local S i; if ! on 02; then for S in $SIZES; do check_encoders $S; done; fi
  for S in $SIZES; do
    for i in $(seq 0 17); do absent outputs/eval_region_$(tag $S $i).csv outputs/preds_region_$(tag $S $i).npz outputs/eval_region_$(tag $S $i)_rand.csv outputs/preds_region_$(tag $S $i)_rand.npz; done
  done
  absent outputs/persistence_month_t_M_fullsrc.csv outputs/persistence_month_t_M_fullsrc.csv.part
}
run_03() { local S; for S in $SIZES; do
  EVAL[$S]=$(submit eval_$S $(dep $(ids ${PRE[$S]:-} $D_DATA)) --export=ALL,SIZE=$S $P/launch_fullsrc_eval.sbatch)
  RAND[$S]=$(submit rand_$S $(dep ${EVAL[$S]}) --export=ALL,SIZE=$S $P/launch_fullsrc_eval_rand.sbatch)
done
D_PERS=$(submit persistence_t_M $(dep ${EVAL[M]}) --export=ALL,PERS_SIZE=M $P/launch_persistence_t_fullsrc.sbatch); }

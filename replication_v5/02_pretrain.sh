# 02_pretrain.sh -- 18 leave-one-region-out encoders (6 held-out regions x seeds 123/456/789) for each size S, M, L.
# GPU per launcher header (S A40; M and L A100). M is the main-text encoder; S and L are the appendix size ladder.
# Note: reported M tasks 10-17 ran through launch_fullsrc_M_a40.sbatch (same trainer and settings, A40 partition);
# this driver uses launch_fullsrc_M.sbatch for all 18. Both launchers ship for provenance.
declare -A PRE=()
pre_02() {
  if ! on 01; then need $DI/baseline_features.parquet $DI/vocab.json $DI/priogrid_static.parquet; fi
  local S i; for S in $SIZES; do for i in $(seq 0 17); do absent data_model/repr_model_$(tag $S $i).pt; done; done
}
run_02() { local S; for S in $SIZES; do PRE[$S]=$(submit pretrain_$S $(dep $(ids $D_BASE $D_PRIO)) $P/launch_fullsrc_$S.sbatch); done; }

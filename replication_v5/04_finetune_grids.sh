# 04_finetune_grids.sh -- low-data grids on each held-out region, 7 shares (0.25-10%) x 5 stratified draws x 3 seeds:
#   nested : (M only) frozen embedding + base vs base at each share (Table 6, Fig 10, Table 12, Fig 17, Table 21)
#   full   : fine-tune ALL encoder parameters vs the same network from scratch (main text Fig 5, Sec 4.2, Tables 18, 22)
#   top2   : fine-tune the last two layers vs scratch (appendix Tables 19, 23; the Ukraine top-two block)
declare -A NEST=() FT2=() FTF=()
pre_04() {
  if ! on 01; then check_data; fi
  local S i FT; if ! on 02; then for S in $SIZES; do check_encoders $S; done; fi
  for i in $(seq 0 17); do absent outputs/share_grid_fullsrc/metrics_M_nested_${RGS[$((i % 6))]}_s${SDS[$((i / 6))]}.csv; done
  for S in $SIZES; do
    for FT in top2 full; do for i in $(seq 0 125); do
      absent outputs/share_grid_fullsrc/metrics_${S}_finetune_${RGS[$((i % 6))]}_s${SDS[$(((i / 6) % 3))]}_f${SHG[$((i / 18))]}_${FT}.csv
    done; done
  done
}
run_04() { local S D; for S in $SIZES; do
  D=$(dep $(ids ${PRE[$S]:-} $D_DATA))
  if [ $S = M ]; then NEST[M]=$(submit nested_M $D --array=0-17 --export=ALL,GRID_SIZE=M,GRID_ARM=nested,VOCAB_REMAP=1 $P/launch_grid_fullsrc.sbatch); fi
  FT2[$S]=$(submit ft_top2_$S $D --array=0-125 --export=ALL,GRID_SIZE=$S,GRID_ARM=finetune,GRID_FT=top2,VOCAB_REMAP=1 $P/launch_grid_fullsrc.sbatch)
  FTF[$S]=$(submit ft_full_$S $D --array=0-125 --export=ALL,GRID_SIZE=$S,GRID_ARM=finetune,GRID_FT=full,VOCAB_REMAP=1 $P/launch_grid_fullsrc.sbatch)
done; }

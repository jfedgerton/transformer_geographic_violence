#!/bin/bash
# run.sh -- the pipeline driver (replaces replication_v5/run_all.sh, _common.sh and the ten stage files).
# Usage: bash pipeline/run.sh <stage> [<stage> ...] | all          DRY=1 bash pipeline/run.sh ... : check and print, submit nothing
#   00 setup  01 build_panel  02 pretrain  03 frozen_eval  04 finetune_grids  05 bootstrap
#   06 alerts  07 appendix  08 brazil  09 figures_tables
# Every job runs pipeline/tasks.sbatch with TASK=<name>; this file sets each job's resources and dependencies.
# All checks for every requested stage run before the first sbatch; if any fails, nothing is submitted. Stages requested
# together chain with afterok; a stage requested alone checks its inputs on disk instead.
set -euo pipefail

# =====================================================================================================================
# paths, constants, helpers
# =====================================================================================================================
ROOT=${PIPE_ROOT:-/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast}   # PIPE_ROOT: synthetic tests only
cd "$ROOT"
T=pipeline/tasks.sbatch
P=scripts/phase4_scaling; X=scripts/crossdomain; V=scripts/replication_v5; DI=data_intermediate
SIZES="S M L"
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
SHG=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.1)   # share_grid_fullsrc.py names files with f"_f{share:g}"
NE_ZIP=data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip
NE_MD5=a33f3bf62d90e1916ebc16fe38959165
DRY=${DRY:-0}                                     # DRY=1: run every check, print every sbatch, submit nothing
mkdir -p logs
LOG=logs/pipeline_$(date +%Y%m%d_%H%M%S).log
NFAIL=0; NDRY=0

# resource blocks used below
OPEN="--account=open --partition=open --requeue"                 # free account (CPU)
CR_STD="--account=jfe4_cr_default --partition=standard"
CR_BASIC="--account=jfe4_cr_default --partition=basic"

die() { echo "FAIL: $*" | tee -a "$LOG" >&2; exit 1; }
need() { for f in "$@"; do test -e "$f" || die "missing $f"; done; }
# absent: outputs that the writing script would refuse to overwrite. In DRY mode an existing file is reported, not fatal,
# so a dry run on a finished tree still prints the whole plan.
absent_glob() { local g f; for g in "$@"; do for f in $g; do absent "$f"; done; done; }
absent() { for f in "$@"; do
  if [ -e "$f" ]; then
    if [ "$DRY" = 1 ]; then echo "DRY-EXISTS $f" >> "$LOG"; NFAIL=$((NFAIL + 1)); else die "$f exists and its writer refuses to overwrite; archive it first"; fi
  fi; done; }
tag() { echo "${1}_contrastive_e_fullsrc_loro_${RGS[$(($2 % 6))]}_s${SDS[$(($2 / 6))]}"; }   # tag SIZE i, i = 0..17
ids() { local o="" x; for x in "$@"; do if [ -n "$x" ]; then o="$o:$x"; fi; done; echo "${o#:}"; }
dep() { if [ -n "${1:-}" ]; then echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes"; fi; }
submit() {  # submit NAME sbatch-args... -> prints a job id (a fake DRYn id in DRY mode)
  local name=$1; shift; local id
  if [ "$DRY" = 1 ]; then  # submit runs inside $(...), so the dry-run counter lives in a file, not a shell variable
    NDRY=$(( $(cat "$LOG.n" 2>/dev/null || echo 0) + 1 )); echo "$NDRY" > "$LOG.n"; id="DRY$NDRY"
    echo "DRY-SUBMIT $name $id sbatch --parsable $*" >> "$LOG"; echo "$id"; return 0; fi
  id=$(sbatch --parsable "$@") || die "sbatch failed for $name"; id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || die "bad job id '$id' for $name"
  echo "SUBMIT $name $id" >> "$LOG"; echo "$id"
}
job() {  # job NAME SLURM_JOB_NAME LOG_STEM sbatch-args... -> one tasks.sbatch job; logs/LOG_STEM.out and .err
  local name=$1 jn=$2 stem=$3; shift 3
  submit "$name" --job-name="$jn" --output="logs/$stem.out" --error="logs/$stem.err" "$@" "$T"
}
# wrapped CPU job on the free open account (Rscript/python one-liners)
wrapcpu() {  # wrapcpu NAME MEM TIME DEPIDS COMMAND
  local name=$1 mem=$2 tim=$3 d=$4; shift 4
  submit "$name" --account=open --partition=open --requeue --cpus-per-task=2 --mem="$mem" --time="$tim" \
    --job-name="v5_$name" --output="logs/v5_${name}_%j.out" $(dep "$d") \
    --wrap "bash -lc 'set -euo pipefail; cd $ROOT; module load anaconda; module load r 2>/dev/null || true; export R_PROFILE_USER=$ROOT/$P/Rprofile_cairo; $*'"
}
check_encoders() {  # $1 = size: 18 LORO encoders and their completion lines
  local i; for i in $(seq 0 17); do
    need data_model/repr_model_$(tag $1 $i).pt
    grep -qx "FULLSRC_DONE_$(tag $1 $i)" logs/fullsrc_*.out || die "no FULLSRC_DONE_$(tag $1 $i) in logs/fullsrc_*.out"
  done
}
check_data() { need $DI/baseline_features.parquet $DI/baseline_features_super.parquet $DI/vocab_super.json $DI/priogrid_static.parquet \
  $DI/pgm_fatalities_target.parquet $DI/acled_with_grid.parquet; for rg in "${RGS[@]}"; do need $DI/seen_tokens_fullsrc_loro_$rg.json; done; }

# job ids shared across stages (empty when the producing stage is not requested)
D_BASE=""; D_SUPER=""; D_PRIO=""; D_PGM=""; D_SEEN=""; D_DATA=""; D_PERS=""
D_BOOT=""; D_ALERT=""; D_UKR=""; D_DOM=""; D_BR=""; D_BRBOOT=""
declare -A PRE=() EVAL=() RAND=() NEST=() FT2=() FTF=()

# =====================================================================================================================
# 00 setup -- raw inputs, environments and a provenance record. Submits no jobs.
# =====================================================================================================================
pre_00() {
  need scripts/00_master.py   # CONFIG and ensure_dirs(), imported by every stage-01 script
  need $T venv/bin/python venv_xd/bin/python env/requirements_frozen_2026-10-19.txt env/versions_2026-10-19.txt
  need data_raw/acled_full.csv data_raw/ucdp/GEDEvent_v25_1.csv data_raw/br_centroids/municipios.csv
  ls data_raw/priogrid/static/*.json >/dev/null 2>&1 || die "no PRIO-GRID static json in data_raw/priogrid/static/"
  ls data_raw/sim_brazil/dores/DOBR*.dbc >/dev/null 2>&1 || die "no SIM DOBR*.dbc in data_raw/sim_brazil/dores/ (run $X/fetch_sim_dores.sh on the login node)"
  need data_raw/ces/cumulative_2006-2025.feather data_raw/crimecount/code_crime_core_all.parquet
  need $NE_ZIP; [ "$(md5sum < $NE_ZIP | cut -d' ' -f1)" = "$NE_MD5" ] || die "$NE_ZIP md5 differs from $NE_MD5 (Natural Earth 50 m admin 0 countries)"
  command -v sbatch >/dev/null || die "sbatch not found"
}
run_00() {
  sbatch --version | tee -a "$LOG"
  md5sum pipeline/run.sh pipeline/tasks.sbatch pipeline/tests/*.py $V/* >> "$LOG"
  (venv/bin/python -m pip freeze 2>&1 || echo PIP_FREEZE_FAILED) > logs/pip_freeze_pipeline.txt
  if diff -q logs/pip_freeze_pipeline.txt env/requirements_frozen_2026-10-19.txt >/dev/null; then
    echo "pip freeze matches env/requirements_frozen_2026-10-19.txt" | tee -a "$LOG"
  else
    echo "WARNING: pip freeze differs from env/requirements_frozen_2026-10-19.txt" | tee -a "$LOG"
    diff logs/pip_freeze_pipeline.txt env/requirements_frozen_2026-10-19.txt >> "$LOG" || true
  fi
}

# =====================================================================================================================
# 01 build_panel -- ACLED + UCDP ingest and merge, PRIO-GRID cell-month panel, sequences, baseline features, PRIO-GRID
# statics, fatality target (input to the Brazil panels) and the per-region seen-token lists.
# =====================================================================================================================
pre_01() {  # these builders overwrite silently, so every output must be absent
  absent $DI/ucdp_cleaned.parquet $DI/events_unified.parquet $DI/acled_with_grid.parquet $DI/grid_month_panel.parquet $DI/splits.json \
         $DI/sequences.parquet $DI/sequences_super.parquet $DI/vocab.json $DI/vocab_super.json \
         $DI/baseline_features.parquet $DI/baseline_features_super.parquet $DI/priogrid_static.parquet $DI/pgm_fatalities_target.parquet \
         outputs/vocab_unseen_counts_fullsrc.csv
  for rg in "${RGS[@]}"; do absent $DI/seen_tokens_fullsrc_loro_$rg.json; done
}
run_01() {
  D_BASE=$(job data_base tgv_rebuild rebuild_%j $CR_STD --cpus-per-task=8 --mem=128G --time=12:00:00 --export=ALL,TASK=rebuild)
  D_SUPER=$(job data_super tgv_featbuild featbuild_%j $CR_STD --cpus-per-task=8 --mem=128G --time=06:00:00 $(dep $D_BASE) --export=ALL,TASK=featbuild)
  D_PRIO=$(submit data_prio --account=open --partition=open --cpus-per-task=2 --mem=32G --time=02:00:00 --output=logs/prio_static_%j.out \
           --wrap "bash -lc 'cd $ROOT && module load anaconda && venv/bin/python scripts/build_priogrid_static.py'")
  D_PGM=$(submit data_pgm --account=open --partition=open --cpus-per-task=2 --mem=48G --time=02:00:00 --output=logs/pgm_target_%j.out $(dep $D_BASE) \
          --wrap "bash -lc 'cd $ROOT && module load anaconda && venv/bin/python scripts/build_pgm_fatalities_target.py'")
  D_SEEN=$(job data_seen seen_tok_fs seen_tok_fs_%j $OPEN --cpus-per-task=2 --mem=96G --time=08:00:00 $(dep $(ids $D_BASE $D_SUPER)) --export=ALL,TASK=seen_tokens)
  D_DATA=$(ids $D_BASE $D_SUPER $D_PRIO $D_PGM $D_SEEN)
}

# =====================================================================================================================
# 02 pretrain -- 18 leave-one-region-out encoders (6 held-out regions x seeds 123/456/789) for each size S, M, L.
# M is the main-text encoder; S and L are the appendix size ladder. Reported M tasks 10-17 ran on A40 GPUs (same trainer
# and settings); this driver runs all 18 M tasks on A100.
# =====================================================================================================================
pre_02() {
  if ! on 01; then need $DI/baseline_features.parquet $DI/vocab.json $DI/priogrid_static.parquet; fi
  local S i; for S in $SIZES; do for i in $(seq 0 17); do absent data_model/repr_model_$(tag $S $i).pt; done; done
}
run_02() {
  local D=$(dep $(ids $D_BASE $D_PRIO))
  PRE[S]=$(job pretrain_S tgv_fs_S fullsrc_S_%A_%a $CR_STD --gres=gpu:a40:1  --cpus-per-task=8 --mem=128G --time=3-00:00:00 --array=0-17 $D --export=ALL,TASK=pretrain,SIZE=S)
  PRE[M]=$(job pretrain_M tgv_fs_M fullsrc_M_%A_%a $CR_STD --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G --time=3-00:00:00 --array=0-17 $D --export=ALL,TASK=pretrain,SIZE=M)
  PRE[L]=$(job pretrain_L tgv_fs_L fullsrc_L_%A_%a $CR_STD --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G --time=5-00:00:00 --array=0-17 $D --export=ALL,TASK=pretrain,SIZE=L)
}

# =====================================================================================================================
# 03 frozen_eval -- persistence, base XGBoost, base + pretrained embedding (Fig 3, Sec 4.1, Tables 2, 13, 15, T17), the
# random-weight encoder control (Tables 4, 5, Fig 9) and month-t persistence (M only; Table 2).
# =====================================================================================================================
pre_03() {
  if ! on 01; then check_data; fi
  local S i; if ! on 02; then for S in $SIZES; do check_encoders $S; done; fi
  for S in $SIZES; do
    for i in $(seq 0 17); do absent outputs/eval_region_$(tag $S $i).csv outputs/preds_region_$(tag $S $i).npz outputs/eval_region_$(tag $S $i)_rand.csv outputs/preds_region_$(tag $S $i)_rand.npz; done
  done
  absent outputs/persistence_month_t_M_fullsrc.csv outputs/persistence_month_t_M_fullsrc.csv.part
}
run_03() {
  local S
  for S in $SIZES; do
    EVAL[$S]=$(job eval_$S tgv_fseval fseval_%A_%a $CR_STD --gres=gpu:a100:1 --cpus-per-task=8 --mem=192G --time=08:00:00 --array=0-17 $(dep $(ids ${PRE[$S]:-} $D_DATA)) --export=ALL,TASK=eval,SIZE=$S)
    RAND[$S]=$(job rand_$S tgv_fsrand fsrand_%A_%a $CR_STD --gres=gpu:a100:1 --cpus-per-task=8 --mem=192G --time=08:00:00 --array=0-17 $(dep ${EVAL[$S]}) --export=ALL,TASK=eval_rand,SIZE=$S)
  done
  D_PERS=$(job persistence_t_M tgv_pers_t pers_t_%j $OPEN --cpus-per-task=4 --mem=48G --time=02:00:00 $(dep ${EVAL[M]}) --export=ALL,TASK=persistence_t,PERS_SIZE=M)
}

# =====================================================================================================================
# 04 finetune_grids -- low-data grids on each held-out region, 7 shares x 5 draws x 3 seeds:
#   nested : (M only) frozen embedding + base vs base (Table 6, Fig 10, Table 12, Fig 17, Table 21)
#   full   : fine-tune ALL encoder parameters vs scratch (Fig 5, Sec 4.2, Tables 18, 22)
#   top2   : fine-tune the last two layers vs scratch (Tables 19, 23; the Ukraine top-two block)
# =====================================================================================================================
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
run_04() {
  local S D G="$CR_STD --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G --time=2-00:00:00"
  for S in $SIZES; do
    D=$(dep $(ids ${PRE[$S]:-} $D_DATA))
    if [ $S = M ]; then NEST[M]=$(job nested_M tgv_gridfs gridfs_%x_%A_%a $G --array=0-17 $D --export=ALL,TASK=grid,GRID_SIZE=M,GRID_ARM=nested,VOCAB_REMAP=1); fi
    FT2[$S]=$(job ft_top2_$S tgv_gridfs gridfs_%x_%A_%a $G --array=0-125 $D --export=ALL,TASK=grid,GRID_SIZE=$S,GRID_ARM=finetune,GRID_FT=top2,VOCAB_REMAP=1)
    FTF[$S]=$(job ft_full_$S tgv_gridfs gridfs_%x_%A_%a $G --array=0-125 $D --export=ALL,TASK=grid,GRID_SIZE=$S,GRID_ARM=finetune,GRID_FT=full,VOCAB_REMAP=1)
  done
}

# =====================================================================================================================
# 05 bootstrap -- paired bootstrap intervals (1,000 replicates; 5x5-degree blocks and single cells): persistence + nested
# (Fig 3, Table 11, Fig 18), random encoder (Table 11), frozen shares (M only; Table 12, Fig 10/17), fine-tune vs scratch
# for both arms (Fig 5, Tables 18, 19, 22, 23). S, M and L (size ladder, Tables 17-19).
# =====================================================================================================================
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
    absent outputs/bootstrap_ci_${S}_fullsrc_cell_smoke.csv   # the smoke run would stop the job late
    absent outputs/bootstrap_ci_${S}_fullsrc_block5.csv outputs/bootstrap_ci_${S}_fullsrc_cell.csv outputs/bootstrap_ci_${S}_fullsrc_block5_rand.csv outputs/bootstrap_ci_${S}_fullsrc_cell_rand.csv
    for FT in top2 full; do absent outputs/bootstrap_ft_fullsrc_${S}_${FT}_block5.csv outputs/bootstrap_ft_fullsrc_${S}_${FT}_cell.csv; done
  done
  absent outputs/bootstrap_ci_M_fullsrc_block5_shares.csv outputs/bootstrap_ci_M_fullsrc_cell_shares.csv
}
run_05() {
  local S J
  for S in $SIZES; do
    J=$(job boot_pn_$S tgv_bootfs bootfs_%j $OPEN --cpus-per-task=4 --mem=96G --time=12:00:00 $(dep ${EVAL[$S]:-}) --export=ALL,TASK=boot_ci,BOOT_SIZE=$S); D_BOOT=$(ids $D_BOOT $J)
    J=$(job boot_rand_$S tgv_bootrand bootrand_%j $CR_STD --cpus-per-task=4 --mem=96G --time=12:00:00 $(dep ${RAND[$S]:-}) --export=ALL,TASK=boot_rand,BOOT_SIZE=$S); D_BOOT=$(ids $D_BOOT $J)
    if [ $S = M ]; then
      J=$(job boot_shares_M tgv_bootsh bootsh_%j $CR_STD --cpus-per-task=4 --mem=128G --time=24:00:00 $(dep $(ids ${EVAL[M]:-} ${NEST[M]:-})) --export=ALL,TASK=boot_shares,BOOT_SIZE=M); D_BOOT=$(ids $D_BOOT $J)
    fi
    J=$(job boot_ft_top2_$S tgv_bootft bootftfs_%j $CR_STD --cpus-per-task=4 --mem=128G --time=24:00:00 $(dep ${FT2[$S]:-}) --export=ALL,TASK=boot_ft,BOOT_SIZE=$S,BOOT_FT=top2); D_BOOT=$(ids $D_BOOT $J)
    J=$(job boot_ft_full_$S tgv_bootft bootftfs_%j $CR_STD --cpus-per-task=4 --mem=128G --time=24:00:00 $(dep ${FTF[$S]:-}) --export=ALL,TASK=boot_ft,BOOT_SIZE=$S,BOOT_FT=full); D_BOOT=$(ids $D_BOOT $J)
  done
}

# =====================================================================================================================
# 06 alerts -- equal-budget alert counts and cell maps for the M encoder: counts vs persistence and nested (Sec 4.1
# 4,762 / 1,695 / 710 / 197; App C; App J), per-cell files, Fig 19; fine-tune counts, full arm (Sec 4.2: 2,204 at 0.25%);
# country alert rates (App J: 65 countries).
# =====================================================================================================================
pre_06() {
  if ! on 03; then local i; for i in $(seq 0 17); do need outputs/preds_region_$(tag M $i).npz; done; fi
  if ! on 04; then local i; for i in $(seq 0 125); do need outputs/share_grid_fullsrc/preds_M_finetune_${RGS[$((i % 6))]}_s${SDS[$(((i / 6) % 3))]}_f${SHG[$((i / 18))]}_full.npz; done; fi
  absent outputs/alert_counts_persistence_M_fullsrc.csv outputs/cell_alerts_incidence_M_fullsrc.csv.gz outputs/cell_alerts_persistence_M_fullsrc.csv.gz
  absent outputs/cell_alerts_fig_M_fullsrc outputs/alert_counts_finetune_M_fullsrc_full.csv outputs/country_alert_rates_M_fullsrc_ne50m.csv
}
run_06() {
  D_ALERT=$(job alerts_M tgv_alertsfs alertsfs_%j $OPEN --cpus-per-task=4 --mem=96G --time=08:00:00 $(dep $(ids ${EVAL[M]:-} ${NEST[M]:-})) --export=ALL,TASK=alert_chain,SIZE=M,NE_ZIP=$ROOT/$NE_ZIP)
  # alert_counts_finetune_fullsrc_full.py has no launcher (it ran as job 56907106); its gate needs alert_counts_nested_M_fullsrc.csv
  wrapcpu alert_ft_full_M 32G 01:00:00 "$(ids $D_ALERT ${FTF[M]:-})" "venv/bin/python $P/alert_counts_finetune_fullsrc_full.py" >/dev/null
  wrapcpu country_rates_M 16G 00:30:00 "$D_ALERT" \
    "Rscript $P/country_alert_rates.R outputs/cell_alerts_incidence_M_fullsrc.csv.gz $NE_ZIP outputs/country_alert_rates_M_fullsrc_ne50m.csv" >/dev/null
}

# =====================================================================================================================
# 07 appendix -- Ukraine (Table 7, App D-F, Figs 11-15), M encoder, BOTH fine-tuning arms (top2 as first reported, full =
# main-text arm): country slice + 2022 window, map exports at 0.25% and 10%, alert-gain figure. Other domains (Table 8,
# App G, App Y): CES turnout, US city crime, Brazil dengue.
# Input not built here: data_intermediate/dengue_panel_raw.csv.gz (dengue panel R scripts; see README).
# =====================================================================================================================
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
  absent outputs/audit_cells_in.csv outputs/audit_cell_country.csv outputs/sudan_cells_keep.csv outputs/sudan_exclude_cells.csv   # Sudan chain (read at import by the Ukraine scripts)
  for A in top2 full; do absent outputs/ukraine_maps_M_f0025/ukraine_share_selection_M_fullsrc_${A}.csv outputs/ukraine_maps_M_f01/ukraine_share_selection_M_fullsrc_${A}.csv outputs/ukraine_alert_gain_M_${A}/ukraine_alert_gain_cells_M_fullsrc_${A}.csv; done
  absent $DI/ces_tokens.parquet $DI/ces_vocab.json
  absent_glob 'outputs/share_grid/metrics_ces_*' 'outputs/share_grid/metrics_crime_*' 'outputs/share_grid/metrics_dengue_*' \
              'data_model/repr_model_L_contrastive_ces*' 'data_model/repr_model_L_contrastive_crime*' 'data_model/repr_model_L_contrastive_dengue*' \
              "$DI/crime_vocab_*.json" "$DI/dengue_vocab_*.json"
}
run_07() {
  local A D_SL D_WI D_LO D_HI DG D_SUD
  DG=$(ids ${FT2[M]:-} ${FTF[M]:-} ${NEST[M]:-})
  # country_slice/window_fullsrc.py read outputs/sudan_cells_keep.csv at import; it comes from this 3-step chain (one script)
  # (cell-to-country audit under Natural Earth 50 m borders). Assigned apart from `local` so a failed submission stops the driver.
  D_SUD=$(wrapcpu sudan_cells_keep 16G 00:30:00 "$D_DATA" "venv/bin/python $P/sudan_cells.py")
  DG=$(ids $DG $D_SUD)
  for A in top2 full; do
    D_SL=$(job ukr_slice_$A tgv_slicefs slicefs_%j $OPEN --cpus-per-task=2 --mem=48G --time=02:00:00 $(dep $DG) --export=ALL,TASK=country_slice,SIZE=M,SLICE_TARGET=incidence,SLICE_FT=$A)
    D_WI=$(job ukr_window_$A tgv_windowfs windowfs_%j $OPEN --cpus-per-task=2 --mem=48G --time=03:00:00 $(dep $DG) --export=ALL,TASK=country_window,SIZE=M,WIN_TARGET=incidence,WIN_FT=$A)
    D_LO=$(job ukr_map_f0025_$A tgv_ukrmapfs ukraine_mapfs_%j $OPEN --cpus-per-task=2 --mem=48G --time=02:00:00 $(dep $DG) --export=ALL,TASK=ukraine_map,SIZE=M,UKR_SHARE=0.0025,UKR_FT=$A,OUTSFX=_f0025)
    D_HI=$(job ukr_map_f01_$A tgv_ukrmapfs ukraine_mapfs_%j $OPEN --cpus-per-task=2 --mem=48G --time=02:00:00 $(dep $DG) --export=ALL,TASK=ukraine_map,SIZE=M,UKR_SHARE=0.1,UKR_FT=$A,OUTSFX=_f01)
    wrapcpu ukr_gain_$A 48G 02:00:00 "$(ids $D_LO $D_HI)" \
      "source venv/bin/activate; mkdir -p outputs/ukraine_alert_gain_M_$A; UKR_ARM=$A Rscript $V/figures.R --one ukr_gain outputs/ukraine_maps_M_f0025/ukraine_cells_f0.0025_M_fullsrc_$A.csv outputs/ukraine_maps_M_f01/ukraine_cells_f0.1_M_fullsrc_$A.csv outputs/ukraine_alert_gain_M_$A" >/dev/null
    D_UKR=$(ids $D_UKR $D_SL $D_WI)
  done
  # ---- CES turnout ----
  local C1 C2 C3 C4 K1 K2 K3 K4 K5 K6 K7 G
  G="$CR_STD --gres=gpu:a100:1 --cpus-per-task=8"
  C1=$(job ces_prep ces_prep ces_prep_%j --account=open --partition=open --cpus-per-task=2 --mem=32G --time=01:00:00 --export=ALL,TASK=ces_prep)
  C2=$(job ces_pretrain ces_pre ces_pre_%A_%a $G --mem=64G --time=1-00:00:00 --array=0-26 $(dep $C1) --export=ALL,TASK=ces_pretrain)
  C3=$(job ces_nested ces_grid cesgrid_%x_%A_%a $G --mem=64G --time=1-00:00:00 --array=0-26 $(dep $C2) --export=ALL,TASK=ces_grid,GRID_ARM=nested,VOCAB_REMAP=1)
  C4=$(job ces_finetune ces_grid cesgrid_%x_%A_%a $G --mem=64G --time=1-00:00:00 --array=0-188 $(dep $C2) --export=ALL,TASK=ces_grid,GRID_ARM=finetune,VOCAB_REMAP=1)
  # ---- crime and dengue ----
  K1=$(wrapcpu crime_ingest 48G 02:00:00 "" "venv/bin/python $P/crime_ingest.py")
  K2=$(job dengue_ingest dengue_ingest dengue_ingest_%j $CR_BASIC --cpus-per-task=4 --mem=32G --time=01:00:00 --export=ALL,TASK=dengue_ingest)
  K3=$(job dom_pretrain dom_pre dompre_%A_%a $G --mem=96G --time=2-00:00:00 --array=0-17 $(dep $(ids $K1 $K2)) --export=ALL,TASK=domain_pretrain)
  K4=$(job crime_nested dom_grid domgrid_%x_%A_%a $G --mem=96G --time=2-00:00:00 --array=0-14 $(dep $K3) --export=ALL,TASK=domain_grid,DOMAIN=crime,GRID_ARM=nested)
  K5=$(job crime_finetune dom_grid domgrid_%x_%A_%a $G --mem=96G --time=2-00:00:00 --array=0-104 $(dep $K3) --export=ALL,TASK=domain_grid,DOMAIN=crime,GRID_ARM=finetune)
  K6=$(job dengue_nested dom_grid domgrid_%x_%A_%a $G --mem=96G --time=2-00:00:00 --array=0-14 $(dep $K3) --export=ALL,TASK=domain_grid,DOMAIN=dengue,GRID_ARM=nested)
  K7=$(job dengue_finetune dom_grid domgrid_%x_%A_%a $G --mem=96G --time=2-00:00:00 --array=0-104 $(dep $K3) --export=ALL,TASK=domain_grid,DOMAIN=dengue,GRID_ARM=finetune)
  D_DOM=$(ids $C3 $C4 $K4 $K5 $K6 $K7)
  wrapcpu domain_tp_fp 16G 00:30:00 "$(ids $C4 $K5 $K7)" "venv/bin/python scripts/domain_tp_fp_f0025.py > logs/domain_tp_fp_f0025_v5.txt" >/dev/null
}

# =====================================================================================================================
# 08 brazil -- cross-domain test on Brazilian mortality records (SIM, DATASUS), encoders with the Americas held out.
#   panels : homicide, traffic, suicide and drowning cell-month panels (App G.2, Table 9)
#   M      : homicide headline (fine-tune 1%, nested) + nested placebos traffic/suicide/drowning (Fig 6, Fig 16, Table 10,
#            App B, T.3, T.4); bootstraps xd_m_boot.csv, xd_placebo_boot_v3M.csv
#   L      : Table 20 / T.4 L column, from the two earliest grid versions, exactly as run (xd_L_c0 = xd_grid_brazil.py
#            ft_rich tasks 1,5,9; xd_L_c1 = xd_grid_brazil_v2.py c_nested tasks 0-2). xd_boot.py (XD_CAP=M) reads both.
#   S      : homicide headline (Table 20, T.4); bootstrap xd_s_boot.csv
# Not run (not reported): thinned-homicide panels, c4/geo placebo jobsets, the S placebo grid. xd_placebo_boot_v3M.py writes
# INCOMPLETE rows for those comparisons; Fig 6 and Fig 16 use only the complete nested_americas_rich_bpaper_capM rows.
# =====================================================================================================================
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
  local DD B_HOM B_PAN B_TRF B_PLA D H_M H_S PL d J L0 L1 G
  G="$CR_STD --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=36:00:00"
  DD=$(ids $D_SUPER $D_PGM)   # panels need vocab_super.json and pgm_fatalities_target.parquet
  B_HOM=$(job br_sim_hom tgv_simprep simprep_%j --account=open --partition=open --cpus-per-task=2 --mem=48G --time=08:00:00 --export=ALL,TASK=br_sim_hom)
  B_PAN=$(job br_panel_hom tgv_brpanel brpanel_%j --account=open --partition=open --cpus-per-task=2 --mem=96G --time=06:00:00 $(dep $(ids $B_HOM $DD)) --export=ALL,TASK=br_panel_hom)
  B_TRF=$(job br_traffic tgv_trafprep trafprep_%j $CR_STD --cpus-per-task=4 --mem=96G --time=03:00:00 $(dep $DD) --export=ALL,TASK=br_traffic)
  B_PLA=$(job br_placebos tgv_plprep2 plprep2_%j $CR_STD --cpus-per-task=4 --mem=96G --time=08:00:00 $(dep $DD) --export=ALL,TASK=br_placebos)
  D_BR=$(ids $B_PAN $B_TRF $B_PLA)
  # ---- M ----
  D=$(dep $(ids ${PRE[M]:-} $D_DATA $D_BR))
  H_M=$(job xd_head_M tgv_xdM xdM_%A_%a $G --array=0-5 $D --export=ALL,TASK=xd_head,XD_CAP=M)
  PL=""; for d in traffic suicide drowning; do
    J=$(job xd_pl_nested_${d}_M tgv_xdplM xdplM_%A_%a $G --array=0-2 $D --export=ALL,TASK=xd_placebo,XD_DOMAIN=$d); PL=$(ids $PL $J)
  done
  # ---- L (Table 20 L column) ----
  D=$(dep $(ids ${PRE[L]:-} $D_DATA $D_BR))
  L0=$(job xd_L_c0 tgv_xdbr xdbr_%A_%a $CR_STD --gres=gpu:p100:1 --cpus-per-task=4 --mem=96G --time=36:00:00 --array=1,5,9 $D --export=ALL,TASK=xd_L_c0)
  L1=$(job xd_L_c1 tgv_xdlad xdlad_%A_%a $G --array=0-2 $D --export=ALL,TASK=xd_L_c1)
  # ---- S ----
  D=$(dep $(ids ${PRE[S]:-} $D_DATA $D_BR))
  H_S=$(job xd_head_S tgv_xdS xdS_%A_%a $G --array=0-5 $D --export=ALL,TASK=xd_head,XD_CAP=S)
  # ---- bootstraps ----
  J=$(job xd_boot_M tgv_xdMboot xdMboot_%j $CR_STD --cpus-per-task=4 --mem=64G --time=04:00:00 $(dep $(ids $H_M $L0 $L1)) --export=ALL,TASK=xd_boot,XD_CAP=M); D_BRBOOT=$(ids $D_BRBOOT $J)
  J=$(job xd_plboot_M tgv_xpbM xpbM_%j $CR_STD --cpus-per-task=4 --mem=64G --time=06:00:00 $(dep $(ids $H_M $PL)) --export=ALL,TASK=xd_plboot); D_BRBOOT=$(ids $D_BRBOOT $J)
  J=$(job xd_boot_S tgv_xdSboot xdSboot_%j $CR_STD --cpus-per-task=4 --mem=64G --time=04:00:00 $(dep $H_S) --export=ALL,TASK=xd_boot,XD_CAP=S); D_BRBOOT=$(ids $D_BRBOOT $J)
}

# =====================================================================================================================
# 09 figures_tables -- every generated figure and LaTeX table, from the result files (CPU, open account).
# Output: figures_tables/v5/ (figures) and figures_tables/tables_v5/ (tab_<label>.tex). Built in earlier stages: Fig 19
# (stage 06), Figs 11-15 (stage 07). The alert chain exits 0 even when it skips Fig 19 (missing R package), so the figures
# job fails loudly if that png is absent. Hand-made: Figs 1, 2, 7 (TikZ), Tables 1 and 9.
# =====================================================================================================================
FO=figures_tables/v5
pre_09() {
  absent $FO figures_tables/tables_v5
  if ! on 03; then need outputs/persistence_month_t_M_fullsrc.csv; fi
  if ! on 05; then need outputs/bootstrap_ci_M_fullsrc_block5.csv outputs/bootstrap_ci_M_fullsrc_block5_shares.csv outputs/bootstrap_ft_fullsrc_M_full_block5.csv; fi
  if ! on 06; then need outputs/cell_alerts_persistence_M_fullsrc.csv.gz outputs/cell_alerts_incidence_M_fullsrc.csv.gz outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png; fi
  if ! on 08; then need outputs/crossdomain/xd_placebo_boot_v3M.csv outputs/crossdomain/xd_m_boot.csv outputs/crossdomain/xd_s_boot.csv; fi
}
run_09() {
  local D
  D=$(ids $D_BOOT $D_ALERT $D_UKR $D_DOM $D_BRBOOT $D_PERS)
  wrapcpu figures 16G 01:00:00 "$D" \
    "test -f outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png || { echo FIG19_MISSING_stage06_skipped_fig_cell_alerts.R; exit 1; }; mkdir -p $FO; \
     Rscript $V/figures.R $FO $ROOT" >/dev/null
  wrapcpu tables 8G 00:30:00 "$D" "TABLES_ROOT=$ROOT venv/bin/python $V/make_tables.py --ukraine-full" >/dev/null
}

# =====================================================================================================================
# main
# =====================================================================================================================
main() {
  local ALL="00 01 02 03 04 05 06 07 08 09" s
  [ $# -ge 1 ] || { echo "usage: bash pipeline/run.sh {00..09|all} ..."; exit 2; }
  ARGS="$*"; [ "$ARGS" = all ] && ARGS="$ALL"
  for s in $ARGS; do case "$s" in 00|01|02|03|04|05|06|07|08|09) RUN[$s]=1 ;; *) echo "unknown stage $s"; exit 2 ;; esac; done
  echo "pipeline/run.sh $ARGS start $(date '+%F %T') DRY=$DRY" | tee "$LOG"
  trap 'rc=$?; ids=$(awk "/^SUBMIT /{print \$3}" "$LOG" | tr "\n" " "); if [ $rc -ne 0 ] && [ -n "$ids" ]; then echo "Jobs submitted before the failure. To cancel them: scancel $ids" | tee -a "$LOG" >&2; fi' EXIT
  # preflight: every check before any sbatch
  for s in $ALL; do if on $s; then "pre_$s"; fi; done
  if [ "$DRY" = 1 ]; then echo "PREFLIGHT_DONE (DRY) existing outputs that a real run would refuse: $NFAIL (lines DRY-EXISTS in $LOG)" | tee -a "$LOG"
  else echo "PREFLIGHT_OK stages: $ARGS" | tee -a "$LOG"; fi
  # submission, in stage order
  for s in $ALL; do if on $s; then "run_$s"; fi; done
  if [ "$DRY" = 1 ]; then echo "DRY: $(grep -c '^DRY-SUBMIT' "$LOG") sbatch calls printed in $LOG, none submitted" | tee -a "$LOG"
  else awk '/^SUBMIT /{printf "%s=%s ", $2, $3} END {print ""}' "$LOG"; fi
  echo "pipeline/run.sh $ARGS done $(date '+%F %T')" | tee -a "$LOG"
}
declare -A RUN=()
on() { [ -n "${RUN[$1]:-}" ]; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi

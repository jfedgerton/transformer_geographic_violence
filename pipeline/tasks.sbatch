#!/bin/bash
# tasks.sbatch -- the one Slurm job script of the pipeline. Every job is `sbatch <resources> --export=ALL,TASK=<name>,...
# pipeline/tasks.sbatch`; resources (account, partition, GPU, memory, time, array, log names) are set by pipeline/run.sh.
# Each branch below is the body of one of the original launchers (named in its comment), with the same environment
# variables and the same command. Scripts are called in place and unedited.
set -eo pipefail
ROOT=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $ROOT
# module load anaconda happens inside each branch whose original launcher did it (the Brazil launchers did not)
P=scripts/phase4_scaling; X=scripts/crossdomain
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
SHS=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.10)
i=${SLURM_ARRAY_TASK_ID:-}
: "${TASK:?pass TASK=<name> in --export}"
echo "[tasks.sbatch] TASK=$TASK array_task=${i:-none} job=${SLURM_JOB_ID:-none} $(date '+%F %T')"

case "$TASK" in

# ================= stage 01: panel =================
rebuild)            # was rebuild_pipeline.sbatch (in scripts/): ACLED + UCDP ingest .. 05 base features
  module load anaconda
  source venv/bin/activate
  export PHASE_MODE=global
  for s in 01_ingest_acled 01b_ingest_ucdp 01c_merge_sources 02_build_grid 03_aggregate_location_month 04_build_sequences 05_build_baseline_features; do
    echo "=== $(date) RUN $s ==="
    python scripts/$s.py
    echo "=== $(date) DONE $s ==="
  done
  echo ALL_BUILD_DONE ;;

featbuild)          # was phase4_scaling/launch_featbuild.sbatch: 9-field sequences + vocab_super, then 05 with suffix _super
  module load anaconda
  source venv/bin/activate
  echo "=== $(date) build_sequences_super ==="
  python scripts/build_sequences_super.py
  echo "=== $(date) 05 (suffix=_super) ==="
  REPR_SEQ_SUFFIX=_super python scripts/05_build_baseline_features.py
  echo SUPERBUILD_DONE ;;

seen_tokens)        # was launch_seen_tokens_fullsrc.sbatch
  module load anaconda
  source venv/bin/activate
  md5sum $P/build_seen_tokens_fullsrc.py
  python $P/build_seen_tokens_fullsrc.py ;;

# ================= stage 02: pretraining =================
pretrain)           # was launch_fullsrc_{S,M,L}.sbatch (identical except size and the locked per-size LR)
  module load anaconda
  source venv/bin/activate
  : "${SIZE:?set SIZE}"
  case "$SIZE" in S) LR=1e-3 ;; M) LR=5e-4 ;; L) LR=3e-4 ;; *) echo "SIZE must be S, M or L"; exit 1 ;; esac
  CAP=$SIZE; unset SIZE   # SIZE is a dispatch variable only; the original launchers did not pass it to the trainer
  md5sum $P/train_repr_obj.py
  RG=${RGS[$((i % 6))]}; SD=${SDS[$((i / 6))]}
  export REPR_LORO=$RG REPR_VARIANT=full
  export REPR_TAG=${CAP}_contrastive_e_fullsrc_loro_${RG}_s${SD}
  export REPR_CAPACITY=$CAP REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_SEED=$SD REPR_LR=$LR REPR_MAX_EPOCHS=30 REPR_PATIENCE=15
  test ! -e data_model/repr_model_${REPR_TAG}.pt || { echo "checkpoint exists: $REPR_TAG"; exit 1; }
  echo "[fullsrc] i=$i TAG=$REPR_TAG RG=$RG SEED=$SD CAP=$CAP LR=$LR"
  python $P/train_repr_obj.py
  echo FULLSRC_DONE_$REPR_TAG ;;

# ================= stage 03: frozen evaluation =================
eval|eval_rand)     # was launch_fullsrc_eval.sbatch (eval) and launch_fullsrc_eval_rand.sbatch (eval_rand)
  module load anaconda
  source venv/bin/activate
  : "${SIZE:?set SIZE}"
  case "$SIZE" in S|M|L|XL) ;; *) echo "bad SIZE=$SIZE"; exit 1;; esac
  RG=${RGS[$((i % 6))]}; SD=${SDS[$((i / 6))]}
  if [ "$TASK" = eval ]; then RND=0; else RND=1; fi
  export EVAL_TAG=${SIZE}_contrastive_e_fullsrc_loro_${RG}_s${SD} EVAL_REGION=${RG} VOCAB_REMAP=1 RANDOM_ENCODER=$RND
  test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing encoder ${EVAL_TAG}"; exit 1; }
  if [ $RND = 0 ]; then
    grep -qx "FULLSRC_DONE_${EVAL_TAG}" logs/fullsrc_*.out || { echo "training not finished for ${EVAL_TAG}"; exit 1; }
  else
    test -f outputs/eval_region_${EVAL_TAG}.csv || { echo "pretrained eval missing for ${EVAL_TAG}"; exit 1; }
  fi
  test -f data_intermediate/seen_tokens_fullsrc_loro_${RG}.json || { echo "missing full-source seen tokens for ${RG}"; exit 1; }
  if [ $RND = 0 ]; then OUTF=outputs/eval_region_${EVAL_TAG}.csv; else OUTF=outputs/eval_region_${EVAL_TAG}_rand.csv; fi
  test ! -e $OUTF || { echo "output exists, refusing to overwrite"; exit 1; }
  md5sum $P/transfer_eval_region_fullpreds.py $P/vocab_remap.py
  if [ $RND = 0 ]; then echo "[cfg] TAG=${EVAL_TAG} REGION=${RG} SIZE=${SIZE} SEED=${SD}"
  else echo "[cfg] RANDOM TAG=${EVAL_TAG} REGION=${RG} SIZE=${SIZE} SEED=${SD}"; fi
  python $P/transfer_eval_region_fullpreds.py
  if [ $RND = 0 ]; then echo FSEVAL_DONE_${EVAL_TAG}; else echo FSRAND_DONE_${EVAL_TAG}; fi ;;

persistence_t)      # was launch_persistence_t_fullsrc.sbatch
  module load anaconda
  md5sum $P/persistence_month_t_fullsrc.py
  venv/bin/python $P/persistence_month_t_fullsrc.py ;;

# ================= stage 04: low-data grids =================
grid)               # was launch_grid_fullsrc.sbatch
  module load anaconda
  source venv/bin/activate
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  : "${GRID_SIZE:?pass GRID_SIZE}"; : "${VOCAB_REMAP:?pass VOCAB_REMAP=1}"
  case "$GRID_ARM" in nested|finetune) ;; *) echo "GRID_ARM must be nested or finetune"; exit 1;; esac
  export SMOKE=${SMOKE:-0} GRID_DRAWS=0,1,2,3,4 GRID_LRS=3e-5,1e-4,3e-4,1e-3 GRID_WDS=0,0.01 GRID_MAX_EP=50 GRID_PATIENCE=5
  export GRID_SWEEP=${GRID_SWEEP:-draw0} GRID_TARGET=${GRID_TARGET:-incidence}
  unset GRID_SHARES GRID_ROOT
  export EVAL_REGION=${RGS[$((i % 6))]} EVAL_SEED=${SDS[$(((i / 6) % 3))]}
  if [ "$GRID_ARM" = "finetune" ]; then : "${GRID_FT:?pass GRID_FT=top2 or full}"; export GRID_SHARES=${SHS[$((i / 18))]}; fi
  TAG=${GRID_SIZE}_contrastive_e_fullsrc_loro_${EVAL_REGION}_s${EVAL_SEED}
  grep -qx "FULLSRC_DONE_${TAG}" logs/fullsrc_*.out || { echo "training not finished for ${TAG}"; exit 1; }
  test -f data_intermediate/seen_tokens_fullsrc_loro_${EVAL_REGION}.json || { echo "missing seen tokens"; exit 1; }
  export GRID_SIZE GRID_FT VOCAB_REMAP
  md5sum $P/share_grid_fullsrc.py $P/vocab_remap.py
  echo "[task] $i SIZE=$GRID_SIZE ARM=$GRID_ARM FT=${GRID_FT:--} TARGET=$GRID_TARGET RG=$EVAL_REGION SEED=$EVAL_SEED SHARE=${GRID_SHARES:-all} SMOKE=$SMOKE"
  python $P/share_grid_fullsrc.py
  echo GRIDFS_DONE_${TAG}_${GRID_ARM}_${GRID_SHARES:-all} ;;

# ================= stage 05: bootstraps =================
boot_ci)            # was launch_bootstrap_ci_fullsrc.sbatch: persistence + nested, 5-replicate smoke then cell and block5
  module load anaconda
  : "${BOOT_SIZE:?set BOOT_SIZE}"; export BOOT_COMPS=persistence,nested
  md5sum $P/bootstrap_ci_fullsrc.py
  BOOT_B=5 BOOT_UNIT=cell BOOT_OUTSFX=_smoke venv/bin/python $P/bootstrap_ci_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=cell   venv/bin/python $P/bootstrap_ci_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=block5 venv/bin/python $P/bootstrap_ci_fullsrc.py
  echo BOOTFS_DONE_${BOOT_SIZE} ;;

boot_rand|boot_shares)  # were launch_bootstrap_rand_fullsrc.sbatch and launch_bootstrap_shares_fullsrc.sbatch
  module load anaconda
  : "${BOOT_SIZE:?set BOOT_SIZE}"
  if [ "$TASK" = boot_rand ]; then export BOOT_COMPS=random; SFX=_rand
  else export BOOT_COMPS=share_0.0025,share_0.005,share_0.0075,share_0.01,share_0.025,share_0.05,share_0.1; SFX=_shares; fi
  md5sum $P/bootstrap_ci_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=block5 BOOT_OUTSFX=$SFX venv/bin/python $P/bootstrap_ci_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=cell BOOT_OUTSFX=$SFX venv/bin/python $P/bootstrap_ci_fullsrc.py
  if [ "$TASK" = boot_rand ]; then echo BOOTRAND_DONE_${BOOT_SIZE}; else echo BOOTSH_DONE_${BOOT_SIZE}; fi ;;

boot_ft)            # was launch_bootstrap_ft_fullsrc.sbatch
  module load anaconda
  : "${BOOT_SIZE:?set BOOT_SIZE}"; : "${BOOT_FT:?set BOOT_FT}"
  md5sum $P/bootstrap_ft_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=block5 venv/bin/python $P/bootstrap_ft_fullsrc.py
  BOOT_B=1000 BOOT_UNIT=cell venv/bin/python $P/bootstrap_ft_fullsrc.py
  echo BOOTFTFS_DONE_${BOOT_SIZE}_${BOOT_FT} ;;

# ================= stage 06: alerts =================
alert_chain)        # was launch_alert_chain_fullsrc.sbatch: six scripts in order (a step whose marker exists is skipped), then Fig 19
  module load anaconda
  : "${SIZE:?set SIZE=S|M|L}"
  case "$SIZE" in S|M|L) ;; *) echo "SIZE must be S, M or L"; exit 1;; esac
  export SIZE OUTSFX=${OUTSFX:-}
  unset FS_ROOT AP_ROOT SWEEP_ROOT
  T=_${SIZE}_fullsrc${OUTSFX}
  md5sum $P/alert_counts_fullsrc.py $P/cell_alerts_agg_fullsrc.py $P/cell_alerts_persistence_fullsrc.py \
         $P/cell_month_decompose_fullsrc.py $P/alert_profile_fullsrc.py $P/alert_budget_sweep_fullsrc.py
  step() {  # step <marker> <script>
    if [ -e "$1" ]; then echo "[skip] $2: $1 exists"; return 0; fi
    echo "[run] $2 $(date)"; venv/bin/python $P/$2; test -e "$1" || { echo "[FATAL] $2 finished without $1"; exit 1; }
  }
  step outputs/alert_counts_persistence${T}.csv          alert_counts_fullsrc.py
  step outputs/cell_alerts_incidence${T}.csv.gz          cell_alerts_agg_fullsrc.py
  step outputs/cell_alerts_persistence${T}.csv.gz        cell_alerts_persistence_fullsrc.py
  step outputs/cell_month_decompose${T}                  cell_month_decompose_fullsrc.py
  step outputs/alert_profile_incidence${T}.DONE          alert_profile_fullsrc.py
  step outputs/alert_budget_sweep${T}.csv                alert_budget_sweep_fullsrc.py
  FD=outputs/cell_alerts_fig${T}
  if [ -e "$FD/fig_cell_alerts_incidence${T}.png" ]; then echo "[skip] figure: png exists"
  elif [ -z "${NE_ZIP:-}" ] || [ ! -f "${NE_ZIP:-/nonexistent}" ]; then echo "[TODO] figure skipped: set NE_ZIP to ne_50m_admin_0_countries.zip"
  else
    module load r 2>/dev/null || true
    export R_PROFILE_USER=$PWD/$P/Rprofile_cairo   # png() on headless nodes needs cairo, not X11
    if command -v Rscript >/dev/null && Rscript -e 'suppressPackageStartupMessages({library(ggplot2); library(sf); library(dplyr); library(readr); library(patchwork); library(scales)})'; then
      mkdir -p "$FD"
      Rscript scripts/replication_v5/figures.R --one fig19 outputs/cell_alerts_incidence${T}.csv.gz "$NE_ZIP" incidence "$FD/fig_cell_alerts_incidence${T}.png"
    else echo "[TODO] figure skipped: Rscript or an R package (ggplot2 sf dplyr readr patchwork scales) is missing"; fi
  fi
  echo ALERTSFS_DONE_${SIZE}${OUTSFX} ;;

# ================= stage 07: Ukraine, CES, crime, dengue =================
country_slice)      # was launch_country_slice_fullsrc.sbatch
  module load anaconda
  : "${SIZE:?set SIZE=S|M|L}"; : "${SLICE_TARGET:?pass SLICE_TARGET=incidence or civilian}"
  export SIZE SLICE_TARGET SLICE_FT=${SLICE_FT:-top2} OUTSFX=${OUTSFX:-}
  unset SLICE_ROOT SLICE_ALLOW_MISSING
  if [ "${SMOKE:-0}" = "1" ]; then [ -n "$OUTSFX" ] || { echo "SMOKE=1 needs OUTSFX"; exit 1; }
  else unset SLICE_DRAWS SLICE_SHARES; fi
  md5sum $P/country_slice_fullsrc.py
  venv/bin/python $P/country_slice_fullsrc.py
  echo SLICEFS_DONE_${SIZE}_${SLICE_TARGET}${OUTSFX} ;;

country_window)     # was launch_country_window_fullsrc.sbatch
  module load anaconda
  : "${SIZE:?set SIZE=S|M|L}"; : "${WIN_TARGET:?pass WIN_TARGET=incidence or civilian}"
  export SIZE WIN_TARGET WIN_FT=${WIN_FT:-top2} OUTSFX=${OUTSFX:-}
  unset WIN_ROOT
  if [ "${SMOKE:-0}" = "1" ]; then [ -n "$OUTSFX" ] || { echo "SMOKE=1 needs OUTSFX"; exit 1; }
  else unset WIN_DRAWS WIN_SHARES WIN_B; fi
  md5sum $P/country_window_fullsrc.py
  venv/bin/python $P/country_window_fullsrc.py
  echo WINDOWFS_DONE_${SIZE}_${WIN_TARGET}${OUTSFX} ;;

ukraine_map)        # was launch_ukraine_map_fullsrc.sbatch: cell export at one share, then the four period maps
  module load anaconda
  source venv/bin/activate
  module load r 2>/dev/null || true
  command -v Rscript >/dev/null || { echo "Rscript not found (load the R module that has ggplot2 and sf)"; exit 1; }
  : "${SIZE:?set SIZE=S|M|L}"
  SHARE=${UKR_SHARE:-auto}; FT=${UKR_FT:-top2}; OSFX=${OUTSFX:-}
  OUTD=outputs/ukraine_maps_${SIZE}${OSFX}
  md5sum $P/export_ukraine_map_fullsrc.py scripts/replication_v5/figures.R
  SIZE=$SIZE python -u $P/export_ukraine_map_fullsrc.py --share "$SHARE" --ft "$FT" --out-dir "$OUTD" | tee logs/ukraine_mapfs_export_${SLURM_JOB_ID}.txt
  CHOSEN=$(grep -o "map uses share [0-9.]*" logs/ukraine_mapfs_export_${SLURM_JOB_ID}.txt | awk '{print $4}')
  test -n "$CHOSEN" || { echo "no share chosen"; exit 1; }
  Rscript scripts/replication_v5/figures.R --one ukr_map "$OUTD/ukraine_cells_f${CHOSEN}_${SIZE}_fullsrc_${FT}.csv"
  echo UKRMAPFS_DONE_${SIZE}${OSFX} share=$CHOSEN ;;

ces_prep)           # was launch_ces_prep.sbatch
  module load anaconda
  source venv/bin/activate
  python $P/prep_ces.py ;;

ces_pretrain)       # was launch_ces_pretrain.sbatch: 9 held-out Census divisions x 3 seeds
  module load anaconda
  source venv/bin/activate
  export DIV=$((i % 9)) SEED=${SDS[$((i / 9))]} SMOKE=0 EPOCHS=30 PATIENCE=5
  echo "[task] $i DIV=$DIV SEED=$SEED"
  python $P/pretrain_ces.py ;;

ces_grid)           # was launch_ces_grid.sbatch
  module load anaconda
  source venv/bin/activate
  export SMOKE=${SMOKE_OVERRIDE:-0} GRID_DRAWS=0,1,2,3,4 GRID_LRS=3e-5,1e-4,3e-4,1e-3 GRID_WDS=0,0.01 GRID_MAX_EP=50 GRID_PATIENCE=5
  export GRID_SWEEP=${GRID_SWEEP:-draw0}
  unset GRID_SHARES
  case "$GRID_ARM" in nested|finetune) ;; *) echo "GRID_ARM must be nested or finetune"; exit 1;; esac
  export DIV=$((i % 9)) SEED=${SDS[$(((i / 9) % 3))]}
  if [ "$GRID_ARM" = "finetune" ]; then export GRID_SHARES=${SHS[$((i / 27))]}; fi
  : "${VOCAB_REMAP:?pass VOCAB_REMAP=1 (main) or 0 (_vorig) in --export}"
  export VOCAB_REMAP
  case "$VOCAB_REMAP" in 0|1) ;; *) echo "VOCAB_REMAP must be 0 or 1"; exit 1;; esac
  echo "[task] $i VOCAB_REMAP=$VOCAB_REMAP SWEEP=$GRID_SWEEP ARM=$GRID_ARM DIV=$DIV SEED=$SEED SHARE=${GRID_SHARES:-all}"
  python $P/share_grid_ces.py ;;

dengue_ingest)      # was run_dengue_ingest.sbatch (in scripts/)
  module load anaconda
  source venv/bin/activate
  python $P/dengue_ingest.py ;;

domain_pretrain)    # was launch_domain_pretrain_seeds.sbatch: crime x 3 seeds (tasks 0-2), dengue 5 macro-regions x 3 seeds (3-17)
  module load anaconda
  source venv/bin/activate
  if [ $i -lt 3 ]; then export DOMAIN=crime SEED=${SDS[$i]}; OUTF=data_model/repr_model_L_contrastive_crime_dom_s${SEED}.pt
  else j=$((i - 3)); export DOMAIN=dengue REGION=$((j % 5)) SEED=${SDS[$((j / 5))]}
       OUTF=data_model/repr_model_L_contrastive_dengue_dom_r$((REGION + 1))_s${SEED}.pt; fi
  test ! -e $OUTF || { echo "refusing to overwrite $OUTF (archive smoke encoders first)"; exit 1; }
  echo "[task] $i DOMAIN=$DOMAIN REGION=${REGION:-na} SEED=$SEED -> $OUTF"
  python $P/pretrain_domain.py ;;

domain_grid)        # was launch_grid_domain.sbatch
  module load anaconda
  source venv/bin/activate
  export SMOKE=${SMOKE_OVERRIDE:-0} GRID_DRAWS=0,1,2,3,4 GRID_LRS=3e-5,1e-4,3e-4,1e-3 GRID_WDS=0,0.01 GRID_MAX_EP=50 GRID_PATIENCE=5
  export GRID_SWEEP=${GRID_SWEEP:-draw0}
  unset GRID_SHARES
  case "$DOMAIN" in crime|dengue) ;; *) echo "DOMAIN must be crime or dengue"; exit 1;; esac
  case "$GRID_ARM" in nested|finetune) ;; *) echo "GRID_ARM must be nested or finetune"; exit 1;; esac
  export CITY=$((i % 5)) SEED=${SDS[$(((i / 5) % 3))]}
  if [ "$GRID_ARM" = "finetune" ]; then export GRID_SHARES=${SHS[$((i / 15))]}; fi
  echo "[task] $i DOMAIN=$DOMAIN ARM=$GRID_ARM CITY=$CITY SEED=$SEED SHARE=${GRID_SHARES:-all}"
  python $P/share_grid_domain.py ;;

# ================= stage 08: Brazil =================
br_sim_hom)         # was crossdomain/launch_prep_sim.sbatch (prep_sim.py was prep_sim_homicide.py)
  md5sum $X/prep_sim.py
  XD_CAUSE=homicide venv_xd/bin/python $X/prep_sim.py
  echo SIMPREP_DONE ;;

br_panel_hom)       # was crossdomain/launch_brazil_panel.sbatch (brazil_panel.py was build_brazil_panel.py)
  md5sum $X/brazil_panel.py
  XD_CAUSE=homicide venv/bin/python $X/brazil_panel.py
  echo BRPANEL_DONE ;;

br_traffic)         # was crossdomain/launch_sim_traffic.sbatch (was prep_sim_traffic.py, build_brazil_traffic_panel.py)
  md5sum $X/prep_sim.py $X/brazil_panel.py
  XD_CAUSE=traffic venv_xd/bin/python $X/prep_sim.py
  echo TRAFPREP_DONE
  XD_CAUSE=traffic venv/bin/python $X/brazil_panel.py
  echo TRAFPANEL_DONE ;;

br_placebos)        # was crossdomain/launch_xd_placebos_prep_v2.sbatch (was prep_sim_placebos_v2.py, build_brazil_placebo_panel.py)
  md5sum $X/prep_sim.py $X/brazil_panel.py $X/xd_profile_causes.py
  XD_CAUSE=placebos venv_xd/bin/python $X/prep_sim.py
  echo PLPREP_DONE
  for c in suicide drowning dengue; do XD_CAUSE=$c venv/bin/python $X/brazil_panel.py; echo "PANEL_DONE $c"; done
  venv/bin/python $X/xd_profile_causes.py ;;

xd_head)            # was crossdomain/launch_xd_m.sbatch (XD_CAP=M) and launch_xd_s.sbatch (XD_CAP=S): tasks 0-2 fine-tune 1%, 3-5 nested
  : "${XD_CAP:?set XD_CAP=M or S}"
  case "$XD_CAP" in M|S) ;; *) echo "XD_CAP must be M or S"; exit 1;; esac
  S=${SDS[$((i % 3))]}
  unset GRID_SHARES GRID_DRAWS GRID_LRS GRID_WDS GRID_MAX_EP GRID_PATIENCE
  export XD_CAP XD_ENC=americas EVAL_SEED=$S XD_DOMAIN=homicide XD_MAPPING=rich XD_BASE=rich XD_PCA=0 XD_EMPTY=zero XD_SURGE=t XD_GEO=0
  if [ $i -lt 3 ]; then export GRID_ARM=finetune GRID_SHARES=0.01; else export GRID_ARM=nested XD_BASE=paper; fi
  echo "[task $i] size=$XD_CAP enc=americas seed=$S arm=$GRID_ARM base=$XD_BASE"
  md5sum $X/xd_grid_brazil_v6.py data_model/repr_model_${XD_CAP}_contrastive_e_fullsrc_loro_americas_s${S}.pt
  venv/bin/python $X/xd_grid_brazil_v6.py
  echo XDBR_DONE ;;

xd_placebo)         # was crossdomain/launch_xd_placeboM.sbatch, jobset pl_nested (the only reported one)
  : "${XD_DOMAIN:?set XD_DOMAIN}"
  export XD_JOBSET=pl_nested   # kept as in the original environment
  S=${SDS[$i]}
  unset GRID_SHARES GRID_DRAWS GRID_LRS GRID_WDS GRID_MAX_EP GRID_PATIENCE
  export XD_CAP=M XD_MAPPING=rich XD_BASE=rich XD_PCA=0 XD_EMPTY=zero XD_SURGE=t XD_GEO=0
  export GRID_ARM=nested XD_BASE=paper
  export XD_ENC=americas EVAL_SEED=$S
  echo "[task $i] size=M jobset=pl_nested domain=$XD_DOMAIN seed=$S arm=$GRID_ARM base=$XD_BASE empty=$XD_EMPTY surge=$XD_SURGE geo=$XD_GEO"
  md5sum $X/xd_grid_brazil_v6.py
  venv/bin/python $X/xd_grid_brazil_v6.py
  echo XDBR_DONE ;;

xd_L_c0)            # was crossdomain/launch_xd_brazil.sbatch, jobset ft_rich (Brazil L column; first grid version)
  ENCS=(americas all); SHARES=(0.0025 0.01 0.05 0.10)
  E=${ENCS[$((i / 12))]}; S=${SDS[$(((i / 4) % 3))]}; SH=${SHARES[$((i % 4))]}
  export XD_JOBSET=ft_rich GRID_ARM=finetune XD_MAPPING=rich GRID_SHARES=$SH   # XD_JOBSET kept as in the original environment
  export XD_ENC=$E EVAL_SEED=$S
  echo "[task $i] jobset=ft_rich enc=$E seed=$S arm=$GRID_ARM mapping=$XD_MAPPING shares=${GRID_SHARES:-default}"
  md5sum $X/xd_grid_brazil.py
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  venv/bin/python $X/xd_grid_brazil.py
  echo XDBR_DONE ;;

xd_L_c1)            # was crossdomain/launch_xd_ladder.sbatch, jobset c_nested tasks 0-2 (Brazil L column; second grid version)
  ENCS=(americas all)
  export XD_JOBSET=c_nested XD_BASE=rich XD_PCA=0   # XD_JOBSET kept as in the original environment
  export GRID_ARM=nested XD_BASE=paper
  if [ "$i" -lt 12 ]; then j=$((i % 6)); E=${ENCS[$((j / 3))]}; S=${SDS[$((j % 3))]}; export XD_MAPPING=rich
    if [ "$i" -ge 6 ]; then export XD_PCA=16; else export XD_PCA=0; fi
  else E=americas; S=${SDS[$((i - 12))]}; export XD_MAPPING=rich_proxy XD_PCA=16; fi
  export XD_ENC=$E EVAL_SEED=$S
  echo "[task $i] jobset=c_nested enc=$E seed=$S arm=$GRID_ARM mapping=$XD_MAPPING base=${XD_BASE:-rich} pca=${XD_PCA:-0} shares=${GRID_SHARES:-default}"
  md5sum $X/xd_grid_brazil_v2.py
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  venv/bin/python $X/xd_grid_brazil_v2.py
  echo XDBR_DONE ;;

xd_boot)            # was crossdomain/launch_xd_m_boot.sbatch (XD_CAP=M) and launch_xd_s_boot.sbatch (XD_CAP=S); since Phase 2
                    # both run the merged xd_boot.py (byte-identical outputs to xd_m_boot.py / xd_s_boot.py, parity_phase2.py)
  : "${XD_CAP:?set XD_CAP=M or S}"
  case "$XD_CAP" in M|S) ;; *) echo "XD_CAP must be M or S"; exit 1 ;; esac
  export XD_CAP
  md5sum $X/xd_boot.py $X/xd_grid_brazil_v6.py
  venv/bin/python $X/xd_boot.py ;;

xd_plboot)          # was crossdomain/launch_xd_placebo_boot_v3M.sbatch
  md5sum $X/xd_placebo_boot_v3M.py
  venv/bin/python $X/xd_placebo_boot_v3M.py ;;

*) echo "unknown TASK=$TASK"; exit 1 ;;
esac
echo "[tasks.sbatch] TASK=$TASK done $(date '+%F %T')"

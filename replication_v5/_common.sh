# _common.sh -- shared helpers for replication_v5 (sourced by run_all.sh; never run directly).
# Every launcher and script is called in place, unedited. ROOT is the only machine-specific path (see README).
ROOT=${V5_ROOT:-/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast}   # V5_ROOT: synthetic tests only
cd "$ROOT"
P=scripts/phase4_scaling; X=scripts/crossdomain; V=scripts/replication_v5; DI=data_intermediate
SIZES="S M L"
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
SHG=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.1)   # share_grid_fullsrc.py names jobs with f"_f{share:g}"
DRY=${DRY:-0}                                     # DRY=1: run every check, print every sbatch, submit nothing
mkdir -p logs
LOG=logs/run_all_v5_$(date +%Y%m%d_%H%M%S).log
NFAIL=0; NDRY=0

die() { echo "FAIL: $*" | tee -a "$LOG" >&2; exit 1; }
need() { for f in "$@"; do test -e "$f" || die "missing $f"; done; }
# absent: outputs that the writing script would refuse to overwrite. In DRY mode an existing file is reported, not fatal,
# so a dry run on a finished tree still prints the whole plan.
absent_glob() { local g f; for g in "$@"; do for f in $g; do absent "$f"; done; done; }   # every existing match of each glob must be absent
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
# wrapped CPU job on the free open account (Rscript/python one-liners that have no launcher of their own)
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

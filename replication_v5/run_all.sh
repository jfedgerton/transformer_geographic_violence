#!/bin/bash
# run_all.sh -- staged Slurm driver for the replication package (v5, 2026-11-02). See replication_v5/README.md.
# Usage: bash replication_v5/run_all.sh <stage> [<stage> ...] | all          DRY=1 bash ... : check and print, submit nothing
#   00 setup  01 build_panel  02 pretrain  03 frozen_eval  04 finetune_grids  05 bootstrap
#   06 alerts  07 appendix  08 brazil  09 figures_tables
# All checks for every requested stage run before the first sbatch; if any fails, nothing is submitted. Stages requested
# together chain with afterok; a stage requested alone checks its inputs on disk instead. Launchers and scripts are
# called in place and never edited.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_common.sh"
ALL="00 01 02 03 04 05 06 07 08 09"
[ $# -ge 1 ] || { echo "usage: bash replication_v5/run_all.sh {00..09|all} ..."; exit 2; }
ARGS="$*"; [ "$ARGS" = all ] && ARGS="$ALL"
declare -A RUN=()
for s in $ARGS; do case "$s" in 00|01|02|03|04|05|06|07|08|09) RUN[$s]=1 ;; *) echo "unknown stage $s"; exit 2 ;; esac; done
on() { [ -n "${RUN[$1]:-}" ]; }
for f in 00_setup 01_build_panel 02_pretrain 03_frozen_eval 04_finetune_grids 05_bootstrap 06_alerts 07_appendix 08_brazil 09_figures_tables; do
  source "$HERE/$f.sh"
done
echo "run_all.sh v5 $ARGS start $(date '+%F %T') DRY=$DRY" | tee "$LOG"
trap 'rc=$?; ids=$(awk "/^SUBMIT /{print \$3}" "$LOG" | tr "\n" " "); if [ $rc -ne 0 ] && [ -n "$ids" ]; then echo "Jobs submitted before the failure. To cancel them: scancel $ids" | tee -a "$LOG" >&2; fi' EXIT

# ---------------- preflight: every check before any sbatch ----------------
for s in $ALL; do if on $s; then "pre_$s"; fi; done
if [ "$DRY" = 1 ]; then echo "PREFLIGHT_DONE (DRY) existing outputs that a real run would refuse: $NFAIL (lines DRY-EXISTS in $LOG)" | tee -a "$LOG"
else echo "PREFLIGHT_OK stages: $ARGS" | tee -a "$LOG"; fi

# ---------------- submission, in stage order ----------------
for s in $ALL; do if on $s; then "run_$s"; fi; done

if [ "$DRY" = 1 ]; then echo "DRY: $(grep -c '^DRY-SUBMIT' "$LOG") sbatch calls printed in $LOG, none submitted" | tee -a "$LOG"
else awk '/^SUBMIT /{printf "%s=%s ", $2, $3} END {print ""}' "$LOG"; fi
echo "run_all.sh v5 $ARGS done $(date '+%F %T')" | tee -a "$LOG"

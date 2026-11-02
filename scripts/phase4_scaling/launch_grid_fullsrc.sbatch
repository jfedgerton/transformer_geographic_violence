#!/bin/bash
#SBATCH --job-name=tgv_gridfs
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gridfs_%x_%A_%a.out
#SBATCH --error=logs/gridfs_%x_%A_%a.err
# 2026-10-26 share grid for the FULL-SOURCE encoders (share_grid_fullsrc.py). Required: GRID_SIZE, GRID_ARM, VOCAB_REMAP=1;
# finetune also GRID_FT=top2|full.
#   nested:   sbatch --array=0-17  --export=ALL,GRID_SIZE=M,GRID_ARM=nested,VOCAB_REMAP=1 launch_grid_fullsrc.sbatch
#   finetune: sbatch --array=0-125 --export=ALL,GRID_SIZE=M,GRID_ARM=finetune,GRID_FT=top2,VOCAB_REMAP=1 launch_grid_fullsrc.sbatch
#   smoke:    add SMOKE=1 (outputs carry _smoke)
# Task i: region = i % 6, seed = (123,456,789)[(i / 6) % 3], share (finetune) = SHS[i / 18]
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
SHS=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.10)
i=$SLURM_ARRAY_TASK_ID
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
md5sum scripts/phase4_scaling/share_grid_fullsrc.py scripts/phase4_scaling/vocab_remap.py
echo "[task] $i SIZE=$GRID_SIZE ARM=$GRID_ARM FT=${GRID_FT:--} TARGET=$GRID_TARGET RG=$EVAL_REGION SEED=$EVAL_SEED SHARE=${GRID_SHARES:-all} SMOKE=$SMOKE"
python scripts/phase4_scaling/share_grid_fullsrc.py
echo GRIDFS_DONE_${TAG}_${GRID_ARM}_${GRID_SHARES:-all}

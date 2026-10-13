#!/bin/bash
#SBATCH --job-name=ces_grid
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/cesgrid_%x_%A_%a.out
#SBATCH --error=logs/cesgrid_%x_%A_%a.err
# CES share grid (after launch_ces_pretrain.sbatch). GRID_ARM passed with --export.
#   nested:   sbatch --array=0-26  --export=ALL,GRID_ARM=nested,VOCAB_REMAP=1   launch_ces_grid.sbatch   (9 divisions x 3 seeds)
#   finetune: sbatch --array=0-188 --export=ALL,GRID_ARM=finetune,VOCAB_REMAP=1 launch_ces_grid.sbatch   (9 x 3 x 7 shares)
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
SDS=(123 456 789); SHS=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.10); i=$SLURM_ARRAY_TASK_ID
export SMOKE=${SMOKE_OVERRIDE:-0} GRID_DRAWS=0,1,2,3,4 GRID_LRS=3e-5,1e-4,3e-4,1e-3 GRID_WDS=0,0.01 GRID_MAX_EP=50 GRID_PATIENCE=5
export GRID_SWEEP=${GRID_SWEEP:-draw0}
unset GRID_SHARES
case "$GRID_ARM" in nested|finetune) ;; *) echo "GRID_ARM must be nested or finetune"; exit 1;; esac
export DIV=$((i % 9)) SEED=${SDS[$(((i / 9) % 3))]}
if [ "$GRID_ARM" = "finetune" ]; then export GRID_SHARES=${SHS[$((i / 27))]}; fi
# vocabulary: no default. Main runs: --export=ALL,GRID_ARM=...,VOCAB_REMAP=1; original-vocabulary sensitivity runs:
# --export=ALL,GRID_ARM=...,VOCAB_REMAP=0 (outputs carry "_vorig"). Always pass it in --export so an exported shell
# value is never inherited by accident.
: "${VOCAB_REMAP:?pass VOCAB_REMAP=1 (main) or 0 (_vorig) in --export}"
export VOCAB_REMAP
case "$VOCAB_REMAP" in 0|1) ;; *) echo "VOCAB_REMAP must be 0 or 1"; exit 1;; esac
echo "[task] $i VOCAB_REMAP=$VOCAB_REMAP SWEEP=$GRID_SWEEP ARM=$GRID_ARM DIV=$DIV SEED=$SEED SHARE=${GRID_SHARES:-all}"
python scripts/phase4_scaling/share_grid_ces.py

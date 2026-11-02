#!/bin/bash
#SBATCH --job-name=tgv_xdM
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/xdM_%A_%a.out
#SBATCH --error=logs/xdM_%A_%a.err
# Brazil headline with the M Americas-held-out encoder (Jared 2026-10-25: main size M). Same construction as the L runs:
#   array 0-2: C0 finetune 1% rich (top2 = last 2 of M's 4 layers), seeds 123/456/789
#   array 3-5: C1 nested, paper base, rich, all shares, seeds 123/456/789
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
i=${SLURM_ARRAY_TASK_ID}; SEEDS=(123 456 789); S=${SEEDS[$((i % 3))]}
unset GRID_SHARES GRID_DRAWS GRID_LRS GRID_WDS GRID_MAX_EP GRID_PATIENCE
export XD_CAP=M XD_ENC=americas EVAL_SEED=$S XD_DOMAIN=homicide XD_MAPPING=rich XD_BASE=rich XD_PCA=0 XD_EMPTY=zero XD_SURGE=t XD_GEO=0
if [ $i -lt 3 ]; then export GRID_ARM=finetune GRID_SHARES=0.01; else export GRID_ARM=nested XD_BASE=paper; fi
echo "[task $i] size=M enc=americas seed=$S arm=$GRID_ARM base=$XD_BASE"
md5sum scripts/crossdomain/xd_grid_brazil_v6.py data_model/repr_model_M_contrastive_e_fullsrc_loro_americas_s${S}.pt
venv/bin/python scripts/crossdomain/xd_grid_brazil_v6.py
echo XDBR_DONE

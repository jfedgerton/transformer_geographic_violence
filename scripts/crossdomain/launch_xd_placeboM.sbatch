#!/bin/bash
#SBATCH --job-name=tgv_xdplM
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/xdplM_%A_%a.out
#SBATCH --error=logs/xdplM_%A_%a.err
# 2026-10-26 placebos, sparsity control and geography-only with the M Americas-held-out encoder (main size M).
# Same jobsets and construction as launch_xd_placebo2.sbatch, but xd_grid_brazil_v6.py with XD_CAP=M (outputs carry _capM).
# array 0-2 = seeds 123/456/789; XD_JOBSET and XD_DOMAIN set at submission.
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
: "${XD_JOBSET:?set XD_JOBSET}"; : "${XD_DOMAIN:?set XD_DOMAIN}"
i=${SLURM_ARRAY_TASK_ID}; SEEDS=(123 456 789); S=${SEEDS[$i]}
unset GRID_SHARES GRID_DRAWS GRID_LRS GRID_WDS GRID_MAX_EP GRID_PATIENCE
export XD_CAP=M XD_MAPPING=rich XD_BASE=rich XD_PCA=0 XD_EMPTY=zero XD_SURGE=t XD_GEO=0
case "$XD_JOBSET" in
  pl_nested) export GRID_ARM=nested XD_BASE=paper ;;
  pl_c4nested) export GRID_ARM=nested XD_BASE=paper XD_EMPTY=keep XD_SURGE=prev ;;
  pl_geo) export GRID_ARM=nested XD_BASE=paper XD_EMPTY=keep XD_SURGE=prev XD_GEO=1 ;;
  *) echo "bad XD_JOBSET $XD_JOBSET"; exit 1 ;;
esac
export XD_ENC=americas EVAL_SEED=$S
echo "[task $i] size=M jobset=$XD_JOBSET domain=$XD_DOMAIN seed=$S arm=$GRID_ARM base=$XD_BASE empty=$XD_EMPTY surge=$XD_SURGE geo=$XD_GEO"
md5sum scripts/crossdomain/xd_grid_brazil_v6.py
venv/bin/python scripts/crossdomain/xd_grid_brazil_v6.py
echo XDBR_DONE

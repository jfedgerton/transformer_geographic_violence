#!/bin/bash
#SBATCH --job-name=tgv_xdlad
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/xdlad_%A_%a.out
#SBATCH --error=logs/xdlad_%A_%a.err
# Cross-domain C3 (Brazil). sbatch --export=ALL,XD_JOBSET=nested|ft_rich|ft_min --array=... launch_xd_brazil.sbatch
#   nested : array 0-11  = ENC{americas,all} x SEED{123,456,789} x MAPPING{rich,min}
#   ft_rich: array 0-23  = ENC x SEED x SHARE{0.0025,0.01,0.05,0.10}   (MAPPING=rich)
#   ft_min : array 0-5   = ENC x SEED at SHARE 0.01                     (MAPPING=min)
#   c_nested: array 0-14 (2026-10-24 favorable ladder): 0-5 C1 ENC x SEED (paper base); 6-11 C2 ENC x SEED (paper base +
#             PCA16); 12-14 C3 SEED (americas, rich_proxy, paper base + PCA16)
#   c3_ft  : array 0-2   = SEED, americas, rich_proxy, SHARE 0.01 (top2/full/scratch/top2_rand)
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
: "${XD_JOBSET:?set XD_JOBSET}"
i=${SLURM_ARRAY_TASK_ID}
ENCS=(americas all); SEEDS=(123 456 789); SHARES=(0.0025 0.01 0.05 0.10)
export XD_BASE=rich XD_PCA=0                 # explicit defaults; only c_nested overrides
case "$XD_JOBSET" in
  c_nested) export GRID_ARM=nested XD_BASE=paper
           if [ "$i" -lt 12 ]; then j=$((i % 6)); E=${ENCS[$((j / 3))]}; S=${SEEDS[$((j % 3))]}; export XD_MAPPING=rich
             if [ "$i" -ge 6 ]; then export XD_PCA=16; else export XD_PCA=0; fi
           else E=americas; S=${SEEDS[$((i - 12))]}; export XD_MAPPING=rich_proxy XD_PCA=16; fi ;;
  c3_ft)   E=americas; S=${SEEDS[$i]}; export GRID_ARM=finetune XD_MAPPING=rich_proxy GRID_SHARES=0.01 ;;
  *) echo "bad XD_JOBSET $XD_JOBSET"; exit 1 ;;
esac
export XD_ENC=$E EVAL_SEED=$S
echo "[task $i] jobset=$XD_JOBSET enc=$E seed=$S arm=$GRID_ARM mapping=$XD_MAPPING base=${XD_BASE:-rich} pca=${XD_PCA:-0} shares=${GRID_SHARES:-default}"
md5sum scripts/crossdomain/xd_grid_brazil_v2.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
venv/bin/python scripts/crossdomain/xd_grid_brazil_v2.py
echo XDBR_DONE

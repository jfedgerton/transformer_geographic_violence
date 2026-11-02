#!/bin/bash
#SBATCH --job-name=tgv_xdbr
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/xdbr_%A_%a.out
#SBATCH --error=logs/xdbr_%A_%a.err
# Cross-domain C3 (Brazil). sbatch --export=ALL,XD_JOBSET=nested|ft_rich|ft_min --array=... launch_xd_brazil.sbatch
#   nested : array 0-11  = ENC{americas,all} x SEED{123,456,789} x MAPPING{rich,min}
#   ft_rich: array 0-23  = ENC x SEED x SHARE{0.0025,0.01,0.05,0.10}   (MAPPING=rich)
#   ft_min : array 0-5   = ENC x SEED at SHARE 0.01                     (MAPPING=min)
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
: "${XD_JOBSET:?set XD_JOBSET}"
i=${SLURM_ARRAY_TASK_ID}
ENCS=(americas all); SEEDS=(123 456 789); SHARES=(0.0025 0.01 0.05 0.10)
case "$XD_JOBSET" in
  nested)  MAPS=(rich min); E=${ENCS[$((i / 6))]}; S=${SEEDS[$(((i / 2) % 3))]}; M=${MAPS[$((i % 2))]}
           export GRID_ARM=nested XD_MAPPING=$M ;;
  ft_rich) E=${ENCS[$((i / 12))]}; S=${SEEDS[$(((i / 4) % 3))]}; SH=${SHARES[$((i % 4))]}
           export GRID_ARM=finetune XD_MAPPING=rich GRID_SHARES=$SH ;;
  ft_min)  E=${ENCS[$((i / 3))]}; S=${SEEDS[$((i % 3))]}
           export GRID_ARM=finetune XD_MAPPING=min GRID_SHARES=0.01 ;;
  *) echo "bad XD_JOBSET $XD_JOBSET"; exit 1 ;;
esac
export XD_ENC=$E EVAL_SEED=$S
echo "[task $i] jobset=$XD_JOBSET enc=$E seed=$S arm=$GRID_ARM mapping=$XD_MAPPING shares=${GRID_SHARES:-default}"
md5sum scripts/crossdomain/xd_grid_brazil.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
venv/bin/python scripts/crossdomain/xd_grid_brazil.py
echo XDBR_DONE

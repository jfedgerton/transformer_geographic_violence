#!/bin/bash
#SBATCH --job-name=dom_grid
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/domgrid_%x_%A_%a.out
#SBATCH --error=logs/domgrid_%x_%A_%a.err
# Domain share grid (after launch_domain_pretrain_seeds.sbatch). DOMAIN and GRID_ARM passed with --export.
#   nested:   --array=0-14   (5 units x 3 seeds)        finetune: --array=0-104 (5 x 3 x 7 shares)
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
SDS=(123 456 789); SHS=(0.0025 0.005 0.0075 0.01 0.025 0.05 0.10); i=$SLURM_ARRAY_TASK_ID
# full-run settings set explicitly (SMOKE may be overridden only via SMOKE_OVERRIDE=1 for the smoke test)
export SMOKE=${SMOKE_OVERRIDE:-0} GRID_DRAWS=0,1,2,3,4 GRID_LRS=3e-5,1e-4,3e-4,1e-3 GRID_WDS=0,0.01 GRID_MAX_EP=50 GRID_PATIENCE=5
export GRID_SWEEP=${GRID_SWEEP:-draw0}
unset GRID_SHARES
case "$DOMAIN" in crime|dengue) ;; *) echo "DOMAIN must be crime or dengue"; exit 1;; esac
case "$GRID_ARM" in nested|finetune) ;; *) echo "GRID_ARM must be nested or finetune"; exit 1;; esac
export CITY=$((i % 5)) SEED=${SDS[$(((i / 5) % 3))]}
if [ "$GRID_ARM" = "finetune" ]; then export GRID_SHARES=${SHS[$((i / 15))]}; fi
echo "[task] $i DOMAIN=$DOMAIN ARM=$GRID_ARM CITY=$CITY SEED=$SEED SHARE=${GRID_SHARES:-all}"
python scripts/phase4_scaling/share_grid_domain.py

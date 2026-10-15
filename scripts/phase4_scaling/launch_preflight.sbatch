#!/bin/bash
#SBATCH --job-name=tgv_prefl
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=logs/prefl_%A_%a.out
#SBATCH --error=logs/prefl_%A_%a.err
# Count-only preflight for the civilian fine-tune grid (Codex 9, point 4): which region x share x draw cells pass the
# positive floors. No GPU, no training. Writes outputs/share_grid/preflight_civilian_{region}.csv
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
RGS=(africa mena europe americas south_asia asia_pacific)
export EVAL_REGION=${RGS[$SLURM_ARRAY_TASK_ID]} EVAL_SEED=123 GRID_ARM=finetune GRID_TARGET=civilian VOCAB_REMAP=1 \
       GRID_PREFLIGHT=1 GRID_SHARES=0.0025,0.005,0.0075,0.01,0.025,0.05,0.10 GRID_DRAWS=0,1,2,3,4 SMOKE=0
unset GRID_ROOT
venv/bin/python scripts/phase4_scaling/share_grid_conflict.py

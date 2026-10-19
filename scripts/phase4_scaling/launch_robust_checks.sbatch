#!/bin/bash
#SBATCH --job-name=tgv_robust
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/robust_%A_%a.out
#SBATCH --error=logs/robust_%A_%a.err
# Appendix sensitivity checks (embedding permutation + reporting-bias measurement error): 6 regions x 3 seeds.
# Needs the share-grid nested outputs (outputs/share_grid/preds_nested_*.npz), which it reproduces before running.
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
export EVAL_REGION=${RGS[$((i % 6))]} EVAL_SEED=${SDS[$((i / 6))]} VOCAB_REMAP=1
unset ROBUST_ROOT ROBUST_DEBUG_DIR
echo "[task] $i RG=$EVAL_REGION SEED=$EVAL_SEED"
python scripts/phase4_scaling/robust_checks.py

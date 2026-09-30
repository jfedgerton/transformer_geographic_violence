#!/bin/bash
#SBATCH --job-name=tgv_teval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/teval_%A_%a.out
#SBATCH --error=logs/teval_%A_%a.err
# transfer_eval over caps x seeds = 4 x 3 = 12 tasks. Requires repr_model_{CAP}_holdout_s{SEED}.pt
# (and ideally the matching africa checkpoint) to exist in data_model/ first.
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate

CAPS=(S M L XL)
SEEDS=(123 456 789)
i=${SLURM_ARRAY_TASK_ID}
export EVAL_CAP=${CAPS[$(( i / 3 ))]}
export EVAL_SEED=${SEEDS[$(( i % 3 ))]}
export EVAL_MECH=both
echo "[teval] task=$i CAP=$EVAL_CAP SEED=$EVAL_SEED host=$(hostname)"
python scripts/phase4_scaling/transfer_eval.py
echo "[teval] task=$i DONE"

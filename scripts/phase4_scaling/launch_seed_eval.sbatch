#!/bin/bash
#SBATCH --array=0-11
#SBATCH --job-name=tgv_sdeval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sdeval_%A_%a.out
#SBATCH --error=logs/sdeval_%A_%a.err
# Seeds 456/789 LORO evals (both modes) + saved Mode B forecasts. Tasks 0-5 = s456, 6-11 = s789.
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(456 789)
RG=${RGS[$((SLURM_ARRAY_TASK_ID % 6))]}; SD=${SDS[$((SLURM_ARRAY_TASK_ID / 6))]}
module load anaconda; source venv/bin/activate
export EVAL_TAG=L_contrastive_e_loro_${RG}_s${SD} EVAL_REGION=${RG}
test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing encoder ${EVAL_TAG}"; exit 1; }
test ! -e outputs/eval_region_${EVAL_TAG}.csv || { echo "output exists, refusing to overwrite"; exit 1; }
echo "[cfg] TAG=${EVAL_TAG} REGION=${RG}"
python scripts/phase4_scaling/transfer_eval_region_fullpreds.py

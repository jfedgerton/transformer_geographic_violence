#!/bin/bash
#SBATCH --array=0-35
#SBATCH --job-name=tgv_veval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/veval_%A_%a.out
#SBATCH --error=logs/veval_%A_%a.err
# LORO seed evals (both modes + saved Mode B forecasts) for all 18 encoders x 2 vocab specs (2026-10-11).
# Task i: region = i % 6, seed = (123,456,789)[(i / 6) % 3], VOCAB_REMAP = 1 for i < 18 (main), 0 for i >= 18 (_vorig).
# Submit only for encoders that are FINAL: pass --array with the right indices, and for encoders still in the retrain
# (job 56237902) use --dependency=afterok on that array task.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789); i=$SLURM_ARRAY_TASK_ID
RG=${RGS[$((i % 6))]}; SD=${SDS[$(((i / 6) % 3))]}
if [ $i -lt 18 ]; then export VOCAB_REMAP=1; SUF=""; else export VOCAB_REMAP=0; SUF="_vorig"; fi
module load anaconda; source venv/bin/activate
export EVAL_TAG=L_contrastive_e_loro_${RG}_s${SD} EVAL_REGION=${RG}
test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing encoder ${EVAL_TAG}"; exit 1; }
test ! -e outputs/eval_region_${EVAL_TAG}${SUF}.csv || { echo "output exists, refusing to overwrite"; exit 1; }
echo "[cfg] task=$i TAG=${EVAL_TAG} REGION=${RG} VOCAB_REMAP=${VOCAB_REMAP} ckpt_mtime=$(stat -c %y data_model/repr_model_${EVAL_TAG}.pt)"
python scripts/phase4_scaling/transfer_eval_region_fullpreds.py

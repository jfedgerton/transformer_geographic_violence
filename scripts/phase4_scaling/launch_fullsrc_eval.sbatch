#!/bin/bash
#SBATCH --array=0-17
#SBATCH --job-name=tgv_fseval
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fseval_%A_%a.out
#SBATCH --error=logs/fseval_%A_%a.err
# 2026-10-20 region eval for the FULL-SOURCE LORO encoders (2026-10-19 rerun), any size.
# Same evaluator as the paper's seed evals (transfer_eval_region_fullpreds.py), now with:
#   Mode B (head <=2020-12, test >=2022-01, forecasts saved), Mode A (pooled CV), and V_val2021 (head <=2020-12,
#   scored on 2021; used ONLY to choose the encoder size); XGBoost head seed fixed at 123; manifest scope check.
# Usage: sbatch --export=ALL,SIZE=L scripts/phase4_scaling/launch_fullsrc_eval.sbatch
#        (SIZE = S, M, L or XL; add --gres=gpu:a40:1 to run on A40s)
# Task i: region = i % 6, seed = (123,456,789)[i / 6]. Refuses to run unless that encoder finished training.
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
: "${SIZE:?set SIZE=S, M, L or XL with --export}"
case "$SIZE" in S|M|L|XL) ;; *) echo "bad SIZE=$SIZE"; exit 1;; esac
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
RG=${RGS[$((i % 6))]}; SD=${SDS[$((i / 6))]}
export EVAL_TAG=${SIZE}_contrastive_e_fullsrc_loro_${RG}_s${SD} EVAL_REGION=${RG} VOCAB_REMAP=1 RANDOM_ENCODER=0
test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing encoder ${EVAL_TAG}"; exit 1; }
grep -qx "FULLSRC_DONE_${EVAL_TAG}" logs/fullsrc_*.out || { echo "training not finished for ${EVAL_TAG}"; exit 1; }
test -f data_intermediate/seen_tokens_fullsrc_loro_${RG}.json || { echo "missing full-source seen tokens for ${RG}"; exit 1; }
test ! -e outputs/eval_region_${EVAL_TAG}.csv || { echo "output exists, refusing to overwrite"; exit 1; }
md5sum scripts/phase4_scaling/transfer_eval_region_fullpreds.py scripts/phase4_scaling/vocab_remap.py
echo "[cfg] TAG=${EVAL_TAG} REGION=${RG} SIZE=${SIZE} SEED=${SD}"
python scripts/phase4_scaling/transfer_eval_region_fullpreds.py
echo FSEVAL_DONE_${EVAL_TAG}

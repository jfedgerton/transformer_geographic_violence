#!/bin/bash
#SBATCH --array=0-17
#SBATCH --job-name=tgv_fsrand
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fsrand_%A_%a.out
#SBATCH --error=logs/fsrand_%A_%a.err
# 2026-10-26 random-encoder control for the full-source encoders: identical to launch_fullsrc_eval.sbatch except
# RANDOM_ENCODER=1 (same architecture, default init from the seed, no checkpoint) -> eval_region_{TAG}_rand.csv, preds_region_{TAG}_rand.npz
# Usage: sbatch --export=ALL,SIZE=M scripts/phase4_scaling/launch_fullsrc_eval_rand.sbatch
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
: "${SIZE:?set SIZE=S, M, L or XL with --export}"
case "$SIZE" in S|M|L|XL) ;; *) echo "bad SIZE=$SIZE"; exit 1;; esac
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789)
i=$SLURM_ARRAY_TASK_ID
RG=${RGS[$((i % 6))]}; SD=${SDS[$((i / 6))]}
export EVAL_TAG=${SIZE}_contrastive_e_fullsrc_loro_${RG}_s${SD} EVAL_REGION=${RG} VOCAB_REMAP=1 RANDOM_ENCODER=1
test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing encoder ${EVAL_TAG}"; exit 1; }
test -f outputs/eval_region_${EVAL_TAG}.csv || { echo "pretrained eval missing for ${EVAL_TAG}"; exit 1; }
test -f data_intermediate/seen_tokens_fullsrc_loro_${RG}.json || { echo "missing full-source seen tokens for ${RG}"; exit 1; }
test ! -e outputs/eval_region_${EVAL_TAG}_rand.csv || { echo "output exists, refusing to overwrite"; exit 1; }
md5sum scripts/phase4_scaling/transfer_eval_region_fullpreds.py scripts/phase4_scaling/vocab_remap.py
echo "[cfg] RANDOM TAG=${EVAL_TAG} REGION=${RG} SIZE=${SIZE} SEED=${SD}"
python scripts/phase4_scaling/transfer_eval_region_fullpreds.py
echo FSRAND_DONE_${EVAL_TAG}

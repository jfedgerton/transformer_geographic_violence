#!/bin/bash
#SBATCH --array=0-17
#SBATCH --job-name=tgv_rand
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rand_%A_%a.out
#SBATCH --error=logs/rand_%A_%a.err
# Random-encoder control (2026-10-13): same LORO eval as launch_eval_vocab.sbatch (main spec, VOCAB_REMAP=1) with the
# pretrained weights replaced by a default-initialized encoder of the same architecture and seed.
# Task i: region = i % 6, seed = (123,456,789)[i / 6].  Americas tasks (3, 9, 15) need 180G. Submit as TWO calls:
#   sbatch --array=0-2,4-8,10-14,16-17 launch_eval_random.sbatch
#   sbatch --array=3,9,15 --mem=184320 launch_eval_random.sbatch
set -o pipefail
R=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd "$R" || { echo "cd failed"; exit 1; }
RGS=(africa mena europe americas south_asia asia_pacific); SDS=(123 456 789); i=$SLURM_ARRAY_TASK_ID
RG=${RGS[$((i % 6))]}; SD=${SDS[$((i / 6))]}
if [ "$RG" = americas ] && [ "${SLURM_MEM_PER_NODE:-0}" -lt 184320 ]; then echo "americas needs --mem=184320 (got ${SLURM_MEM_PER_NODE:-unset})"; exit 1; fi
module load anaconda || { echo "module load failed"; exit 1; }
PY=$R/venv/bin/python; test -x "$PY" || { echo "missing $PY"; exit 1; }
export VOCAB_REMAP=1 RANDOM_ENCODER=1
export EVAL_TAG=L_contrastive_e_loro_${RG}_s${SD} EVAL_REGION=${RG}
grep -q RANDOM_ENCODER scripts/phase4_scaling/transfer_eval_region_fullpreds.py || { echo "fullpreds not patched for RANDOM_ENCODER"; exit 1; }
test -f data_model/repr_model_${EVAL_TAG}.pt || { echo "missing pretrained encoder ${EVAL_TAG}"; exit 1; }
for f in outputs/eval_region_${EVAL_TAG}_rand.csv outputs/preds_region_${EVAL_TAG}_rand.npz; do
  test ! -e "$f" || { echo "output exists, refusing to overwrite: $f"; exit 1; }; done
LOCK=outputs/.lock_rand_${EVAL_TAG}
mkdir "$LOCK" 2>/dev/null || { echo "another job holds $LOCK"; exit 1; }   # atomic; blocks a concurrent duplicate
trap 'rmdir "$LOCK"' EXIT                                                 # released when this job ends
echo "[cfg] task=$i TAG=${EVAL_TAG} REGION=${RG} VOCAB_REMAP=${VOCAB_REMAP} RANDOM_ENCODER=${RANDOM_ENCODER} py=$PY $($PY -c 'import torch,xgboost;print(torch.__version__,xgboost.__version__)')"
"$PY" scripts/phase4_scaling/transfer_eval_region_fullpreds.py
RC=$?; test $RC -eq 0 && test -f outputs/eval_region_${EVAL_TAG}_rand.csv || { echo "eval failed (rc=$RC) or no CSV"; exit 1; }

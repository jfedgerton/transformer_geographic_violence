#!/bin/bash
#SBATCH --job-name=tgv_featsmoke
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=logs/featsmoke_%A_%a.out
#SBATCH --error=logs/featsmoke_%A_%a.err
#
# Smoke: each of the 5 feature sets under MLM, 1 epoch on 20k rows, to validate
# field selection (b/c/f) and the PRIO CLS context token (e/f) run end-to-end.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
SETS=(a b c e f)
FS=${SETS[$SLURM_ARRAY_TASK_ID]}
export REPR_TAG=fsmoke_${FS}
export REPR_CAPACITY=M REPR_VARIANT=holdout REPR_SEED=123 REPR_OBJECTIVE=mlm
export REPR_FEATSET=$FS
export REPR_PANEL=$D/data_intermediate/baseline_features_super.parquet
export REPR_VOCAB=$D/data_intermediate/vocab_super.json
export REPR_MAX_ROWS=20000 REPR_MAX_EPOCHS=1 REPR_WARMUP_STEPS=50 REPR_PATIENCE=1
echo "[featsmoke] FEATSET=$FS"
python scripts/phase4_scaling/train_repr_obj.py
echo "FEATSMOKE_DONE_${FS}"

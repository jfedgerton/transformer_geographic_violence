#!/bin/bash
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o slurm-dftsf2-%j.out
module load anaconda
source venv/bin/activate
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
nvidia-smi
export ARM=scratch_ft
export FRAC=1.0
export SEED=123
export BATCH=32
for i in 1 2 3 4 5; do
  echo "[attempt $i] ARM=$ARM BATCH=$BATCH"
  python scripts/phase4_scaling/dengue_finetune.py && break
  echo "[attempt $i failed, sleeping 20s]"
  sleep 20
done

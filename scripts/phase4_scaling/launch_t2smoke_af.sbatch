#!/bin/bash
#SBATCH --job-name=tgv_t2smk_af
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/t2smoke_af_%j.out
#SBATCH --error=logs/t2smoke_af_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
export REPR2_MODE=both REPR2_CAP=L REPR2_FEAT=e REPR2_SEED=123 REGION=africa SMOKE=1
export CKPT=$D/data_model/repr_model_L_contrastive_e_s123.pt
echo "[t2smoke_af] MODE=$REPR2_MODE REGION=$REGION SMOKE=$SMOKE ckpt=$(basename $CKPT)"
python scripts/phase4_scaling/tier2_crossattn_finetune.py
echo "T2SMOKE_AF_DONE"

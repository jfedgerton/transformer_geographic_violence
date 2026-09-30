#!/bin/bash
#SBATCH --job-name=tgv_mvpsmk
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/mvpsmk_%j.out
#SBATCH --error=logs/mvpsmk_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
export REPR2_MODE=learned REPR2_CAP=L REPR2_FEAT=e REPR2_SEED=123 REGION=africa SMOKE=1 REPR2_K=64 REPR2_HOPS=2
export CKPT=$D/data_model/repr_model_L_contrastive_e_s123.pt
echo "[mvpsmk] MODE=$REPR2_MODE K=$REPR2_K HOPS=$REPR2_HOPS REGION=$REGION SMOKE=$SMOKE"
python scripts/phase4_scaling/tier2_mvp_finetune.py
echo "MVPSMK_DONE"

#!/bin/bash
#SBATCH --array=0-17
#SBATCH --job-name=dom_pre
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dompre_%A_%a.out
#SBATCH --error=logs/dompre_%A_%a.err
# Domain encoders for the appendix grid: crime x 3 seeds (tasks 0-2), dengue 5 macro-regions x 3 seeds (tasks 3-17).
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
SDS=(123 456 789); i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 3 ]; then export DOMAIN=crime SEED=${SDS[$i]}; OUTF=data_model/repr_model_L_contrastive_crime_dom_s${SEED}.pt
else j=$((i - 3)); export DOMAIN=dengue REGION=$((j % 5)) SEED=${SDS[$((j / 5))]}
     OUTF=data_model/repr_model_L_contrastive_dengue_dom_r$((REGION + 1))_s${SEED}.pt; fi
test ! -e $OUTF || { echo "refusing to overwrite $OUTF (archive smoke encoders first)"; exit 1; }
echo "[task] $i DOMAIN=$DOMAIN REGION=${REGION:-na} SEED=$SEED -> $OUTF"
python scripts/phase4_scaling/pretrain_domain.py

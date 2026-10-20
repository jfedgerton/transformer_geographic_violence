#!/bin/bash
#SBATCH --job-name=tgv_lossdown
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/loss_vs_downstream_%j.out
#SBATCH --error=logs/loss_vs_downstream_%j.err
# APPENDIX: encoder validation loss vs downstream PR-AUC across the 54 full-source LORO encoders (S, M, L).
# Run only after all S/M/L encoders AND their region evals (outputs/eval_region_*_fullsrc_loro_*.csv) exist.
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
md5sum scripts/phase4_scaling/loss_vs_downstream.py
python scripts/phase4_scaling/loss_vs_downstream.py

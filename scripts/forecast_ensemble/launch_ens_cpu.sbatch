#!/bin/bash
# Forecasting ensemble, CPU steps. ENS_STEP = preflight_mt | baseline_mt | combine (four-head design; default)
#   or preflight | baseline (single-target design, ENS_KIND=ft for combine).
#   preflight_mt: sbatch --export=ALL,ENS_STEP=preflight_mt,ENS_ENC_SEED=123,ENS_PRETRAIN_CUTOFF_OK=1 launch_ens_cpu.sbatch
#   baseline_mt:  sbatch --export=ALL,ENS_STEP=baseline_mt launch_ens_cpu.sbatch
#   combine (x4): sbatch --export=ALL,ENS_STEP=combine,ENS_TARGET=<incidence|civilian|onset|fatal25> launch_ens_cpu.sbatch
#   preflight: sbatch --export=ALL,ENS_STEP=preflight,ENS_TARGET=civilian,ENS_ENC_SEED=123,ENS_PRETRAIN_CUTOFF_OK=1 launch_ens_cpu.sbatch
#   baseline:  sbatch --export=ALL,ENS_STEP=baseline,ENS_TARGET=civilian launch_ens_cpu.sbatch
#   combine pilot (values with commas go in the submitting environment, not --export):
#              ENS_SEEDS=123,456,789 ENS_STEP=combine ENS_TARGET=civilian sbatch --export=ALL launch_ens_cpu.sbatch
#SBATCH --job-name=ens_cpu
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ens_cpu_%j.out
#SBATCH --error=logs/ens_cpu_%j.err
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
: "${ENS_STEP:?set ENS_STEP}"
echo "[cpu] step=$ENS_STEP target=${ENS_TARGET:-all} seeds=${ENS_SEEDS:-all} enc=${ENS_ENC_SEED:-}"
case "$ENS_STEP" in
  preflight_mt) ENS_PREFLIGHT=1 ENS_WORKERS=0 python scripts/forecast_ensemble/ens_multitask.py ;;
  baseline_mt)  python scripts/forecast_ensemble/ens_baseline_mt.py ;;
  preflight) ENS_PREFLIGHT=1 ENS_WORKERS=0 python scripts/forecast_ensemble/ens_finetune.py ;;
  baseline)  python scripts/forecast_ensemble/ens_baseline.py ;;
  combine)   python scripts/forecast_ensemble/ens_combine.py ;;
  *) echo "bad ENS_STEP $ENS_STEP"; exit 1 ;;
esac

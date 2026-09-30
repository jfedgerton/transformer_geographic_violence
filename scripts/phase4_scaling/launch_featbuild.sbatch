#!/bin/bash
#SBATCH --job-name=tgv_featbuild
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/featbuild_%j.out
#SBATCH --error=logs/featbuild_%j.err
#
# Part C data: build the 9-field superset sequences + vocab_super, then run 05
# (suffix=_super) to add lag/country features -> baseline_features_super.parquet.
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
echo "=== $(date) build_sequences_super ==="
python scripts/build_sequences_super.py
echo "=== $(date) 05 (suffix=_super) ==="
REPR_SEQ_SUFFIX=_super python scripts/05_build_baseline_features.py
echo SUPERBUILD_DONE

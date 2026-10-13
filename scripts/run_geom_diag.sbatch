#!/bin/bash
#SBATCH --job-name=geom_diag
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/geom_diag_%j.out
#SBATCH --error=logs/geom_diag_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load r/4.3.1
Rscript scripts/09_geom_diagnostic.R

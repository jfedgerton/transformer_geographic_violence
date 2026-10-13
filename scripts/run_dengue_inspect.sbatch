#!/bin/bash
#SBATCH --job-name=dengue_inspect
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dengue_inspect_%j.out
#SBATCH --error=logs/dengue_inspect_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load r/4.3.1
Rscript scripts/08_inspect_dengue.R

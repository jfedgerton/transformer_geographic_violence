#!/bin/bash
#SBATCH --job-name=tgv_ucdpfix
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/ucdpfix_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/ucdpfix_%j.err

set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
python scripts/01b_ingest_ucdp.py
echo 'DONE'
python -c "import pandas as pd; df = pd.read_parquet('data_intermediate/ucdp_cleaned.parquet'); print('cols:', list(df.columns)); print('rows:', len(df)); print('type_of_violence dist:', df['type_of_violence'].value_counts().to_dict())"

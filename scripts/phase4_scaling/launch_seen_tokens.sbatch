#!/bin/bash
#SBATCH --job-name=seen_tok
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/seen_tok_%j.out
#SBATCH --error=logs/seen_tok_%j.err
# Builds data_intermediate/seen_tokens_loro_{region}.json (input to vocab_remap.py) and outputs/vocab_unseen_counts.csv.
# CPU only, labels never read. Must finish before any seed eval or share-grid job with VOCAB_REMAP=1.
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
python scripts/phase4_scaling/build_seen_tokens.py

#!/bin/bash
#SBATCH --job-name=tgv_audit
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.err
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda; source venv/bin/activate
python scripts/phase4_scaling/audit_code.py
echo AUDIT_DONE

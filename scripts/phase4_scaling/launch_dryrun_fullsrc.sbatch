#!/bin/bash
#SBATCH --job-name=tgv_dryrun
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/dryrun_fullsrc_%A_%a.out
#SBATCH --error=logs/dryrun_fullsrc_%A_%a.err
# 2026-10-19 count-only dry run for the full-source rerun (no training). Tasks 0-5 = LORO regions, 6 = all regions.
# Each writes logs/dryrun_repr_model_<TAG>.manifest.json and exits before building datasets.
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
python -m py_compile scripts/phase4_scaling/train_repr_obj.py
md5sum scripts/phase4_scaling/train_repr_obj.py
regions=(africa mena europe americas south_asia asia_pacific)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 6 ]; then RG=${regions[$i]}; export REPR_LORO=$RG; export REPR_TAG=dryrun_L_contrastive_e_fullsrc_loro_${RG}_s123
else unset REPR_LORO; export REPR_TAG=dryrun_L_contrastive_e_fullsrc_all_s123; fi
export REPR_VARIANT=full REPR_DRYRUN=1
export REPR_CAPACITY=L REPR_OBJECTIVE=contrastive REPR_FEATSET=e REPR_SEED=123 REPR_LR=3e-4
# panel and vocab left at the script defaults, as in launch_loro*.sbatch (logs: 10,079,405 panel rows, actor vocab 502)
python scripts/phase4_scaling/train_repr_obj.py
echo DRYRUN_DONE_$i

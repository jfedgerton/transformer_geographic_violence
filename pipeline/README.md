# Pipeline (consolidated, Phase 1, 2026-11-02)

Code that reproduces every number, table and figure reported in the main text and appendix of
"Spatial Violence Embeddings for Geographic Transfer Learning". Results that are no longer reported are not covered.

Phase 1 of the consolidation replaced the v5 drivers (`replication_v5/`, 13 files) and the 37 Slurm launchers with two
files. The analysis scripts are unchanged. The v5 layout is kept at git tag `v5-as-run` and stays in the tree until the
review of this phase is finished.

## Layout

```
pipeline/
  run.sh           driver: checks, then submits every job with its resources and dependencies (stages 00-09)
  tasks.sbatch     the only Slurm job script; TASK=<name> selects the job body (one branch per former launcher)
  manifest.csv     every file the pipeline runs, with md5, size and role
  tests/parity_phase1.py   proves run.sh + tasks.sbatch submit and run what the v5 drivers + launchers did (tag v5-as-run)
  tests/v5_baseline_blobs.txt   git blob hashes of the 51 v5 files the test rebuilds from the tag
scripts/           analysis code, byte-identical to the code that produced the reported results
scripts/replication_v5/   code written for the package (figure and table generators, author tools)
```

## Stages

| Stage | What it runs | Reported items |
|---|---|---|
| 00 setup | input and environment checks, provenance log | none |
| 01 build_panel | ACLED + UCDP ingest and merge, PRIO-GRID cell-month panel, sequences, baseline features, PRIO-GRID statics, seen-token lists | Tables 1, 21 inputs |
| 02 pretrain | 18 leave-one-region-out encoders per size (S, M, L) | all |
| 03 frozen_eval | persistence, base, base + embedding, random-weight encoder (S, M, L); month-t persistence (M) | Fig 3, Sec 4.1, Tables 2, 4, 5, 13, 15, 17 |
| 04 finetune_grids | nested grid (M) and top-two and full fine-tuning grids (S, M, L), 7 shares x 5 draws x 3 seeds | Fig 5, Sec 4.2, Tables 6, 18, 19, 21-23 |
| 05 bootstrap | paired block and cell bootstraps | intervals throughout, Tables 11, 12 |
| 06 alerts | equal-budget alert counts, cell files and maps, country rates | Sec 4.1-4.2 counts, App C, App J, Fig 19 |
| 07 appendix | Ukraine (top-two and full arms), CES turnout, crime, dengue | Table 7, Figs 11-15, App D-F, Table 8, App Y |
| 08 brazil | SIM mortality panels, M/S/L Brazil runs, bootstraps | Fig 6, Fig 16, Tables 10, 20, App B, T.3-T.4 |
| 09 figures_tables | all figures and LaTeX tables from the result files | Figs 3-6, 9, 10, 16-18; Tables 2, 4-8, 10-13, 15, 17-23 |

Not generated (hand-made in the manuscript source): Figs 1, 2 and 7 (TikZ), Table 1 (design choices), Table 9 (ICD-10 mapping).

## Running

```
bash pipeline/run.sh all             # every stage, chained with afterok
bash pipeline/run.sh 03 04 05        # a subset; missing upstream outputs are checked on disk
DRY=1 bash pipeline/run.sh all       # run every check and print every sbatch line; submit nothing
venv/bin/python pipeline/tests/parity_phase1.py   # no data needed; works in a scratch copy, submits nothing
```

All checks for the requested stages run before the first job is submitted. If any check fails, nothing is submitted.
Most writing scripts refuse to overwrite their outputs, and the stage checks require those outputs to be absent, so a
rerun needs a clean tree (or archived outputs). A DRY run on a finished tree lists every existing output as `DRY-EXISTS`
in `logs/pipeline_*.log`. Stages 03 and 04 require the line `FULLSRC_DONE_<tag>` in `logs/fullsrc_*.out` for every
encoder (written by the pretraining jobs), so keep `logs/` between stages.

Seeds: encoders 123, 456, 789; XGBoost, training-share draws and bootstraps use fixed seeds set inside the scripts.

## What Phase 1 changed, and how it was checked

- Each former launcher is one `TASK` branch in `tasks.sbatch` with the same environment and command. Launchers that
  differed only by a parameter share a branch: pretraining S/M/L (`SIZE`), frozen eval and random-encoder eval,
  random-encoder and share bootstraps, the Brazil M and S headline runs and their bootstraps (`XD_CAP`).
- Resources (account, partition, GPU, CPUs, memory, time, array, job name, log name) moved from launcher headers to
  `run.sh`, using the effective values of the v5 runs (header plus the v5 driver's overrides).
- Branches of the old Brazil launchers for unreported job sets were not carried over.
- `tasks.sbatch` runs with `set -eo pipefail`. Eleven jobs whose old launchers had no `set -e` (CES, crime and dengue
  grids and pretraining, seen tokens, dengue ingest) now stop if `module load anaconda` fails instead of carrying on.
- `module load anaconda` runs only in the branches whose launchers did it; the Brazil branches do not load it, as before.
- `run_dengue_ingest` now has a job name and writes `logs/dengue_ingest_<jobid>.out` (was `slurm-<jobid>.out` in the root).
- `tests/parity_phase1.py` (strengthened after the Codex review; runs in a temporary copy against stubs, submits nothing):
  - baseline: the v5 drivers and launchers are rebuilt from git tag `v5-as-run` (commit b3ed0e0c28e3) and checked
    against pinned blob hashes, so the test does not need `replication_v5/` or the launchers in the working tree;
  - plan: both drivers run against a stub `sbatch` that records arguments, full environment and working directory:
    77 of 77 submissions, no duplicate names or ids, identical effective options, dependencies, wrapped commands and
    environment at submission;
  - preflight: `pre_01`-`pre_09` identical;
  - bodies: all 77 submissions including the wrapped CPU jobs, every array index, with the full environment logged at
    every python/Rscript call plus every `module` and `activate` call; repeated with `module` failing and with every
    analysis call failing. 1,726 runs; no difference except the pinned `set -e` stops above;
  - rules: `tasks.sbatch` must contain no `#SBATCH` line (resources live in `run.sh`), and every submission must use
    sbatch's default environment export (ALL), the only mode the test models;
  - self-test: a duplicate tables submission, an exported `CRIME_TARGET`, a global `module load`, a changed learning
    rate, a changed memory request, a `#SBATCH` line in `tasks.sbatch` and an `--export=NONE` submission are each caught.
  - not covered: stage 00 (run by every DRY run), login-shell profiles, the analysis scripts themselves (unchanged).

## Phase 2: merged post-processing scripts

Each merge replaces scripts that differed only by a setting with one script. Proof: `pipeline/tests/parity_phase2.py
<case>` (run on Roar; needs the real inputs) runs the old scripts from tag `v5-as-run` and the new script on the same
inputs in a scratch folder; outputs must be byte-identical (PDFs after removing their creation and modification dates),
and so must stdout where it holds no scratch paths. The Phase 1 test maps each merged script back to the scripts it
replaced (`PHASE2_MAP` for a selector, `PHASE2_CHAIN` for a chain of calls in a wrapped job) and requires the merged call
to be present.

| Merged script | Replaces | Selector | Parity (Roar) |
|---|---|---|---|
| `scripts/crossdomain/xd_boot.py` | `xd_m_boot.py`, `xd_s_boot.py` | `XD_CAP=M` / `S` | job 56955161: both outputs and stdout byte-identical |
| `scripts/phase4_scaling/sudan_cells.py` | `audit_cells_export.py`, `audit_cell_country.R` (embedded verbatim, run with `Rscript`), `sudan_cells_fix.py` | none (runs all three steps) | job 56961249: 4 outputs and stdout byte-identical; outputs equal the published files |

Sudan: `country_slice_fullsrc.py` and `country_window_fullsrc.py` (unchanged analysis scripts) still name the old
Sudan scripts in a comment; the file they read, `outputs/sudan_cells_keep.csv`, is unchanged.

Note: rerunning the unchanged old Brazil bootstrap today reproduces the published `xd_m_boot.csv`/`xd_s_boot.csv` values
exactly except in the last printed digits (largest difference about 1e-16; likely floating-point summation order, not verified); reported
values are unaffected.

## Requirements

- Slurm cluster. GPU stages use A100, A40 or P100 GPUs as set in `run.sh`. CPU stages use the `open` account where possible.
- Python environment `venv/` pinned in `env/requirements_frozen_2026-10-19.txt`; `venv_xd/` (adds pyreaddbc) for the SIM
  `.dbc` files. R with tidyverse, sf, scales and patchwork; `scripts/phase4_scaling/Rprofile_cairo` sets the cairo device.
- Paths. `run.sh`, `tasks.sbatch` and the scripts hard-code the project root
  `/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast`. To run elsewhere, replace
  that string (one `sed` command) or create the path as a symlink.

## Data (not included)

| Input | Path | Source |
|---|---|---|
| ACLED events | `data_raw/acled_full.csv` | acleddata.com (terms of use do not allow redistribution) |
| UCDP GED v25.1 | `data_raw/ucdp/GEDEvent_v25_1.csv` | ucdp.uu.se |
| PRIO-GRID statics | `data_raw/priogrid/static/*.json` | prio.org |
| SIM death records | `data_raw/sim_brazil/dores/DOBR*.dbc` | DATASUS FTP, `scripts/crossdomain/fetch_sim_dores.sh` |
| Municipality centroids | `data_raw/br_centroids/municipios.csv` | [verify source] |
| CES cumulative | `data_raw/ces/cumulative_2006-2025.feather` | Kuriwaki, Harvard Dataverse doi:10.7910/DVN/II2DB6 |
| Crime counts | `data_raw/crimecount/code_crime_core_all.parquet` | CODE (Crime Open Database) [verify] |
| Dengue panel | `data_intermediate/dengue_panel_raw.csv.gz` | built from SINAN by R scripts not in this package [verify] |
| Natural Earth 50 m countries | `data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip` (md5 a33f3bf6...) | naturalearthdata.com |
| Natural Earth 10 m, Ukraine view | `data_raw/shapefiles/ne_10m_admin_0_ukr/` | naturalearthdata.com |

## Open items (from the v5 Codex review)

- M2: stage checks do not yet cover every input when an upstream stage is not requested.
- M3: dengue panel builder and the centroid, crime and dengue sources (marked [verify]).

## Departures from how the reported results were first run (unchanged from v5)

- Reported M encoder tasks 10-17 ran on A40 GPUs (same trainer and settings); stage 02 runs all 18 M tasks on A100.
- The Ukraine 2022 window job runs on the free `open` partition.
- The Brazil L column comes from the two earliest grid versions (`xd_grid_brazil.py`, `xd_grid_brazil_v2.py`), run exactly
  as reported; `xd_grid_brazil_v6.py` produced every M and S Brazil result.
- Unreported Brazil runs are not run, so `xd_placebo_boot_v3M.csv` gains `INCOMPLETE` rows for them; reported rows are unchanged.
- `alert_counts_finetune_fullsrc_full.py`, the crime ingest and the Sudan cell chain run as wrapped CPU jobs.

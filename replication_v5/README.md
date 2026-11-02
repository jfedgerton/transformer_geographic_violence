# Replication package (v5, 2026-11-02)

Code that reproduces every number, table and figure reported in the main text and appendix of
"Spatial Violence Embeddings for Geographic Transfer Learning". Results that are no longer reported are not covered.

## Layout

```
replication_v5/
  README.md        this file
  run_all.sh       staged Slurm driver (sources the files below)
  _common.sh       shared helpers (paths, checks, job submission)
  00_setup.sh .. 09_figures_tables.sh   one file per stage
  manifest.csv     every file the package runs, with md5, size and role
scripts/           the analysis code, byte-identical to the code that produced the reported results
scripts/replication_v5/   code written for this package (figure and table generators; see below)
```

`manifest.csv` lists every file the drivers call (directly or through launchers and Python imports) plus config files,
the A40 launcher kept for provenance and the two author tools, each with md5 and role. Everything else in `scripts/`
belongs to earlier generations of the project and is not needed.

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
bash replication_v5/run_all.sh all             # every stage, chained with afterok
bash replication_v5/run_all.sh 03 04 05        # a subset; missing upstream outputs are checked on disk
DRY=1 bash replication_v5/run_all.sh all       # run every check and print every sbatch line; submit nothing
```

All checks for the requested stages run before the first job is submitted. If any check fails, nothing is submitted.
Most writing scripts refuse to overwrite their outputs, and the stage checks require those outputs to be absent, so a
rerun needs a clean tree (or archived outputs). Some scripts overwrite silently (stage 01 builders, crime/dengue ingest,
the Brazil bootstraps, `country_alert_rates.R`, `domain_tp_fp_f0025.py`, `fig_cell_alerts.R`); the checks also cover
their outputs where a stale file would mislead. A DRY run on a finished tree lists every existing output as `DRY-EXISTS`
in `logs/run_all_v5_*.log`. DRY mode submits nothing but still writes its log and `logs/pip_freeze_run_all_v5.txt`, and
it needs `sbatch` on the path (stage 00 checks for it).

Completion markers. Stages 03 and 04 also require the line `FULLSRC_DONE_<tag>` in `logs/fullsrc_*.out` for every
encoder (written by the pretraining launchers), so keep `logs/` between stages.

Partial runs. `make_tables.py --ukraine-full` (stage 09) needs the full-arm Ukraine files from stage 07; run 09 after 07.

Seeds: encoders 123, 456, 789; XGBoost, training-share draws and bootstraps use fixed seeds set inside the scripts (123
unless stated in the script header).

## Requirements

- Slurm cluster. GPU stages (02, 04, the CES/crime/dengue grids and Brazil runs) use A100 or A40 GPUs as set in each
  launcher header. CPU stages use the `open` account where possible.
- Python environment `venv/` pinned in `env/requirements_frozen_2026-10-19.txt` (versions in `env/versions_2026-10-19.txt`);
  `venv_xd/` (adds pyreaddbc) for reading the SIM `.dbc` files. R with tidyverse (ggplot2, dplyr, readr, tidyr), sf,
  scales and patchwork; `scripts/phase4_scaling/Rprofile_cairo` sets the cairo bitmap device for headless nodes.
  The alert-chain launcher skips Fig 19 (and still exits 0) if an R package is missing; stage 09 then fails on purpose.
- `launch_xd_brazil.sbatch` (Brazil L column) requests a P100 GPU, as originally run.
- `scripts/00_master.py` holds the panel configuration imported by every stage 01 script.
- Paths. Every launcher and script hard-codes the project root
  `/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast`. The scripts are kept
  byte-identical to the versions that produced the results, so they were not rewritten. To run elsewhere, replace that
  string in `scripts/` and `replication_v5/_common.sh` (one `sed` command) or create the path as a symlink.

## Data (not included)

| Input | Path | Source |
|---|---|---|
| ACLED events | `data_raw/acled_full.csv` | acleddata.com (terms of use do not allow redistribution) |
| UCDP GED v25.1 | `data_raw/ucdp/GEDEvent_v25_1.csv` | ucdp.uu.se |
| PRIO-GRID statics | `data_raw/priogrid/static/*.json` | prio.org |
| SIM death records | `data_raw/sim_brazil/dores/DOBR*.dbc` | DATASUS FTP, `scripts/crossdomain/fetch_sim_dores.sh` |
| Municipality centroids | `data_raw/br_centroids/municipios.csv` | municipality list with centroid coordinates [verify source] |
| CES cumulative | `data_raw/ces/cumulative_2006-2025.feather` | Kuriwaki, Harvard Dataverse doi:10.7910/DVN/II2DB6 |
| Crime counts | `data_raw/crimecount/code_crime_core_all.parquet` | CODE (Crime Open Database) [verify] |
| Dengue panel | `data_intermediate/dengue_panel_raw.csv.gz` | built by `scripts/08-11_*.R` from SINAN [verify] (not run by this package) |
| Natural Earth 50 m countries | `data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip` (md5 a33f3bf6...) | naturalearthdata.com |
| Natural Earth 10 m, Ukraine view | `data_raw/shapefiles/ne_10m_admin_0_ukr/` | naturalearthdata.com |

Derived panels, sequences and pretrained encoders are not included because they are built from ACLED events.

## Code written for this package (`scripts/replication_v5/`)

- `make_tables.py` writes every generated table as `figures_tables/tables_v5/tab_<label>.tex` from the result CSVs.
- `fig_cell_maps.R` (Fig 4), `fig_random_encoder.R` (Fig 9), `fig_share_curve.R` (Fig 10), `fig_share_regions.R`
  (Fig 17), `fig_persistence_gap.R` (Fig 18): the scripts behind these figures were not kept, so they were rewritten to
  read the reported result files; each checks its plotted values against the corresponding table.
- `fig_ukraine_alert_gain_arm.R`: copy of `scripts/phase4_scaling/fig_ukraine_alert_gain.R` whose output names take the
  fine-tuning arm from `UKR_ARM`, so the top-two and full-arm figures sit side by side.
- `check_tables.py` (author check, not run by the pipeline) compares the generated tables with the manuscript source.

## Departures from how the reported results were first run

- Reported M encoder tasks 10-17 ran through `launch_fullsrc_M_a40.sbatch` (same trainer and settings on A40 GPUs;
  shipped for provenance); stage 02 runs all 18 M tasks through `launch_fullsrc_M.sbatch`.
- `launch_country_window_fullsrc.sbatch` is submitted to the free `open` partition (its header says `basic`).
- The Brazil L column (Table 20) comes from the two earliest grid versions, run exactly as reported:
  `xd_grid_brazil.py` via `launch_xd_brazil.sbatch` (ft_rich tasks 1, 5, 9) and `xd_grid_brazil_v2.py` via
  `launch_xd_ladder.sbatch` (c_nested tasks 0-2). `xd_grid_brazil_v6.py` produced every M and S Brazil result.
- Unreported Brazil runs (thinned-homicide controls, c4/geo variants, the S placebo grid) are not run, so
  `xd_placebo_boot_v3M.csv` gains `INCOMPLETE` rows for those comparisons; the reported rows are unchanged.
- `alert_counts_finetune_fullsrc_full.py` had no launcher; stage 06 runs it as a wrapped CPU job.
- `audit_cells_export.py`, `audit_cell_country.R` and `sudan_cells_fix.py` run in stage 07 because the Ukraine slice and window scripts read its output at import; Sudan
  results are not reported.
- Launchers that also produce a few unreported files are run unchanged: the Brazil placebo prep builds a dengue panel and
  a cause profile, the 10 percent Ukraine map export also renders four period maps (only its cell file feeds Fig 12),
  and `run_dengue_ingest.sbatch` writes its log as `slurm-<jobid>.out` in the project root.

## Output names versus manuscript figure files

| Manuscript file (`figures/`) | Produced by the package |
|---|---|
| `fig_loro_gains.pdf` (Fig 3) | `figures_tables/v5/fig3_loro_gains/fig_loro_gains.pdf` |
| `fig_cell_alerts_maps_pers_nested_v2.png` (Fig 4) | `figures_tables/v5/fig4_cell_maps/fig_cell_alerts_maps_pers_nested_M_fullsrc.png` |
| `fig_finetune_gains.pdf` (Fig 5) | `figures_tables/v5/fig5_finetune_gains/fig_finetune_gains.pdf` |
| `fig_placebo_gradient_ab_M.pdf` (Fig 6) | `figures_tables/v5/fig6_brazil/fig_placebo_gradient_ab_M.pdf` |
| `appendix_random_temporal_pooled.png` (Fig 9) | `figures_tables/v5/fig9_random_encoder/fig_random_encoder.png` |
| `fig_boot_shares.png` (Fig 10) | `figures_tables/v5/fig10_share_curve/fig_boot_shares.png` |
| `fig_ukraine_f0025_{1..4}_*.pdf` (Figs 11, 13-15) | `outputs/ukraine_maps_M_f0025/fig_ukraine_f0.0025_M_fullsrc_{top2,full}_{1..4}_*.pdf` |
| `fig_ukraine_alert_gain.pdf` (Fig 12) | `outputs/ukraine_alert_gain_M_{top2,full}/fig_ukraine_alert_gain_M_fullsrc_{top2,full}.pdf` |
| `fig_placebo_gradient.pdf` (Fig 16) | `figures_tables/v5/fig16_brazil_random/fig_placebo_gradient_M.pdf` |
| `fig_boot_share_regions.png` (Fig 17) | `figures_tables/v5/fig17_share_regions/fig_boot_share_regions.png` |
| `fig_boot_persistence.png` (Fig 18) | `figures_tables/v5/fig18_persistence_gap/fig_boot_persistence.png` |
| `fig_cell_alerts_countries_wide_M_fullsrc.png` (Fig 19) | `outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png` |

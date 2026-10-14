# Transfer learning for grid-level political violence forecasts

Code for "Learning the Geography of Political Violence: Transfer of Conflict Forecasts Across World Regions."
A transformer encoder is pretrained on event sequences from five world regions and transferred to the sixth
(leave-one-region-out, LORO). Units are PRIO-GRID 0.5-degree cell-months; the target is any violent event
(battles, explosions/remote violence, violence against civilians) in the cell in month t+1; forecasts use
information through month t. Main metric is PR-AUC on forecasts made January 2022 to April 2025.

**Status.** This branch is a snapshot of the working pipeline before simplification. A consolidated layout
(one config, shared library, numbered steps, regression tests) is planned. Older generations of code are kept in
`scripts/phase4_scaling/archive_*` and `archive/` for provenance; nothing is deleted.

## Data (not included)
No raw or derived event data, model checkpoints or forecasts are in this repository.
- ACLED events: obtain under your own ACLED licence (https://acleddata.com). ACLED's terms do not allow
  redistribution of the data or of models from which it could be extracted.
- UCDP GED (CC BY 4.0), PRIO-GRID static covariates, Natural Earth boundaries: public; see `DATA.md`.
The pipeline merges ACLED and UCDP events without cross-source deduplication (documented in the paper).

## Pipeline and where each result comes from
All paths below are relative to `scripts/`. Jobs ran on Penn State Roar (Slurm); launchers are the `.sbatch` files.

| Step | Script(s) | Output |
|---|---|---|
| Ingest and merge events | `01_ingest_acled.py`, `01b_ingest_ucdp.py`, `01c_merge_sources.py` | event table |
| Grid and cell-month panel | `02_build_grid.py`, `03_aggregate_location_month.py` | panel with t+1 target |
| Sequences and features | `build_sequences_super.py`, `05_build_baseline_features.py`, `build_priogrid_static.py` | encoder inputs, base features |
| Seen-token sets (vocabulary remap) | `phase4_scaling/build_seen_tokens.py`, `vocab_remap.py` | per-region token sets |
| Pretraining (18 LORO encoders) | `phase4_scaling/train_repr_obj.py` + `launch_loro*.sbatch` | encoder checkpoints |
| Main evaluation (Tables 1, 3; civilian table; random-encoder appendix) | `phase4_scaling/transfer_eval_region_fullpreds.py` + `launch_eval_vocab.sbatch`, `launch_eval_random.sbatch` | per-run eval CSVs and saved forecasts |
| Persistence with month-t violence | `phase4_scaling/persistence_month_t.py` | `persistence_month_t.csv` |
| Limited local data (Table 2) and fine-tune vs scratch (appendix) | `phase4_scaling/share_grid_conflict.py`, `make_share_tables.py` | share-grid metrics and forecasts |
| Equal-alert-budget counts (Results text) | `phase4_scaling/alert_counts.py` | `alert_counts_*.csv` |
| Bootstrap intervals (cell and 5-degree spatial blocks) | `phase4_scaling/bootstrap_ci.py` + `launch_bootstrap_ci.sbatch` | `bootstrap_ci_{cell,block5}.csv` |
| Random-encoder comparison | `phase4_scaling/cmp_random_v2.py` | `random_control_summary.csv` |
| Cell-level alert maps | `phase4_scaling/cell_alerts_agg.py`, `../figures_tables/fig_cell_alerts.R` | map figure |
| Tables and figures | `../figures_tables/make_tables_v2.py`, `fig_results_main.R`, `fig_results_options.R`, `fig_share_curves.R`, `manuscript_style.R` | `.tex` tables, figures |
| Appendix domains (CES turnout, crime, dengue) | `phase4_scaling/prep_ces.py`, `pretrain_ces.py`, `share_grid_ces.py`, `pretrain_domain.py`, `share_grid_domain.py` | appendix tables |
| Encoder design selection (appendix) | `phase4_scaling/train_repr_obj.py`, `transfer_eval_full.py`, `launch_grid.sbatch` | design comparison |

Pending code review before results are reported: the Ukraine/Sudan onset demonstration (`onset_transfer.py`),
an embedding permutation test, and a reporting-bias measurement-error check.

## Conventions
- Seeds 123, 456, 789 (encoders); XGBoost settings fixed (depth 6, eta 0.1, subsample 0.8, colsample 0.8, 200 rounds).
- Main specification maps tokens unseen in an encoder's pretraining data to `<UNK>` (`VOCAB_REMAP=1`); the original
  vocabulary is an appendix sensitivity check (`_vorig` outputs).
- Scripts refuse to overwrite completed outputs and check saved forecasts against reported scores before use.

## Known issues (to be fixed in the consolidated version)
- `months_since_last_event` is off by one in quiet months (`05_build_baseline_features.py`); it affects base and
  base + embedding identically.
- The XGBoost seed equals the encoder seed in the main evaluator.
- Several R figure scripts expect `manuscript_style.R` one directory up.

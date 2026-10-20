#!/bin/bash
# Builds replication/ as numbered symlinks to the existing code, in run order.
# Moves nothing, so running jobs are unaffected. Prints any missing source; exits 1 if any are missing.
set -u
ROOT=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd "$ROOT"
P=scripts/phase4_scaling; E=scripts/forecast_ensemble
mkdir -p replication
MAP=(
"00_env_requirements.txt|env/requirements_frozen_2026-10-19.txt"
"00_env_versions.txt|env/versions_2026-10-19.txt"
"01a_build_baseline_features.py|$P/05_build_baseline_features.py"
"01b_build_sequences_super.py|$P/build_sequences_super.py"
"02a_design_selection_grid.sbatch|$P/launch_grid.sbatch"
"02b_design_selection_eval.sbatch|$P/launch_eval.sbatch"
"02c_design_selection_eval.py|$P/transfer_eval_full.py"
"02d_design_selection_aggregate.py|$P/aggregate_and_plot_eval.py"
"03a_lr_sweep_by_objective.sbatch|$P/launch_lrsweep_obj.sbatch"
"03b_lr_probe.sbatch|$P/launch_lrprobe.sbatch"
"03c_lr_probe_xl_subsample.sbatch|$P/launch_lrprobe_xlsub.sbatch"
"04_data_scope_dryrun.sbatch|$P/launch_dryrun_fullsrc.sbatch"
"05a_pretrain_encoder.py|$P/train_repr_obj.py"
"05b_pretrain_S.sbatch|$P/launch_fullsrc_S.sbatch"
"05c_pretrain_M.sbatch|$P/launch_fullsrc_M.sbatch"
"05d_pretrain_L.sbatch|$P/launch_fullsrc_L.sbatch"
"05e_pretrain_XL.sbatch|$P/launch_fullsrc_XL.sbatch"
"05f_pretrain_all_region_L.sbatch|$P/launch_fullsrc_all.sbatch"
"05h_pretrain_M_a40.sbatch|$P/launch_fullsrc_M_a40.sbatch"
"05g_regions.py|$P/regions.py"
"06a_build_seen_tokens.py|$P/build_seen_tokens_fullsrc.py"
"06c_build_seen_tokens.sbatch|$P/launch_seen_tokens_fullsrc.sbatch"
"06b_vocab_remap.py|$P/vocab_remap.py"
"07_persistence.py|$P/persistence_month_t.py"
"08a_nested_pooled_eval.py|$P/transfer_eval_region_fullpreds.py"
"08b_nested_pooled_eval.sbatch|$P/launch_fullsrc_eval.sbatch"
"08c_loss_vs_downstream.py|$P/loss_vs_downstream.py"
"08d_loss_vs_downstream.sbatch|$P/launch_loss_vs_downstream.sbatch"
"09a_random_encoder.sbatch|$P/launch_eval_random.sbatch"
"09b_random_encoder_compare.py|$P/cmp_random_v2.py"
"10a_share_grid.py|$P/share_grid_conflict.py"
"10b_share_grid.sbatch|$P/launch_grid_conflict.sbatch"
"10c_share_tables.py|$P/make_share_tables.py"
"11a_bootstrap_ci.py|$P/bootstrap_ci.py"
"11b_bootstrap_finetune.py|$P/bootstrap_ft.py"
"12a_country_slice.py|$P/country_slice.py"
"12b_country_window.py|$P/country_window.py"
"12c_country_window.sbatch|$P/launch_country_window.sbatch"
"13_robustness_permutation.py|$P/robust_checks.py"
"13_robustness_permutation.sbatch|$P/launch_robust_checks.sbatch"
"14a_cell_alerts.py|$P/cell_alerts_agg.py"
"14b_cell_alerts_persistence.py|$P/cell_alerts_persistence.py"
"14c_cell_month_decompose.py|$P/cell_month_decompose.py"
"14d_alert_profile.py|$P/alert_profile.py"
"14e_alert_budget_sweep.py|$P/alert_budget_sweep.py"
"15a_pretrain_ces.py|$P/pretrain_ces.py"
"15b_share_grid_ces.py|$P/share_grid_ces.py"
"15c_pretrain_crime_dengue.py|$P/pretrain_domain.py"
"15d_share_grid_crime_dengue.py|$P/share_grid_domain.py"
"15e_domain_tp_fp.py|scripts/domain_tp_fp_f0025.py"
"16a_ensemble_multitask.py|$E/ens_multitask.py"
"16b_ensemble_baseline.py|$E/ens_baseline_mt.py"
"16c_ensemble_combine.py|$E/ens_combine.py"
"16d_ensemble_bootstrap.py|$E/ens_boot.py"
"16e_ensemble_alerts.py|$E/ens_alerts.py"
"16f_ensemble_alerts_union.py|$E/ens_alerts_union.py"
"16g_ensemble_alerts_totals.py|$E/ens_alerts_totals.py"
"17_README_DATA_SCOPE.md|$P/README_DATA_SCOPE.md"
)
missing=0
for m in "${MAP[@]}"; do
  name=${m%%|*}; src=${m#*|}
  if [ ! -e "$src" ]; then echo "MISSING: $name <- $src"; missing=1; continue; fi
  ln -sfn "../$src" "replication/$name"
done
ls -l replication | awk 'NR>1{print $9, $10, $11}'
echo "TODO (not yet in the repo): R figure scripts (fig_maps_2row.R, fig_cell_alerts.R, Ukraine maps) -> 18_figures_*"
[ $missing -eq 0 ] && echo "ALL LINKED" || { echo "SOME SOURCES MISSING (see above)"; exit 1; }

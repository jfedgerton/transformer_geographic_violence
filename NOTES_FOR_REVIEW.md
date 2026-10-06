# Notes for code review (Codex or otherwise)

This repo was migrated from an HPC working directory that accumulated
many months of exploratory runs. Code-only; data/checkpoints/logs are
gitignored (see DATA.md). Before treating everything here as uniformly
load-bearing for the current manuscript, note the following.

## Confirmed still in use
- scripts/00-05*.py: core ingestion/build pipeline
- scripts/ucdp_release/*: UCDP-only HF release pipeline (ACLED-guarded)
- scripts/phase4_scaling/train_repr_obj.py, train_repr_scaling.py:
  representation training, feeds the manuscript's LORO tables
- scripts/phase4_scaling/transfer_eval*.py, transfer_metrics.py,
  bootstrap_transfer_ci.py: LORO evaluation + CIs
- scripts/phase4_scaling/interp_ci.py: LORO defense / interpretability

## Status unclear, needs a fresh check against the manuscript
- scripts/phase4_scaling/*crossattn*, tier2_crossattn_finetune.py: not
  confirmed to be in the current manuscript's tables. May be an
  abandoned architecture variant. Author (Jared) set this project down
  for several months and can't confirm from memory; please check
  whether these are referenced by anything in scripts/phase4_scaling/
  that IS confirmed in use above, or whether they're an orphaned branch.

## Confirmed scrapped / superseded (kept in repo, not deleted)
- scripts/latam_me_transfer_v1.py: early 2-region transfer probe,
  superseded by the full 6-region LORO design (see manuscript Table 1)
- scripts/fews_transfer_v1.py: cross-domain (conflict -> food
  insecurity) transfer probe. Negative result, but confounded by a
  known data issue (approximate admin-to-grid centroid mapping,
  documented in the script's own docstring). Not trustworthy as-is.
- scripts/count_crps_v1-v4.py, crps_v4_dimcheck.py: CRPS-based
  incremental-validity chain. Candidate appendix material (see
  outputs/crps_v2/v3/v4 results) but predates the current design and
  is single-region (Africa/ME) only.
- scripts/phase4_scaling/crime_finetune.py, crime_ingest.py:
  cross-domain (conflict -> US crime) transfer probe. Candidate
  appendix material as a scope-boundary null result.

## Never actually run
- protest_transfer_v1.py existed only in a Dropbox working folder, not
  on the HPC cluster, and has no corresponding output file anywhere.
  It is NOT included in this repo.

## Excluded on purpose
- archive/pre_20260618/, _backup_phase1_20260528/, _patch_backup_20260914/:
  explicitly named backup/superseded snapshots
- *.b64 files (cf.b64, ci_v2.b64, ic.b64, scripts/.xlat.b64,
  scripts/.xv3.b64, scripts/.xv4.b64): base64-encoded duplicates of
  plain .py files already present elsewhere in the repo. Verified by
  decoding; not malicious, just redundant.

## Update 2026-10-06 (audit + archive)
- Provenance of every manuscript table is in RESULTS_MANIFEST.md (copy of outputs/MANIFEST.md on Roar).
- Table 2 (pretrained+1% vs scratch) used the June Gen-2 encoder (repr_model_L_holdout_s123, MLM objective, 10-unit holdout, ROC-AUC), not the LORO contrastive-e encoder behind Tables 1/3/4. Re-run under the LORO design with PR-AUC: scripts/phase4_scaling/transfer_eval_region_ft.py + launch_region_ft.sbatch (job 56128289). Output: outputs/ft_region_L_contrastive_e_loro_{region}_s123.csv.
- Gen 1-3 pilot checkpoints, lr/tune/smoke checkpoints, embedding diagnostics, CRPS chain, June transfer pilots, and the fews/latam_me probes were moved (not deleted) to archive/gen1-3_pre_loro_20261006/ on Roar. The tracked CSVs among them show as deleted in this commit; they remain in git history and in the Roar archive.
- outputs/figures/heatmap_*_e.png are dated 2026-06-25 and predate the LORO runs; regenerate before use.
- Cross-domain probes (dengue, crime, incl. *_lnopen LayerNorm ablations) are closed as null results; see Dropbox RESULTS_cross_domain_transfer_NEGATIVE.md.

# Manuscript results manifest (audited 2026-10-06)
Settled manuscript = Dropbox apsr_manuscript.zip (2026-09-29). Each row: table -> source file -> encoder -> script.
Do not re-adjudicate which file feeds which table; update this file if a table is re-run.

| Table | Result | Source file | Encoder checkpoint | Script | Metric | Status |
|---|---|---|---|---|---|---|
| 1 | LORO zero-shot emb_base vs persistence, Mode B, 6 regions | outputs/loro_region_summary.csv (Jul 16; eval files re-run Sep 14-15) | data_model/repr_model_L_contrastive_e_loro_{region}_s123.pt (Jul 13-15) | transfer_eval_region.py -> aggregate_region_eval.py | PR-AUC | CURRENT |
| 3 | Nested emb_base vs base, both modes, 12/12 | same as Table 1 | same | same | PR-AUC | CURRENT |
| 4 | Civilian target scope check | outputs/loro_region_summary_civilian.csv (Sep 16) | same | same | PR-AUC | CURRENT |
| 2 | Pretrained+1% vs scratch 100%, 10 units | outputs/transfer_eval_L_s123.csv (Jun 7) | data_model/repr_model_L_holdout_s123.pt (Jun 5, MLM objective) | transfer_eval.py | ROC-AUC | STALE - older encoder and design; RERUN with LORO contrastive-e encoder, PR-AUC |
| 5 | AR-objective robustness, 15 featset x seed | outputs/eval_full_M_autoregressive_{a,b,c,e,f}_s{123,456,789}.csv (Jun 24) | data_model/repr_model_M_autoregressive_*_s*.pt | eval_full (Gen 3 sweep) | PR-AUC-type score | Gen 3, M capacity, full panel (not LORO) - decide whether to rerun at L/LORO |
| - | Null encoders (shuffled/random) and region probe | outputs/interp_defense_summary.csv (Jul 20) | LORO encoders | interp_ci.py INTERP_MODE=interp | PR-AUC | Not in manuscript; candidate appendix |
| - | Bootstrap CIs (different spec: 250k subsample, 300-round head) | outputs/loro_region_ci.csv (Jul 20) | LORO encoders | interp_ci.py INTERP_MODE=ci | PR-AUC | Do not cite for Table 1/3 contrasts |
| fig | heatmap_emb_vs_pers_e.png, heatmap_embbase_vs_base_e.png | outputs/figures/ (Jun 25) | Gen 3 full-panel sweep | - | - | STALE - predate LORO; regenerate from loro_region_summary.csv |

Archived 2026-10-06 (moves only): archive/gen1-3_pre_loro_20261006/ (Gen 1 pilots, smoke/lr/tune checkpoints, embedding diagnostics, CRPS chain, June transfer pilots, fews/latam_me probes).
Cross-domain null results (crime, dengue): see Dropbox RESULTS_cross_domain_transfer_NEGATIVE.md.

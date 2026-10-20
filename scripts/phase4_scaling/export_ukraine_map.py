"""export_ukraine_map.py -- data for the Ukraine forecast map (fig:ukraine_invasion), fine-tuning exercise (7b).

Step 1, choose the share (rule fixed 2026-10-20, before the full-source results exist):
  For each training share, average the 15 fine-tuned runs (3 encoder seeds x 5 draws) of the Europe hold-out
  models, for both the pretrained and the from-scratch network, on Ukraine's rows. Score both averaged forecasts on
  the invasion window (target months 2022-02..2022-07) by PR-AUC. The chosen share is the one with the largest
  PR-AUC gain (pretrained minus scratch); ties go to the smaller share. --share overrides the rule.
  Also reported, not used for the choice: violent cell-months caught at equal alert budgets (budget = violent
  cell-months in the window) and persistence PR-AUC.
Step 2, export the chosen share's Ukraine cell-months (all test months) for fig_ukraine_map.R.

Inputs : {grid_dir}/preds_finetune_europe_s{seed}_f{share}{suffix}.npz  (share_grid_conflict.py, ARM=finetune)
         data_intermediate/baseline_features.parquet (country of each cell-month)
         data_intermediate/acled_with_grid.parquet   (check that y is violence in the month AFTER the row month)
Outputs: {out_dir}/ukraine_share_selection{suffix}.csv   one row per share (the rule's inputs and result)
         {out_dir}/ukraine_cells_f{share}{suffix}.csv     grid_id, year_month (origin), target_month, y, p_mean,
                                                          p_sd, p_scratch, persistence, lat, lon
Never overwrites. Run: python scripts/phase4_scaling/export_ukraine_map.py [--share auto|0.005]
                       [--grid-dir outputs/share_grid] [--suffix ""] [--out-dir outputs/ukraine_maps]
Added 2026-10-20.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
ap = argparse.ArgumentParser()
ap.add_argument("--share", default="auto")
ap.add_argument("--grid-dir", default=f"{R}/outputs/share_grid")
ap.add_argument("--suffix", default="")
ap.add_argument("--out-dir", default=f"{R}/outputs/ukraine_maps")
ap.add_argument("--panel", default=f"{R}/data_intermediate/baseline_features.parquet")
ap.add_argument("--events", default=f"{R}/data_intermediate/acled_with_grid.parquet")
a = ap.parse_args()

SHARES = ["0.0025", "0.005", "0.0075", "0.01", "0.025", "0.05", "0.1"]
SEEDS = [123, 456, 789]; DRAWS = range(5)
WINDOW = ["2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07"]   # target months
N_UKR, POS_UKR = 13200, 4745                                                   # country_slice_incidence.csv
os.makedirs(a.out_dir, exist_ok=True)
sel_path = f"{a.out_dir}/ukraine_share_selection{a.suffix}.csv"
assert a.share == "auto" or a.share in SHARES, f"--share must be auto or one of {SHARES}"
assert not os.path.exists(sel_path), f"refusing to overwrite {sel_path}"

# ---- 1. which rows are Ukraine ----
cm = pd.read_parquet(a.panel, columns=["grid_id", "year_month", "country"])
cm["year_month"] = cm["year_month"].astype(str).str[:7]
UKR_KEYS = set(zip(cm.loc[cm["country"] == "Ukraine", "grid_id"].astype(np.int64), cm.loc[cm["country"] == "Ukraine", "year_month"]))
assert UKR_KEYS, "no Ukraine rows in the panel"


def load_share(sh):
    """Averaged pretrained and scratch forecasts on Ukraine rows for one share (15 runs each)."""
    P, S, ref = [], [], None
    for sd in SEEDS:
        f = f"{a.grid_dir}/preds_finetune_europe_s{sd}_f{sh}{a.suffix}.npz"
        assert os.path.exists(f), f"missing {f}"
        z = np.load(f, allow_pickle=True)
        g, ym, y = z["__grid_id"].astype(np.int64), z["__year_month"].astype(str), z["__y"].astype(int)
        if ref is None:
            ref = (g, ym, y, z["persistence|0|-1"].astype(float))
        assert (g == ref[0]).all() and (ym == ref[1]).all() and (y == ref[2]).all(), f"row order differs at seed {sd}, share {sh}"
        for k in DRAWS:
            P.append(z[f"ft_pretrained|{sh}|{k}"].astype(float)); S.append(z[f"ft_scratch|{sh}|{k}"].astype(float))
    P, S = np.vstack(P), np.vstack(S)
    assert P.shape[0] == 15 and S.shape[0] == 15 and np.isfinite(P).all() and np.isfinite(S).all()
    d = pd.DataFrame(dict(grid_id=ref[0], year_month=ref[1], y=ref[2], persistence=ref[3],
                          p_mean=P.mean(0), p_sd=P.std(0), p_scratch=S.mean(0)))
    d = d[[(gi, m) in UKR_KEYS for gi, m in zip(d.grid_id, d.year_month)]].reset_index(drop=True)
    assert len(d) == N_UKR and d.y.sum() == POS_UKR, f"share {sh}: Ukraine rows {len(d)} / positives {d.y.sum()} do not match the slice"
    assert not d.duplicated(["grid_id", "year_month"]).any()
    d["target_month"] = (pd.PeriodIndex(d.year_month, freq="M") + 1).strftime("%Y-%m")
    return d


def hits(y, p, k):
    """Violent cell-months among the k highest forecasts (stable order on ties)."""
    return int(y[np.argsort(-p, kind="stable")[:k]].sum())


# ---- 2. apply the selection rule (all shares are scored even when --share is given) ----
rows, cache = [], {}
for sh in SHARES:
    d = load_share(sh); cache[sh] = d
    w = d[d.target_month.isin(WINDOW)]
    assert sorted(w.target_month.unique()) == WINDOW
    y = w.y.to_numpy(); k = int(y.sum())
    rows.append(dict(share=float(sh), n=len(w), violent=k,
                     pr_pretrained=average_precision_score(y, w.p_mean), pr_scratch=average_precision_score(y, w.p_scratch),
                     pr_persistence=average_precision_score(y, w.persistence),
                     hits_pretrained=hits(y, w.p_mean.to_numpy(), k), hits_scratch=hits(y, w.p_scratch.to_numpy(), k)))
sel = pd.DataFrame(rows)
sel["pr_gain"] = sel.pr_pretrained - sel.pr_scratch
sel["hits_gain"] = sel.hits_pretrained - sel.hits_scratch
best = sel.sort_values(["pr_gain", "share"], ascending=[False, True]).iloc[0]
rule_share = f"{best.share:g}"
chosen = rule_share if a.share == "auto" else a.share
sel["chosen_by_rule"] = sel.share == best.share
sel["used_for_map"] = sel.share == float(chosen)
print(sel.round(4).to_string(index=False), flush=True)
print(f"[rule] largest Feb-Jul 2022 PR-AUC gain: share {rule_share}; map uses share {chosen}", flush=True)

# ---- 3. check the outcome timing: y must be violence in the month after the row month ----
d = cache[chosen]
ev = pd.read_parquet(a.events, columns=["grid_id", "year_month", "event_type"])
ev["year_month"] = ev["year_month"].astype(str).str[:7]
ev = ev[ev.event_type.isin(["Battles", "Explosions/Remote violence", "Violence against civilians"]) & ev.grid_id.isin(d.grid_id.unique())]
S = set(zip(ev.grid_id.astype(np.int64), ev.year_month))
plus1 = np.array([(g, m) in S for g, m in zip(d.grid_id, d.target_month)]).astype(int)
agree = (plus1 == d.y.to_numpy()).mean()
print(f"[check] y vs violent events in the target month: agreement {agree:.4f}", flush=True)
assert agree > 0.99, "y does not line up with next-month violent events"

# ---- 4. write ----
row = (d.grid_id - 1) // 720; col = (d.grid_id - 1) % 720          # PRIO-GRID gid -> 0.5-degree centroid
d["lat"] = -90 + 0.5 * row + 0.25; d["lon"] = -180 + 0.5 * col + 0.25
assert d.lat.between(43.5, 53).all() and d.lon.between(21.5, 41).all(), "cells fall outside Ukraine's bounding box"
cells_path = f"{a.out_dir}/ukraine_cells_f{chosen}{a.suffix}.csv"
assert not os.path.exists(cells_path), f"refusing to overwrite {cells_path}"
sel.to_csv(sel_path + ".part", index=False); os.replace(sel_path + ".part", sel_path)
d.to_csv(cells_path + ".part", index=False); os.replace(cells_path + ".part", cells_path)
print(f"[done] {sel_path}\n[done] {cells_path}", flush=True)

"""xd_profile_causes.py (2026-10-24) -- profile cell-month death panels by cause so placebos can be compared with
homicide on structure BEFORE any placebo model result. Read-only; writes outputs/crossdomain/xd_profile_causes.csv.
Panels: homicide, traffic, suicide, drowning, dengue (xd_brazil[_{cause}]_panel.parquet). Periods: train (2001-2020)
and test (2022-01..2024-11). Stats per cause x period:
  deaths_per_year      total deaths / years
  mean, share_any      mean deaths_next per cell-month; share of cell-months with >= 1 death
  share_empty_hist     share of rows with no deaths in t-12..t-1 (seq_len == 0)
  ac1_pooled           corr(log1p deaths_t, log1p deaths_next), pooled
  ac1_within           same after removing each cell's mean (temporal persistence net of level)
  fano_cell            median over cells (train mean >= 0.5) of var/mean of monthly deaths (burstiness; 1 = Poisson)
  season_r2            R2 of national monthly totals on calendar month (seasonality)
  spatial_r            corr across cells of log1p(train mean) with the mean of its queen neighbours (clustering)
  top10_share          share of deaths in the top 10% of cells (concentration)"""
import os, numpy as np, pandas as pd
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
P = {"homicide": "xd_brazil_panel", "traffic": "xd_brazil_traffic_panel", "suicide": "xd_brazil_suicide_panel",
     "drowning": "xd_brazil_drowning_panel", "dengue": "xd_brazil_dengue_panel"}
rows = []
for cause, stem in P.items():
    f = f"{R}/data_intermediate/{stem}.parquet"
    if not os.path.exists(f): print(f"[skip] {cause}: no panel"); continue
    d = pd.read_parquet(f, columns=["grid_id", "t", "year_month", "deaths_t", "deaths_next", "seq_len"])
    d = d.sort_values(["grid_id", "t"]).reset_index(drop=True)
    tr = (d.year_month >= "2001-01") & (d.year_month <= "2020-12")
    cm = d[tr].groupby("grid_id")["deaths_t"].mean()
    nb = pd.Series(0.0, index=cm.index); nn = pd.Series(0.0, index=cm.index)
    for off in (-721, -720, -719, -1, 1, 719, 720, 721):
        v = cm.reindex(cm.index + off).to_numpy(); nb += np.nan_to_num(np.log1p(v)); nn += np.isfinite(v)
    ok = nn > 0
    spatial_r = float(np.corrcoef(np.log1p(cm[ok]), (nb[ok] / nn[ok]))[0, 1])
    for per, m in [("train", tr), ("test", (d.year_month >= "2022-01") & (d.year_month <= "2024-11"))]:
        x = d[m]
        yrs = x.year_month.str[:4].nunique()
        a, b = np.log1p(x.deaths_t.astype(float)), np.log1p(x.deaths_next.astype(float))
        aw = a - a.groupby(x.grid_id).transform("mean"); bw = b - b.groupby(x.grid_id).transform("mean")
        g = x.groupby("grid_id")["deaths_t"]; mu = g.mean(); va = g.var()
        busy = mu >= 0.5
        nat = x.groupby("year_month")["deaths_t"].sum()
        mo = nat.index.str[5:7]
        season_r2 = float(1 - ((nat - nat.groupby(mo).transform("mean")) ** 2).sum() / ((nat - nat.mean()) ** 2).sum())
        tot = x.groupby("grid_id")["deaths_t"].sum().sort_values(ascending=False)
        rows.append(dict(cause=cause, period=per, deaths_per_year=float(x.deaths_t.sum() / yrs), mean=float(x.deaths_next.mean()),
                         share_any=float((x.deaths_next > 0).mean()), share_empty_hist=float((x.seq_len == 0).mean()),
                         ac1_pooled=float(np.corrcoef(a, b)[0, 1]), ac1_within=float(np.corrcoef(aw, bw)[0, 1]),
                         fano_cell=float((va[busy] / mu[busy]).median()) if busy.any() else np.nan, n_busy_cells=int(busy.sum()),
                         season_r2=season_r2, spatial_r=spatial_r,
                         top10_share=float(tot.iloc[: max(1, len(tot) // 10)].sum() / max(tot.sum(), 1))))
    print(rows[-2]); print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_csv(f"{R}/outputs/crossdomain/xd_profile_causes.csv", index=False)
pd.set_option("display.width", 250); print(out.round(3).to_string(index=False)); print("PROFILE_DONE")

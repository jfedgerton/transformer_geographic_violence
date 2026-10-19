"""ens_alerts.py -- forecasting ensemble: what the extra correct alerts are, in events and deaths (CPU).
(Forecasting with a pretrained encoding; not the LORO transfer design.)

Targets: civilian, onset, fatal25 by default; incidence via ENS_ALERT_TARGETS=incidence (separate output file).
Rule (same as the paper's 836): each model flags k test cell-months, k = number of events in the test window
(all regions pooled); flags = top-k by forecast, stable sort. Compared models: ens_mean vs base (XGBoost).
  gained = event cell-months flagged by ens_mean but not by base; lost = flagged by base but not by ens_mean.
  net = gained - lost (equals the difference in hits at k).
For gained and lost cell-months the outcome month (t+1) is described by:
  vac_events  ACLED violence-against-civilians events (panel etype_violence_against_civilians at t+1)
  civ_deaths  ACLED deaths_civilians summed over all events in the cell-month (t+1)
  fatalities  panel count_fatalities at t+1
Breakdowns: ALL, region, outcome year, top 10 countries by |net|.
Intervals: paired cluster bootstrap of the gained/lost sums, stratified by region, 5x5-degree blocks, B=1000,
seed 123. The flag sets are held fixed (intervals condition on the fitted models and the k rule).
Checks: labels in the forecast files are recomputed from the panel and must match exactly; deaths_civilians is
summarised and must not exceed fatalities in any event.
Outputs (never overwritten): outputs/ens/alerts_mt{RTAG}.csv and alerts_mt{RTAG}_ACLED_check.txt; CSV written last.
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
OD = f"{R}/outputs/ens"
RTAG = os.environ.get("ENS_RTAG", "_pilot_s123-456-789")
B = int(os.environ.get("ENS_BOOT_B", "1000"))
SEED = 123
ALL_TARGETS = ["civilian", "onset", "fatal25", "incidence"]   # fixed order: bootstrap seed per target
TARGETS = os.environ.get("ENS_ALERT_TARGETS", "civilian,onset,fatal25").split(",")
assert set(TARGETS) <= set(ALL_TARGETS) and len(TARGETS) == len(set(TARGETS)), TARGETS
OSUF = "" if TARGETS == ["civilian", "onset", "fatal25"] else "_" + "-".join(TARGETS)
OUT_CSV, OUT_CHK = f"{OD}/alerts_mt{RTAG}{OSUF}.csv", f"{OD}/alerts_mt{RTAG}{OSUF}_ACLED_check.txt"
assert not os.path.exists(OUT_CSV), f"refusing to overwrite {OUT_CSV}"


def next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}"


# ---- panel: values at the outcome month ----
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "has_violence", "violence_next_month", "count_fatalities",
                                 "etype_violence_against_civilians", "country"])
panel["year_month"] = panel["year_month"].astype(str)
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months in panel"
cur = panel.set_index(["grid_id", "year_month"])

# ---- ACLED civilian deaths by cell-month ----
ac = pd.read_parquet(f"{R}/data_intermediate/acled_with_grid.parquet",
                     columns=["grid_id", "year_month", "event_type", "fatalities", "deaths_civilians"])
ac["year_month"] = ac["year_month"].astype(str)
ac = ac[ac["year_month"] >= "2022-01"].copy()
ac["deaths_civilians"] = pd.to_numeric(ac["deaths_civilians"], errors="coerce")
ac["fatalities"] = pd.to_numeric(ac["fatalities"], errors="coerce")
chk = []
chk.append(f"ACLED rows 2022+: {len(ac):,}")
chk.append(f"deaths_civilians: missing {int(ac['deaths_civilians'].isna().sum()):,}; describe:\n{ac['deaths_civilians'].describe().to_string()}")
chk.append(f"deaths_civilians value counts (top 10):\n{ac['deaths_civilians'].value_counts().head(10).to_string()}")
vac = ac["event_type"] == "Violence against civilians"
chk.append(f"sum deaths_civilians: all events {ac['deaths_civilians'].sum():,.0f}; VAC events {ac.loc[vac, 'deaths_civilians'].sum():,.0f}; "
           f"sum fatalities in VAC events {ac.loc[vac, 'fatalities'].sum():,.0f}")
over = (ac["deaths_civilians"] > ac["fatalities"]).sum()
chk.append(f"events with deaths_civilians > fatalities: {int(over):,}")
open(OUT_CHK + ".part", "w").write("\n\n".join(chk) + "\n")
os.replace(OUT_CHK + ".part", OUT_CHK)
print("\n".join(chk), flush=True)
assert over == 0, "deaths_civilians exceeds fatalities in some events; inspect before using it"
civ = ac.groupby(["grid_id", "year_month"])["deaths_civilians"].sum()

rows = []
for t in TARGETS:
    f = f"{OD}/preds_ens_mt_{t}{RTAG}.npz"
    assert os.path.exists(f), f"missing {f}"
    z = np.load(f)
    te = z["window"].astype(str) == "test"
    g = z["grid_id"][te].astype(np.int64)
    ym = z["year_month"][te].astype(str)
    y = z["y"][te].astype(np.int64)
    region = z["region"][te].astype(str)
    p_e, p_b = z["p_ens_mean"][te].astype(np.float64), z["p_base"][te].astype(np.float64)
    ymn = np.array([next_month(m) for m in ym])
    # outcome-month values
    keys_n = pd.MultiIndex.from_arrays([g, ymn])
    nxt = cur.reindex(keys_n)
    assert nxt["count_fatalities"].notna().all(), f"{t}: outcome month missing from panel"
    vac_n = nxt["etype_violence_against_civilians"].to_numpy(float)
    fat_n = nxt["count_fatalities"].to_numpy(float)
    civd_n = civ.reindex(keys_n).fillna(0).to_numpy(float)
    country = cur.reindex(pd.MultiIndex.from_arrays([g, ym]))["country"].astype(str).to_numpy()
    # labels must match the panel exactly
    now = cur.reindex(pd.MultiIndex.from_arrays([g, ym]))
    if t == "civilian":
        lab = (vac_n > 0).astype(int)
    elif t == "incidence":
        lab = now["violence_next_month"].to_numpy().astype(int)
    elif t == "fatal25":
        lab = (fat_n >= 25).astype(int)
    else:
        assert (now["has_violence"].to_numpy() == 0).all(), "onset rows must be quiet in month t"
        lab = now["violence_next_month"].to_numpy().astype(int)
    assert np.array_equal(lab, y), f"{t}: {int((lab != y).sum())} labels differ from the panel"
    # equal-alert flags
    k = int(y.sum())
    fe, fb = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fe[np.argsort(-p_e, kind="stable")[:k]] = True
    fb[np.argsort(-p_b, kind="stable")[:k]] = True
    gained, lost = fe & ~fb & (y == 1), fb & ~fe & (y == 1)
    assert int((fe & (y == 1)).sum()) - int((fb & (y == 1)).sum()) == int(gained.sum()) - int(lost.sum())
    vals = {"cellmonths": np.ones(len(y)), "vac_events": vac_n, "civ_deaths": civd_n, "fatalities": fat_n}
    # bootstrap draws (block5 within region)
    rowi, coli = (g - 1) // 720, (g - 1) % 720
    cl = (rowi // 10) * 72 + (coli // 10)
    codes, uniq = pd.factorize(pd.Series(list(zip(region, cl))))
    reg_of = np.array([u[0] for u in uniq])
    regs = sorted(set(region))
    rng = np.random.default_rng([SEED, ALL_TARGETS.index(t)])
    W = np.empty((B, len(uniq)))
    for b in range(B):
        cnt = np.zeros(len(uniq))
        for rg in regs:
            c = np.flatnonzero(reg_of == rg)
            cnt += np.bincount(rng.choice(c, size=len(c), replace=True), minlength=len(uniq))
        W[b] = cnt
    years = np.array([m[:4] for m in ymn])
    groups = [("ALL", "ALL", np.ones(len(y), bool))]
    groups += [("region", rg, region == rg) for rg in regs]
    groups += [("year", yr, years == yr) for yr in sorted(set(years))]
    net_c = pd.Series(np.where(gained, 1, 0) - np.where(lost, 1, 0)).groupby(country).sum()
    top = net_c.abs().sort_values(ascending=False).head(10).index
    groups += [("country", c, country == c) for c in top]
    n_months = len(set(ymn))
    for gtype, gname, s in groups:
        rec = dict(target=t, group_type=gtype, group=gname, k_alerts=k, events=int(y[s].sum()),
                   hits_ens=int((fe & (y == 1) & s).sum()), hits_base=int((fb & (y == 1) & s).sum()),
                   n_outcome_months=n_months)
        for vn, v in vals.items():
            a, l = v * (gained & s), v * (lost & s)
            rec[f"{vn}_gained"], rec[f"{vn}_lost"], rec[f"{vn}_net"] = a.sum(), l.sum(), a.sum() - l.sum()
            ca = np.bincount(codes, weights=a, minlength=len(uniq))
            cl_ = np.bincount(codes, weights=l, minlength=len(uniq))
            d = W @ (ca - cl_)
            rec[f"{vn}_net_lo"], rec[f"{vn}_net_hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
        rows.append(rec)
    print(f"[{t}] k={k} gained={int(gained.sum())} lost={int(lost.sum())} net={int(gained.sum() - lost.sum())}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV + ".part", index=False)
os.replace(OUT_CSV + ".part", OUT_CSV)
show = out[out.group_type == "ALL"][["target", "cellmonths_net", "cellmonths_net_lo", "cellmonths_net_hi", "vac_events_net",
                                     "civ_deaths_net", "civ_deaths_net_lo", "civ_deaths_net_hi", "fatalities_net",
                                     "fatalities_net_lo", "fatalities_net_hi"]]
print(show.round(1).to_string(index=False))
print(f"[done] {OUT_CSV}")

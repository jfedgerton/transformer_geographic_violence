"""ens_alerts_totals.py -- gross deaths covered by each model's correct alerts (CPU).
(Forecasting with a pretrained encoding; not the LORO transfer design.)

Same alert rule as ens_alerts.py: for each target, ens_mean and base (XGBoost) each flag k = number of test events
(stable top-k). A hit = flagged event cell-month. For each target and model:
  fatalities_hits = panel count_fatalities at t+1 summed over hits
  civ_deaths_hits = ACLED deaths_civilians at t+1 summed over hits
Denominators: the same sums over all test event cell-months (y == 1) and over all test cell-months.
Union row (incidence, civilian, onset): a cell-month counts once if it is a hit for any of those targets.
Rows: ALL and ALL excluding Ukraine.
Check: hits_ens - hits_base equals the net in ens_alerts.py (gained - lost), asserted per target.
Output (never overwritten): outputs/ens/alerts_totals_mt{RTAG}.csv
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
OD = f"{R}/outputs/ens"
RTAG = os.environ.get("ENS_RTAG", "_pilot_s123-456-789")
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
UNION = ["incidence", "civilian", "onset"]
OUT_CSV = f"{OD}/alerts_totals_mt{RTAG}.csv"
assert not os.path.exists(OUT_CSV), f"refusing to overwrite {OUT_CSV}"


def next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}"


panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "count_fatalities", "country"])
panel["year_month"] = panel["year_month"].astype(str)
assert not panel.duplicated(["grid_id", "year_month"]).any()
cur = panel.set_index(["grid_id", "year_month"])
ac = pd.read_parquet(f"{R}/data_intermediate/acled_with_grid.parquet", columns=["grid_id", "year_month", "fatalities", "deaths_civilians"])
ac["year_month"] = ac["year_month"].astype(str)
ac = ac[ac["year_month"] >= "2022-01"]
ac = ac.assign(deaths_civilians=pd.to_numeric(ac["deaths_civilians"], errors="coerce"), fatalities=pd.to_numeric(ac["fatalities"], errors="coerce"))
assert ac["deaths_civilians"].notna().all() and not (ac["deaths_civilians"] > ac["fatalities"]).any()
civ = ac.groupby(["grid_id", "year_month"])["deaths_civilians"].sum()

rows = []
uni = {"ens": {}, "base": {}, "events": {}}   # (grid_id, ym) -> (fatalities, civ_deaths, country)
for t in TARGETS:
    z = np.load(f"{OD}/preds_ens_mt_{t}{RTAG}.npz")
    te = z["window"].astype(str) == "test"
    g = z["grid_id"][te].astype(np.int64); ym = z["year_month"][te].astype(str); y = z["y"][te].astype(np.int64)
    p_e, p_b = z["p_ens_mean"][te].astype(np.float64), z["p_base"][te].astype(np.float64)
    kn = pd.MultiIndex.from_arrays([g, np.array([next_month(m) for m in ym])])
    nx = cur.reindex(kn)
    assert nx["count_fatalities"].notna().all(), f"{t}: outcome month missing from panel"
    fat = nx["count_fatalities"].to_numpy(float)
    cd = civ.reindex(kn).fillna(0).to_numpy(float)
    country = cur.reindex(pd.MultiIndex.from_arrays([g, ym]))["country"].astype(str).to_numpy()
    k = int(y.sum())
    fe, fb = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fe[np.argsort(-p_e, kind="stable")[:k]] = True
    fb[np.argsort(-p_b, kind="stable")[:k]] = True
    he, hb, ev = fe & (y == 1), fb & (y == 1), y == 1
    ga, lo = fe & ~fb & (y == 1), fb & ~fe & (y == 1)
    assert np.isclose((fat * he).sum() - (fat * hb).sum(), (fat * ga).sum() - (fat * lo).sum())
    for scope, s in (("ALL", np.ones(len(y), bool)), ("ALL_ex_Ukraine", country != "Ukraine")):
        rows.append(dict(target=t, scope=scope, k_alerts=k, events=int((ev & s).sum()),
                         hits_ens=int((he & s).sum()), hits_base=int((hb & s).sum()),
                         fatalities_hits_ens=(fat * (he & s)).sum(), fatalities_hits_base=(fat * (hb & s)).sum(),
                         fatalities_events=(fat * (ev & s)).sum(), fatalities_all_test=(fat * s).sum(),
                         civ_deaths_hits_ens=(cd * (he & s)).sum(), civ_deaths_hits_base=(cd * (hb & s)).sum(),
                         civ_deaths_events=(cd * (ev & s)).sum(), civ_deaths_all_test=(cd * s).sum()))
    if t in UNION:
        for name, m in (("ens", he), ("base", hb), ("events", ev)):
            for i in np.flatnonzero(m):
                key = (int(g[i]), ym[i])
                v = (fat[i], cd[i], country[i])
                assert uni[name].setdefault(key, v) == v, f"outcome values differ across targets for {key}"
    print(f"[{t}] k={k} hits ens {int(he.sum())} base {int(hb.sum())}", flush=True)

for scope in ("ALL", "ALL_ex_Ukraine"):
    keep = (lambda c: True) if scope == "ALL" else (lambda c: c != "Ukraine")
    sm = {n: [v for v in d.values() if keep(v[2])] for n, d in uni.items()}
    rows.append(dict(target="union_" + "-".join(UNION), scope=scope, k_alerts=np.nan, events=len(sm["events"]),
                     hits_ens=len(sm["ens"]), hits_base=len(sm["base"]),
                     fatalities_hits_ens=sum(v[0] for v in sm["ens"]), fatalities_hits_base=sum(v[0] for v in sm["base"]),
                     fatalities_events=sum(v[0] for v in sm["events"]), fatalities_all_test=np.nan,
                     civ_deaths_hits_ens=sum(v[1] for v in sm["ens"]), civ_deaths_hits_base=sum(v[1] for v in sm["base"]),
                     civ_deaths_events=sum(v[1] for v in sm["events"]), civ_deaths_all_test=np.nan))
out = pd.DataFrame(rows)
for v in ("fatalities", "civ_deaths"):
    out[f"{v}_net"] = out[f"{v}_hits_ens"] - out[f"{v}_hits_base"]
    out[f"{v}_share_ens"] = out[f"{v}_hits_ens"] / out[f"{v}_events"]
    out[f"{v}_share_base"] = out[f"{v}_hits_base"] / out[f"{v}_events"]
    assert (out[f"{v}_hits_ens"] <= out[f"{v}_events"] + 1e-9).all() and (out[f"{v}_hits_base"] <= out[f"{v}_events"] + 1e-9).all()
out.to_csv(OUT_CSV + ".part", index=False)
os.replace(OUT_CSV + ".part", OUT_CSV)
pd.set_option("display.width", 250)
print(out[["target", "scope", "hits_ens", "hits_base", "fatalities_hits_ens", "fatalities_hits_base", "fatalities_events",
           "fatalities_net", "fatalities_share_ens", "fatalities_share_base", "civ_deaths_hits_ens", "civ_deaths_hits_base",
           "civ_deaths_events"]].round(3).to_string(index=False))
print(f"[done] {OUT_CSV}")

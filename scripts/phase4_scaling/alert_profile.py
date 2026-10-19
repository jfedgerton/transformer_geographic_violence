"""alert_profile.py -- what the embedding's extra correct alerts are (the "836"), in deaths and event types. CPU only.

Same alerts as cell_alerts_agg.py (reviewed): for each held-out region x encoder seed, the saved temporal (Mode B)
forecasts preds_region_L_contrastive_e_loro_{region}_s{seed}.npz, alert budget k = number of violent cell-months in
that region's test window, top-k cell-months flagged separately for base and base + embedding (stable sort; ties
broken by row order). PR-AUC is re-scored and checked against eval_region_*.csv first.
Groups (rows with an observed violent outcome, y = 1)
  emb_only   caught by base + embedding, missed by base
  base_only  caught by base, missed by base + embedding
  net        emb_only minus base_only (counts and sums); net count = the paper's extra-hit figure
  hit_emb    every correct alert of base + embedding (comparison)
  all_viol   every violent cell-month in the test window (comparison)
Outcome month. A row is a cell at origin month t with label "violence in month t+1"; the profile describes the events in
month t+1 of that cell. Events come from acled_with_grid.parquet, the file the panel was built from (ACLED + UCDP, not
de-duplicated across sources), using its own grid_id (Codex 11, P01: canonical mapping; the script also recomputes the
id from latitude/longitude with the pipeline's floor rule and stops on any disagreement or out-of-range coordinate).
Only the event types that define the target count (Battles, Explosions/Remote violence, Violence against civilians).
Check: every y = 1 row must have at least one such event and every y = 0 row none, with ZERO tolerance; mismatches are
written to alert_profile_label_mismatch.csv before the script stops. This validates presence, not counts: events moved
between two violent cells would not be detected, which the canonical grid_id rules out.
Deaths are reported BY SOURCE and never summed across sources, because the same incident can appear in both:
  acled_fat      ACLED fatalities (all target event types)
  acled_vac_fat  ACLED fatalities in Violence against civilians events
  ucdp_fat       UCDP best estimate (UCDP ends Dec 2024)
  ucdp_civ       UCDP civilian deaths (deaths_civilians)
  *_c            the same sums restricted to outcome months through Dec 2024, when both sources cover the period
Event-type counts (n_*) are record counts across both sources, not de-duplicated incidents.
The 836 is a signed, seed-averaged difference (emb_only minus base_only), not a set of 836 rows; both gross groups are
reported. Its unrounded value is reconciled with outputs/cell_alerts_incidence.csv.gz (sum of d_tp).
Counts and sums are computed per region x seed, summed over regions, then averaged over the three seeds.
Outputs (published together; outputs/alert_profile_incidence.DONE marks a complete bundle and blocks re-runs):
  alert_profile_incidence.csv           group x statistic
  alert_profile_countries.csv           country x group, cell-month counts by outcome-month attribution (the cell-month's
                                        modal event country; ties -> alphabetical first) -- for alert counts
  alert_profile_countries_events.csv    country x group, ACLED fatalities and record counts by EACH EVENT's own country
                                        (Codex 11, P02) -- for fatality totals; unknown country -> "unknown"
A bundle without the DONE marker is incomplete and may be overwritten by a re-run (Codex 11, P03).
Env: AP_ROOT for the synthetic test.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("AP_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
SEEDS = [123, 456, 789]
TOL = 1e-6
TYPES = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
OUT, OUT_C, OUT_E = f"{O}/alert_profile_incidence.csv", f"{O}/alert_profile_countries.csv", f"{O}/alert_profile_countries_events.csv"
DONE = f"{O}/alert_profile_incidence.DONE"
assert not os.path.exists(DONE), "refusing to overwrite a completed alert profile"

# ---- events -> cell x outcome month ----
ev = pd.read_parquet(f"{R}/data_intermediate/acled_with_grid.parquet",
                     columns=["grid_id", "latitude", "longitude", "year_month", "event_type", "fatalities", "deaths_civilians",
                              "source_of_record", "country"])
ev["ym"] = ev["year_month"].astype(str).str[:7]
ev = ev[(ev["ym"] >= "2022-02") & ev["event_type"].isin(TYPES)].copy()
assert len(ev) > 0
src = ev["source_of_record"].astype(str).str.upper()
assert src.isin(["ACLED", "UCDP"]).all(), f"unexpected sources {sorted(src.unique())[:5]}"
lat = ev["latitude"].to_numpy(float); lon = ev["longitude"].to_numpy(float)
assert np.isfinite(lat).all() and np.isfinite(lon).all()
assert ((lat >= -90) & (lat < 90) & (lon >= -180) & (lon < 180)).all(), "coordinates outside the PRIO-GRID range"
row = np.floor((lat + 90.0) * 2).astype(np.int64); col = np.floor((lon + 180.0) * 2).astype(np.int64)   # 02_build_grid.py rule
ev["gid"] = ev["grid_id"].astype(np.int64)
assert (ev["gid"].to_numpy() == row * 720 + col + 1).all(), "grid_id disagrees with the pipeline's floor rule"
ev["country"] = ev["country"].fillna("unknown").astype(str)
fat = ev["fatalities"].fillna(0).to_numpy(float); assert (fat >= 0).all()
isA = (src == "ACLED").to_numpy(); isU = ~isA
ev["n_ev"] = 1
ev["n_acled"] = isA.astype(int); ev["n_ucdp"] = isU.astype(int)
for t, nm in zip(TYPES, ["n_battle", "n_explosion", "n_vac"]):
    ev[nm] = (ev["event_type"] == t).astype(int)
ev["acled_fat"] = np.where(isA, fat, 0.0)
ev["acled_vac_fat"] = np.where(isA & (ev["event_type"] == "Violence against civilians").to_numpy(), fat, 0.0)
ev["ucdp_fat"] = np.where(isU, fat, 0.0)
ev["ucdp_civ"] = np.where(isU, ev["deaths_civilians"].fillna(0).to_numpy(float), 0.0)
common = (ev["ym"] <= "2024-12").to_numpy()                       # both sources cover these outcome months
for c_ in ("acled_fat", "ucdp_fat", "ucdp_civ"): ev[c_ + "_c"] = np.where(common, ev[c_], 0.0)
SUMS = ["n_ev", "n_acled", "n_ucdp", "n_battle", "n_explosion", "n_vac", "acled_fat", "acled_vac_fat", "ucdp_fat", "ucdp_civ",
        "acled_fat_c", "ucdp_fat_c", "ucdp_civ_c"]
cmv = ev.groupby(["gid", "ym"])[SUMS].sum()
_cc = ev.groupby(["gid", "ym", "country"]).size().rename("n").reset_index().sort_values(["gid", "ym", "n", "country"],
                                                                                         ascending=[True, True, False, True])
ctry = _cc.drop_duplicates(["gid", "ym"]).set_index(["gid", "ym"])["country"]          # modal country, ties -> alphabetical
n_ctry = _cc.groupby(["gid", "ym"]).size()                                            # > 1 means a mixed-country cell-month
evc = ev.groupby(["gid", "ym", "country"])[["n_ev", "acled_fat"]].sum()              # event-country totals (P02)
print(f"[events] {len(ev):,} target-type events 2022-02+ in {len(cmv):,} cell-months", flush=True)


def alerts(score, k):
    a = np.zeros(len(score), bool)
    if k > 0: a[np.argsort(-score, kind="stable")[:k]] = True
    return a


GROUPS = ["emb_only", "base_only", "hit_emb", "all_viol"]
per_seed = {sd: {g: [] for g in GROUPS} for sd in SEEDS}
checks = []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        fp, fe = f"{O}/preds_region_{tag}.npz", f"{O}/eval_region_{tag}.csv"
        assert os.path.exists(fp) and os.path.exists(fe), f"missing {tag}"
        z = np.load(fp, allow_pickle=True)
        g, ym, y = z["grid_id"].astype(np.int64), z["year_month"].astype(str), z["y"].astype(int)
        pb, pe = z["base"].astype(float), z["emb_base"].astype(float)
        assert len(set(map(len, (g, ym, y, pb, pe)))) == 1 and np.isfinite(pb).all() and np.isfinite(pe).all()
        assert not pd.DataFrame({"g": g, "m": ym}).duplicated().any(), f"{tag}: duplicate cell-months"
        e = pd.read_csv(fe); e = e[(e["target"] == "incidence") & (e["mode"] == "B_temporal")].set_index("model")["score"]
        for m, p in (("base", pb), ("emb_base", pe)):
            assert abs(average_precision_score(y, p) - e[m]) <= TOL, f"{tag}: {m} PR-AUC differs from eval CSV"
        tm = (pd.PeriodIndex(pd.Series(ym).str[:7], freq="M") + 1).strftime("%Y-%m").to_numpy()   # outcome month t+1
        key = pd.MultiIndex.from_arrays([g, tm])
        X = cmv.reindex(key).fillna(0.0).reset_index(drop=True)
        X["country"] = ctry.reindex(key).to_numpy()
        X["mixed_country"] = (n_ctry.reindex(key).fillna(0).to_numpy() > 1).astype(int)
        X["gid"] = g; X["ym"] = tm
        has = X["n_ev"].to_numpy() > 0
        # label check: the join must reproduce the target (UCDP ends 2024-12, ACLED 2025-05; test outcomes end 2025-05)
        bad = has != (y == 1)
        checks.append(dict(region=rg, seed=sd, rows=len(y), y1_no_event=int(((y == 1) & ~has).sum()),
                           y0_with_event=int(((y == 0) & has).sum()), mismatch=float(bad.mean())))
        if bad.any():                                                  # Codex 11, P01: zero tolerance, diagnostics first
            pd.DataFrame({"grid_id": g[bad], "origin_month": ym[bad], "outcome_month": tm[bad], "y": y[bad],
                          "target_events": X["n_ev"].to_numpy()[bad]}).to_csv(f"{O}/alert_profile_label_mismatch.csv", index=False)
        assert not bad.any(), f"{tag}: event join disagrees with labels on {int(bad.sum())} rows (see alert_profile_label_mismatch.csv)"
        k = int(y.sum())
        ab, ae = alerts(pb, k), alerts(pe, k)
        v = y == 1
        masks = {"emb_only": ae & ~ab & v, "base_only": ab & ~ae & v, "hit_emb": ae & v, "all_viol": v}
        assert masks["emb_only"].sum() - masks["base_only"].sum() == (ae & v).sum() - (ab & v).sum()
        for gname, m in masks.items():
            sub = X[m].copy(); sub["region"] = rg
            per_seed[sd][gname].append(sub)

C = pd.DataFrame(checks)
print(C.to_string(index=False), flush=True)


def profile(df):
    n = len(df)
    out = {"cell_months": n}
    for s in SUMS: out[s] = float(df[s].sum())
    has_a = df["n_acled"] > 0
    out["acled_fat_median"] = float(df.loc[has_a, "acled_fat"].median()) if has_a.any() else np.nan
    out["share_any_acled_fat"] = float((df["acled_fat"] > 0).mean()) if n else np.nan
    out["share_10plus_acled_fat"] = float((df["acled_fat"] >= 10).mean()) if n else np.nan
    out["share_any_vac"] = float((df["n_vac"] > 0).mean()) if n else np.nan
    out["mixed_country_cell_months"] = float(df["mixed_country"].sum())
    return out


rows = []
for gname in GROUPS:
    P = pd.DataFrame([profile(pd.concat(per_seed[sd][gname], ignore_index=True)) for sd in SEEDS], index=SEEDS)
    for stat in P.columns:
        rows.append(dict(group=gname, stat=stat, mean=P[stat].mean(), s123=P.loc[123, stat], s456=P.loc[456, stat],
                         s789=P.loc[789, stat]))
D = pd.DataFrame(rows)
W = D.pivot(index="stat", columns="group", values="mean")
net = (W["emb_only"] - W["base_only"]).where(~W.index.str.startswith(("share", "acled_fat_median")))
for stat, val in net.items():
    rows.append(dict(group="net", stat=stat, mean=val, s123=np.nan, s456=np.nan, s789=np.nan))
D = pd.DataFrame(rows)

cnt = []
for sd in SEEDS:
    for gname in ("emb_only", "base_only"):
        df = pd.concat(per_seed[sd][gname], ignore_index=True)
        c = df.groupby("country").agg(cell_months=("n_ev", "size"), acled_fat=("acled_fat", "sum"))
        c["group"], c["seed"] = gname, sd
        cnt.append(c.reset_index())
K = pd.concat(cnt).groupby(["country", "group"])[["cell_months", "acled_fat"]].sum().div(len(SEEDS)).unstack("group").fillna(0)
K.columns = [f"{a}_{b}" for a, b in K.columns]
K["net_cell_months"] = K["cell_months_emb_only"] - K["cell_months_base_only"]
K["net_acled_fat"] = K["acled_fat_emb_only"] - K["acled_fat_base_only"]
K = K.sort_values("net_cell_months", ascending=False)

evrows = []
for sd in SEEDS:
    for gname in ("emb_only", "base_only"):
        df = pd.concat(per_seed[sd][gname], ignore_index=True)
        k_ = pd.MultiIndex.from_arrays([df["gid"].to_numpy(np.int64), df["ym"].to_numpy()])
        e_ = evc.reset_index().merge(pd.DataFrame({"gid": k_.get_level_values(0), "ym": k_.get_level_values(1)}).drop_duplicates(),
                                     on=["gid", "ym"])
        # a cell-month can enter a group in only one region x seed; repeated rows would double count
        assert not df.duplicated(["gid", "ym"]).any(), "duplicate cell-months within a group"
        c_ = e_.groupby("country")[["n_ev", "acled_fat"]].sum(); c_["group"], c_["seed"] = gname, sd
        evrows.append(c_.reset_index())
KE = pd.concat(evrows).groupby(["country", "group"])[["n_ev", "acled_fat"]].sum().div(len(SEEDS)).unstack("group").fillna(0)
KE.columns = [f"{a}_{b}" for a, b in KE.columns]
KE["net_acled_fat"] = KE["acled_fat_emb_only"] - KE["acled_fat_base_only"]
KE = KE.sort_values("net_acled_fat", ascending=False)
assert abs(KE["acled_fat_emb_only"].sum() - W.loc["acled_fat", "emb_only"]) < 1e-6 * max(1, W.loc["acled_fat", "emb_only"]), \
    "event-country fatality totals do not reconcile with the global total"
assert abs(K["net_cell_months"].sum() - net["cell_months"]) < 1e-6, "country cell-month totals do not reconcile with the net"
# reconcile the net count with the paper's alert file (seed-averaged d_tp summed over cells)
ca = f"{O}/cell_alerts_incidence.csv.gz"
if os.path.exists(ca):
    dtp = float(pd.read_csv(ca, usecols=["d_tp"])["d_tp"].sum())
    print(f"[reconcile] net cell-months {net['cell_months']:.4f} vs cell_alerts d_tp {dtp:.4f}", flush=True)
    assert abs(dtp - net["cell_months"]) < 0.01, "net count disagrees with cell_alerts_incidence.csv.gz"
# ---- publish: all three files staged, then committed, then the DONE marker (Codex 11, P03) ----
D.to_csv(OUT + ".part", index=False); K.to_csv(OUT_C + ".part"); KE.to_csv(OUT_E + ".part")
for f_ in (OUT, OUT_C, OUT_E): os.replace(f_ + ".part", f_)
open(DONE, "w").write("complete\n")
pd.set_option("display.width", 200)
print("\n" + W.join(net.rename("net")).round(3).to_string())
print("\n" + K.head(12).round(1).to_string() + "\n...\n" + K.tail(6).round(1).to_string())
print(f"[done] {OUT}", flush=True)

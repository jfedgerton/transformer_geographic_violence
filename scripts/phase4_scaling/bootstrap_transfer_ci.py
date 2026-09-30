"""bootstrap_transfer_ci.py -- country-block bootstrap CIs on the transfer deltas.

Consumes the per-(cap,seed) transfer-eval CSVs written by transfer_eval.py:
    outputs/transfer_eval_{CAP}_s{SEED}.csv
columns: cap, seed, unit, unit_type, n_train, n_test, test_pos, africa_saw_unit,
         mech, model, frac, roc, pr

Procedure (the referee-proofing):
  - Block = held-out unit (8 countries + 2 regions). Within-unit correlation across
    seeds/fracs is handled by resampling whole units, not rows.
  - Average the available seeds first -> one value per unit. Then bootstrap over units
    (resample units with replacement, recompute the mean, B times), percentile CI.
  - Strata: country-only (n<=8, the defensible CI), region-only (n<=2, flagged
    uninformative), pooled (n<=10).

Quantities:
  Mech B (headline): delta = PR(ft_pretrained) - PR(ft_scratch), per (cap, frac, unit).
  Mech A: PR(frozen_holdout) - PR(persistence), and PR(frozen_holdout) - PR(frozen_africa),
          per (cap, unit). africa-specialist deltas drop units africa saw (Nigeria, DRC).

Out: outputs/transfer_ci_summary.csv  (+ printed tables)

Run on Roar (no GPU needed):
    module load anaconda && source venv/bin/activate
    python scripts/phase4_scaling/bootstrap_transfer_ci.py
"""
import os
import glob
import numpy as np
import pandas as pd

R = os.environ.get("TGV_ROOT",
                   "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
OUTDIR = f"{R}/outputs"
OUT = f"{OUTDIR}/transfer_ci_summary.csv"
SEED = 123
B = 10000
CAP_ORDER = ["S", "M", "L", "XL"]
rng = np.random.default_rng(SEED)

# ---- load all per-(cap,seed) eval CSVs ----
paths = sorted(glob.glob(f"{OUTDIR}/transfer_eval_*_s*.csv"))
assert paths, f"no transfer_eval_*_s*.csv in {OUTDIR}"
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
# drop skipped/degenerate rows
df = df[df["pr"].notna() & df["model"].isin(
    ["frozen_holdout", "frozen_africa", "persistence", "ft_pretrained", "ft_scratch"])].copy()
caps_present = [c for c in CAP_ORDER if c in set(df["cap"])]
print(f"[load] {len(paths)} files | rows={len(df):,} | caps={caps_present} | "
      f"seeds={sorted(df['seed'].unique())} | units={df['unit'].nunique()}", flush=True)


def boot_ci(unit_vals):
    """unit_vals: 1-D array of per-unit deltas. Returns (mean, lo, hi, n, excl0)."""
    v = np.asarray(unit_vals, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, False
    if n == 1:
        return float(v[0]), np.nan, np.nan, 1, False
    idx = rng.integers(0, n, size=(B, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    excl0 = (lo > 0) or (hi < 0)
    return float(v.mean()), float(lo), float(hi), n, bool(excl0)


def per_unit_delta(sub, model_a, model_b, key_cols):
    """Average seeds, then delta = mean PR(model_a) - mean PR(model_b) per unit.
    key_cols identifies the cell (e.g. ['cap'] or ['cap','frac'])."""
    a = (sub[sub["model"] == model_a]
         .groupby(["unit", "unit_type"], as_index=False)["pr"].mean()
         .rename(columns={"pr": "pr_a"}))
    b = (sub[sub["model"] == model_b]
         .groupby(["unit", "unit_type"], as_index=False)["pr"].mean()
         .rename(columns={"pr": "pr_b"}))
    m = a.merge(b, on=["unit", "unit_type"], how="inner")
    m["delta"] = m["pr_a"] - m["pr_b"]
    return m


def emit(rows, label, cap, frac, m):
    """m has columns unit_type, delta. Emit country/region/pooled CIs."""
    for stratum, mask in [("country", m["unit_type"] == "country"),
                          ("region", m["unit_type"] == "region"),
                          ("pooled", m["unit_type"].notna())]:
        vals = m.loc[mask, "delta"].to_numpy()
        mean, lo, hi, n, excl0 = boot_ci(vals)
        rows.append(dict(quantity=label, cap=cap, frac=frac, stratum=stratum,
                         n_units=n, mean=mean, ci_lo=lo, ci_hi=hi, excludes_0=excl0))


rows = []

# ---- Mechanism B: ft_pretrained - ft_scratch, per cap x frac ----
mb = df[df["mech"] == "B"]
for cap in caps_present:
    for frac in sorted(mb["frac"].dropna().unique()):
        sub = mb[(mb["cap"] == cap) & (mb["frac"] == frac)]
        m = per_unit_delta(sub, "ft_pretrained", "ft_scratch", ["cap", "frac"])
        if len(m):
            emit(rows, "B_pretrained_minus_scratch", cap, float(frac), m)

# ---- Mechanism A: frozen_holdout - persistence, per cap ----
ma = df[df["mech"] == "A"]
for cap in caps_present:
    sub = ma[ma["cap"] == cap]
    m = per_unit_delta(sub, "frozen_holdout", "persistence", ["cap"])
    if len(m):
        emit(rows, "A_holdout_minus_persistence", cap, np.nan, m)

# ---- Mechanism A: frozen_holdout - frozen_africa, per cap (drop units africa saw) ----
afr_saw = set(df.loc[df["africa_saw_unit"] == True, "unit"])
for cap in caps_present:
    sub = ma[(ma["cap"] == cap) & (~ma["unit"].isin(afr_saw))]
    m = per_unit_delta(sub, "frozen_holdout", "frozen_africa", ["cap"])
    if len(m):
        emit(rows, "A_holdout_minus_africa", cap, np.nan, m)

ci = pd.DataFrame(rows)
ci.to_csv(OUT, index=False)
print(f"\n[done] wrote {OUT} ({len(ci)} rows)\n", flush=True)

# ---- printed summary (country + pooled; star = CI excludes 0) ----
pd.set_option("display.width", 160, "display.max_rows", 200)
def show(q, stratum="country"):
    t = ci[(ci["quantity"] == q) & (ci["stratum"] == stratum)].copy()
    if not len(t):
        return
    t["star"] = np.where(t["excludes_0"], "*", "")
    t["95% CI"] = t.apply(lambda r: f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                          if not np.isnan(r.ci_lo) else "n<2", axis=1)
    t["mean"] = t["mean"].map(lambda x: f"{x:+.3f}")
    cols = ["cap", "frac", "n_units", "mean", "95% CI", "star"]
    print(f"### {q}  [{stratum}]")
    print(t[cols].to_string(index=False), "\n", flush=True)

for q in ["B_pretrained_minus_scratch", "A_holdout_minus_persistence", "A_holdout_minus_africa"]:
    show(q, "country")
    show(q, "pooled")

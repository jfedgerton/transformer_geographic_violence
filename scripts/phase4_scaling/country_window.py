"""country_window.py -- Ukraine and Sudan by war-onset window, with paired cell-bootstrap 95% intervals. CPU only.

Uses the saved regional share-grid forecasts (no refitting), as in country_slice.py (reviewed, Codex 9).
Timing. Each test row is a cell at origin month t (row year_month) with the label "violence in month t+1"
(violence_next_month; the panel has no month gaps within cells). Windows are defined on the TARGET month t+1.
  Ukraine  war start 2022-02 (24 Feb 2022). The test origins start 2022-01, so every Ukraine test target is 2022-02 or
           later; "post" equals the full slice.
             post   targets 2022-02 .. end            early  targets 2022-02 .. 2023-01 (first 12 months)
  Sudan    war start 2023-04 (15 Apr 2023).
             pre    targets .. 2023-03                post   targets 2023-04 .. end
             early  targets 2023-04 .. 2024-03 (first 12 months)
Comparisons at each share s, within one country x window (same definition as bootstrap_ft.py):
  finetune  mean over seeds 123/456/789 and draws 0-4 of PR-AUC(ft_pretrained) - PR-AUC(ft_scratch)
  nested    mean over seeds and draws of PR-AUC(emb_base) - PR-AUC(base)            (incidence only)
Persistence (the "persistence|0|-1" forecast saved in every file; month-t violence, as in the main tables) is scored on
the same rows and reported as a point estimate next to each arm. It is not part of any interval.
Bootstrap. Unit = PRIO-GRID cell (all of a cell's months in the window stay together). Each replicate draws the window's
cells with replacement and applies the SAME weights to both models, every seed, draw and share, so differences are
paired. B=1000, seed 123 (rng = [123, country index, window index]). Intervals condition on the fitted networks.
Context. Ukraine's share of Europe's test rows and test positives, and Sudan's share of Africa's (one row each).
Env: WIN_TARGET (incidence | civilian), WIN_B (default 1000); WIN_DRAWS, WIN_SHARES and WIN_ROOT for the synthetic test.
Outputs (never overwritten): outputs/country_window_{target}.csv (one row per country x window x arm x share),
outputs/country_window_context_{target}.csv.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("WIN_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
TARGET = os.environ.get("WIN_TARGET", "incidence"); assert TARGET in ("incidence", "civilian"), TARGET
TSUF = "" if TARGET == "incidence" else "_civ"
B = int(os.environ.get("WIN_B", "1000"))
BOOT_SEED = 123
SEEDS = [123, 456, 789]
DRAWS = [int(x) for x in os.environ.get("WIN_DRAWS", "0,1,2,3,4").split(",")]
SHARES = [float(x) for x in os.environ.get("WIN_SHARES", "0.0025,0.005,0.0075,0.01,0.025,0.05,0.10").split(",")]
ARMS = ["finetune", "nested"] if TARGET == "incidence" else ["finetune"]
PAIR = {"finetune": ("ft_pretrained", "ft_scratch"), "nested": ("emb_base", "base")}
CASES = [("Ukraine", "europe"), ("Sudan", "africa")]
WINDOWS = {  # (first target month, last target month), inclusive; None = open
    "Ukraine": {"post": ("2022-02", None), "early": ("2022-02", "2023-01")},
    "Sudan": {"pre": (None, "2023-03"), "post": ("2023-04", None), "early": ("2023-04", "2024-03")},
}
OUT, OUT_C = f"{O}/country_window_{TARGET}.csv", f"{O}/country_window_context_{TARGET}.csv"
assert not os.path.exists(OUT) and not os.path.exists(OUT_C), "refusing to overwrite outputs"

cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
cm["grid_id"] = cm["grid_id"].astype(np.int64); cm["year_month"] = cm["year_month"].astype(str).str[:7]
assert not cm.duplicated(["grid_id", "year_month"]).any()
ctry = cm.set_index(["grid_id", "year_month"])["country"]


def prep(p, y):
    """Sort once per forecast vector (same as bootstrap_ft.py)."""
    p = np.asarray(p, np.float64)
    order = np.argsort(-p, kind="stable")
    s = p[order]
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    return order.astype(np.int64), y[order].astype(np.float64), ends


def ap_w(pr, w):
    """Weighted average precision with sklearn's step definition (same as bootstrap_ft.py)."""
    order, ys, ends = pr
    ws = w[order]
    tp = np.cumsum(ws * ys)[ends]; fp = np.cumsum(ws * (1.0 - ys))[ends]
    if tp[-1] <= 0:
        return 0.0
    den = tp + fp
    prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def target_month(ym):
    per = pd.PeriodIndex(ym, freq="M") + 1
    return per.strftime("%Y-%m").to_numpy()


def ne_shares(rg):
    """Shares marked not estimable for every seed and draw by the fine-tune availability files (civilian floors)."""
    out = set()
    for sh in SHARES:
        st = []
        for sd in SEEDS:
            fa = f"{O}/share_grid/avail_finetune_{rg}_s{sd}_f{sh:g}{TSUF}.csv"
            st += pd.read_csv(fa).status.tolist() if os.path.exists(fa) else ["ok"] * len(DRAWS)
        if all(x != "ok" for x in st): out.add(sh)
        else: assert all(x == "ok" for x in st), f"{rg} {sh:g}: mixed availability"
    return out


rows, ctx = [], []
for ci, (country, rg) in enumerate(CASES):
    # ---- load every forecast for this region; identical test rows across files ----
    ref = None; F = {}                                  # (arm, model, share, seed, draw) -> forecast; ("pers",) -> persistence
    ne = ne_shares(rg)
    for arm in ARMS:
        for sd in SEEDS:
            tags = ([(f"finetune_{rg}_s{sd}_f{sh:g}{TSUF}", [sh]) for sh in SHARES if sh not in ne] if arm == "finetune"
                    else [(f"nested_{rg}_s{sd}", SHARES)])
            for tag, shs in tags:
                fp, fm = f"{O}/share_grid/preds_{tag}.npz", f"{O}/share_grid/metrics_{tag}.csv"
                assert os.path.exists(fp) and os.path.exists(fm), f"missing run {tag}"
                z = np.load(fp); met = pd.read_csv(fm)
                gid = z["__grid_id"].astype(np.int64); ym = z["__year_month"].astype(str); y = z["__y"].astype(int)
                if ref is None:
                    ref = (gid, ym, y)
                    assert pd.MultiIndex.from_arrays([gid, ym]).is_unique, f"{tag}: duplicate keys"
                assert np.array_equal(gid, ref[0]) and np.array_equal(ym, ref[1]) and np.array_equal(y, ref[2]), \
                    f"{tag}: test rows differ"
                pk = "persistence|0|-1"
                assert pk in z.files, f"{tag}: no persistence forecast"
                pp = z[pk].astype(np.float64)
                if ("pers",) in F: assert np.array_equal(pp, F[("pers",)]), f"{tag}: persistence differs across files"
                else: F[("pers",)] = pp
                for sh in shs:
                    for k in DRAWS:
                        for m in PAIR[arm]:
                            key = f"{m}|{sh:g}|{k}"
                            assert key in z.files, f"{tag}: missing {key}"
                            p = z[key].astype(np.float64)
                            assert len(p) == len(y) and np.isfinite(p).all(), f"{tag} {key}"
                            rep = met[(met.model == m) & np.isclose(met.share, sh) & (met.draw == k)]["pr"]
                            assert len(rep) == 1 and abs(average_precision_score(y, p) - rep.iloc[0]) < 1e-5, f"{tag} {key}"
                            F[(arm, m, sh, sd, k)] = p
    gid, ym, y = ref
    lab = ctry.reindex(pd.MultiIndex.from_arrays([gid, ym])).to_numpy()
    assert not pd.isna(lab).any(), f"{rg}: test rows without a country label"
    inC = lab == country
    tm = target_month(ym)
    ctx.append(dict(target=TARGET, country=country, region=rg, region_rows=len(y), region_pos=int(y.sum()),
                    country_rows=int(inC.sum()), country_pos=int(y[inC].sum()),
                    share_rows=float(inC.mean()), share_pos=float(y[inC].sum() / y.sum()),
                    first_target=tm.min(), last_target=tm.max()))
    print(f"[{country}] {inC.sum():,} of {len(y):,} {rg} test rows, {y[inC].sum():,} of {y.sum():,} positives", flush=True)

    for wi, (wname, (lo_m, hi_m)) in enumerate(WINDOWS[country].items()):
        mask = inC.copy()
        if lo_m is not None: mask &= tm >= lo_m
        if hi_m is not None: mask &= tm <= hi_m
        idx = np.where(mask)[0]; yw = y[idx]
        assert len(idx) > 0 and 0 < yw.sum() < len(yw), f"{country} {wname}: empty or one class"
        units, uidx = np.unique(gid[idx], return_inverse=True)
        P = {k: prep(v[idx], yw) for k, v in F.items()}
        w1 = np.ones(len(idx))
        kchk = next(k for k in F if k[0] == ARMS[0])
        assert abs(ap_w(P[kchk], w1) - average_precision_score(yw, F[kchk][idx])) < 1e-9, "fast AP disagrees with sklearn"
        pers_pr = average_precision_score(yw, F[("pers",)][idx])

        comps = [(a, sh) for a in ARMS for sh in SHARES if not (a == "finetune" and sh in ne)]

        def stats(w):
            out = {}
            for a, sh in comps:
                hi, lo = PAIR[a]
                out[(a, sh)] = float(np.mean([ap_w(P[(a, hi, sh, sd, k)], w) - ap_w(P[(a, lo, sh, sd, k)], w)
                                              for sd in SEEDS for k in DRAWS]))
            return out

        est = stats(w1)
        rng = np.random.default_rng([BOOT_SEED, ci, wi])
        dr = {c: np.empty(B) for c in comps}
        for b in range(B):
            wc = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(np.float64)
            st = stats(wc[uidx])
            for c in comps: dr[c][b] = st[c]
        for a, sh in comps:
            hi, lo = PAIR[a]
            x = dr[(a, sh)]; assert np.isfinite(x).all()
            ph = [average_precision_score(yw, F[(a, hi, sh, sd, k)][idx]) for sd in SEEDS for k in DRAWS]
            pl = [average_precision_score(yw, F[(a, lo, sh, sd, k)][idx]) for sd in SEEDS for k in DRAWS]
            assert abs((np.mean(ph) - np.mean(pl)) - est[(a, sh)]) < 1e-9
            d_ = np.array(ph) - np.array(pl)
            rows.append(dict(target=TARGET, country=country, window=wname, first_target=tm[idx].min(), last_target=tm[idx].max(),
                             arm=a, share=sh, n=len(idx), pos=int(yw.sum()), n_cells=len(units),
                             gain=est[(a, sh)], lo95=float(np.percentile(x, 2.5)), hi95=float(np.percentile(x, 97.5)),
                             boot_share_le0=float(np.mean(x <= 0)), runs_positive=float(np.mean(d_ > 0)), n_runs=len(d_),
                             pr_hi=float(np.mean(ph)), pr_lo=float(np.mean(pl)), pr_persistence=pers_pr,
                             hi_minus_persistence=float(np.mean(ph) - pers_pr), B=B))
        print(f"  [{country} {wname}] targets {tm[idx].min()}..{tm[idx].max()} rows={len(idx):,} pos={int(yw.sum()):,} "
              f"cells={len(units):,} persistence PR={pers_pr:.4f}", flush=True)

D = pd.DataFrame(rows)
exp_n = sum(len(WINDOWS[c]) for c, _ in CASES) * len(ARMS) * len(SHARES)
assert len(D) <= exp_n and len(D) > 0
C = pd.DataFrame(ctx)
C.to_csv(OUT_C + ".part", index=False); os.replace(OUT_C + ".part", OUT_C)
D.to_csv(OUT + ".part", index=False); os.replace(OUT + ".part", OUT)
pd.set_option("display.width", 200)
print("\n" + C.round(4).to_string(index=False))
print("\n" + D[["country", "window", "arm", "share", "pos", "n_cells", "gain", "lo95", "hi95", "runs_positive",
                "pr_hi", "pr_lo", "pr_persistence"]].round(4).to_string(index=False), flush=True)
print(f"[done] {OUT}", flush=True)

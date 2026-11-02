"""xd_m_boot.py (2026-10-25) -- copy of xd_c4_boot.py (9d7877fa) for the M Americas-held-out encoder (Jared: main size M).
Configs: C0 finetune 1% rich and C1 nested paper base, L (as run) and M (_capM). Writes outputs/crossdomain/xd_m_boot.csv.
xd_c4_boot.py docstring follows.
xd_c4_boot.py (2026-10-24) -- bootstrap for C0 (pre-registered, as run) and C4 (corrected construction), with the
Codex 2026-10-23 #1 fix: a verdict is printed ONLY when the exact required manifest is present
(seeds {123,456,789} x draws {0..4}; draw 0 only at share 1.0), every required model key exists, predictions are finite,
and (grid_id, year_month, y) match across seeds. Otherwise the row is INCOMPLETE and carries no verdict.
Effect = mean over (seed, draw) of (dev_a - dev_b) / dev_a (> 0 means b better).
CIs: ci_cell = paired cell bootstrap (the pre-registered rule; B=1000, seed 123);
     ci_2way = cells AND (seed, draw) pairs resampled (adds training randomness; Codex/fable suggestion).
Also reports ub_below_sesoi = upper two-way bound < 2% (Codex: a null is "no meaningful transfer" only if this holds).
Pre-registered verdict (finetune, share 0.01): (a) effect >= 2%, (b) ci_cell excludes 0, (c) top2 beats top2_rand.
Writes outputs/crossdomain/xd_c4_boot.csv. Env XC_DIR (default outputs/crossdomain)."""
import os
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
D = os.environ.get("XC_DIR", f"{R}/outputs/crossdomain")
B, SEED, SESOI = 1000, 123, 0.02
SEEDS, DRAWS = [123, 456, 789], [0, 1, 2, 3, 4]


def unit_dev(y, mu):
    mu = np.clip(mu, 1e-8, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (t - (y - mu))


FT = [("scratch", "top2"), ("top2_rand", "top2"), ("scratch", "full"), ("scratch", "top2_rand")]
NE = [("base", "emb_base"), ("rand_emb_base", "emb_base"), ("base", "rand_emb_base")]
CONFIGS = [("C0_L", "finetune", "finetune_americas_s{S}_rich_f0.01", [0.01], FT),
           ("C0_M", "finetune", "finetune_americas_s{S}_rich_f0.01_capM", [0.01], FT),
           ("C1_L", "nested", "nested_americas_s{S}_rich_bpaper", [0.0025, 0.01, 0.05, 0.1, 1.0], NE),
           ("C1_M", "nested", "nested_americas_s{S}_rich_bpaper_capM", [0.0025, 0.01, 0.05, 0.1, 1.0], NE)]
rows = []
q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
for rung, arm, stem, shares, contrasts in CONFIGS:
    files = {s: f"{D}/preds_brazil_{stem.replace('{S}', str(s))}.npz" for s in SEEDS}
    present = {s: f for s, f in files.items() if os.path.exists(f)}
    base_row = dict(rung=rung, arm=arm, config=stem.replace("_s{S}", ""))
    if len(present) < len(SEEDS):
        for sh in shares:
            for a, b in contrasts:
                rows.append(dict(base_row, share=sh, baseline=a, model=b, status=f"INCOMPLETE seeds {sorted(present)}"))
        print(f"[incomplete] {stem}: seeds present {sorted(present)}", flush=True); continue
    P = {s: np.load(f) for s, f in present.items()}
    z0 = P[SEEDS[0]]; gid, ym, y = z0["__grid_id"], z0["__year_month"], z0["__y"].astype(np.float64)
    for s in SEEDS:
        assert np.array_equal(P[s]["__grid_id"], gid) and np.array_equal(P[s]["__year_month"], ym) and np.array_equal(P[s]["__y"], z0["__y"]), \
            f"{stem}: test rows differ across seeds"
    cells, inv = np.unique(gid, return_inverse=True); nC = len(cells)
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, nC, nC), minlength=nC) for _ in range(B)]).astype(np.float64)
    for sh in shares:
        req_draws = [0] if sh >= 1.0 else DRAWS
        for a, b in contrasts:
            need = [(m, s, k) for m in (a, b) for s in SEEDS for k in req_draws]
            keys = {(m, s, k): f"{m}|{sh:g}|{k}" for m, s, k in need}
            miss = [x for x in need if keys[x] not in P[x[1]].files]
            bad = [x for x in need if x not in miss and not np.isfinite(P[x[1]][keys[x]]).all()]
            if miss or bad:
                rows.append(dict(base_row, share=sh, baseline=a, model=b, status=f"INCOMPLETE missing {len(miss)} nonfinite {len(bad)}"))
                continue
            S = {x: np.bincount(inv, weights=unit_dev(y, P[x[1]][keys[x]].astype(np.float64)), minlength=nC) for x in need}
            pairs = [(s, k) for s in SEEDS for k in req_draws]
            eff = float(np.mean([(S[(a, s, k)].sum() - S[(b, s, k)].sum()) / S[(a, s, k)].sum() for s, k in pairs]))
            per = np.stack([(W @ S[(a, s, k)] - W @ S[(b, s, k)]) / (W @ S[(a, s, k)]) for s, k in pairs], axis=1)  # (B, P)
            bc = per.mean(1)
            PI = np.random.default_rng([SEED, 7]).integers(0, len(pairs), size=(B, len(pairs)))
            b2 = np.take_along_axis(per, PI, axis=1).mean(1)
            per_seed = ";".join(f"{s}:{np.mean([(S[(a, s, k)].sum() - S[(b, s, k)].sum()) / S[(a, s, k)].sum() for k in req_draws]):+.4f}" for s in SEEDS)
            rows.append(dict(base_row, share=sh, baseline=a, model=b, status="complete", n_pairs=len(pairs),
                             dev_baseline=float(np.mean([S[(a, s, k)].sum() for s, k in pairs]) / len(y)),
                             dev_model=float(np.mean([S[(b, s, k)].sum() for s, k in pairs]) / len(y)),
                             rel_reduction=eff, ci_cell_lo=q(bc)[0], ci_cell_hi=q(bc)[1],
                             ci_2way_lo=q(b2)[0], ci_2way_hi=q(b2)[1], ub_below_sesoi=bool(q(b2)[1] < SESOI),
                             per_seed=per_seed))
out = pd.DataFrame(rows)
out.to_csv(f"{D}/xd_m_boot.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.drop(columns=[c for c in ["per_seed"] if c in out]).round(4).to_string(index=False))
for rung, cfg in [("C0_L", "finetune_americas_rich_f0.01"), ("C0_M", "finetune_americas_rich_f0.01_capM")]:
    h = out[(out.config == cfg) & (out.baseline == "scratch") & (out.model == "top2")]
    r = out[(out.config == cfg) & (out.baseline == "top2_rand") & (out.model == "top2")]
    if len(h) != 1 or len(r) != 1 or h.iloc[0].status != "complete" or r.iloc[0].status != "complete":
        print(f"VERDICT {rung} {cfg}: INCOMPLETE (no verdict)"); continue
    h, r = h.iloc[0], r.iloc[0]
    ok = (h.rel_reduction >= SESOI, h.ci_cell_lo > 0, r.rel_reduction > 0)
    print(f"VERDICT {rung} {cfg}: {'TRANSFER' if all(ok) else 'NO TRANSFER (pre-registered rule)'} | "
          f"top2 vs scratch {h.rel_reduction:+.4f} cell[{h.ci_cell_lo:+.4f},{h.ci_cell_hi:+.4f}] 2way[{h.ci_2way_lo:+.4f},{h.ci_2way_hi:+.4f}] | "
          f"top2 vs top2_rand {r.rel_reduction:+.4f} 2way[{r.ci_2way_lo:+.4f},{r.ci_2way_hi:+.4f}] | (a,b,c)={tuple(bool(x) for x in ok)}")
print("XM_DONE")

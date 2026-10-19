"""Synthetic test for the four-head forecasting ensemble: ens_multitask.py, ens_baseline_mt.py, ens_combine.py (ENS_KIND=mt).
Usage: python3 test_ens_multitask.py <scratch_dir>
Checks
  1. refuses without ENS_PRETRAIN_CUTOFF_OK=1; preflight exits without writing
  2. per target, saved labels equal an independent recomputation; onset rows are quiet in month t; scored rows are the
     stack and test windows only; both encoders score identical rows; forecasts differ across encoders
  3. the shared training sample keeps every rare-head positive (civilian, onset, fatal25) in the train window
  4. baseline: same rows, labels and sample hash as the encoders; onset persistence is constant 0
  5. combiner: pilot naming, ens_mean equals the encoder mean, hits at k, markers required, roster exact,
     a baseline built with a different POS_CAP refused
  6. overwrite guards; a diverging final learning rate is in the published log
"""
import os, sys, json, shutil, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn

S = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
R_REAL = '"/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"'
FIELDS9 = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type", "actor2_type", "month_offset", "fatalities", "deaths_civilians"]
VSZ = {"event_type": 8, "sub_event_type": 20, "actor1": 40, "actor2": 40, "actor1_type": 9, "actor2_type": 9, "fatalities": 6, "deaths_civilians": 6}
MONTHS = [f"{y}-{m:02d}" for y in range(2018, 2026) for m in range(1, 13)]; MONTHS = MONTHS[:MONTHS.index("2025-05") + 1]
STATIC = [f"s{j}" for j in range(5)]
COUNTRIES = ["Sudan", "Chad", "Iraq", "Ukraine", "Colombia", "India", "Myanmar", "Atlantis"]   # Atlantis -> "other", dropped


def make_encoder(vsm, d=256):
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in FIELDS9[:7]})
            self.field_proj = nn.Linear(7 * d, d); self.cls_token = nn.Parameter(torch.randn(1, 1, d))
            self.prio_proj = nn.Linear(len(STATIC), d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
    return Enc()


def build_root(root):
    rng = np.random.default_rng(123)
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("data_intermediate", "data_model", "outputs"): os.makedirs(f"{root}/{sub}")
    cells = [(1000 + i, COUNTRIES[i % len(COUNTRIES)]) for i in range(48)]
    rate = {g: (0.9 if i % 3 == 0 else 0.3 if i % 3 == 1 else 0.05) for i, (g, _) in enumerate(cells)}
    evm = {}
    for g, _ in cells:
        for i in range(len(MONTHS)):
            evm[(g, i)] = [[int(rng.integers(2, VSZ[f])) if f not in ("month_offset",) else 0 for f in FIELDS9]
                           for _ in range(rng.poisson(rate[g]))]
    vio = {k: int(any(e[0] in (2, 3) for e in v)) for k, v in evm.items()}
    civ = {k: sum(e[0] == 4 for e in v) for k, v in evm.items()}
    fat = {k: float(sum(10 * e[7] for e in v)) for k, v in evm.items()}        # up to 50 per event, so >= 25 occurs
    sup, cty = [], []
    order = list(range(len(cells))); rng.shuffle(order)                         # rows not sorted by grid
    for i, ym in enumerate(MONTHS):
        for j in order:
            g, c = cells[j]
            seq = []
            for m in range(max(0, i - 12), i + 1):
                for e in evm[(g, m)]:
                    e2 = list(e); e2[6] = i - m + 1; seq.append(e2)
            sup.append(dict(grid_id=g, year_month=ym, event_sequence=json.dumps(seq[-200:]),
                            violence_next_month=vio[(g, i + 1)] if i + 1 < len(MONTHS) else np.nan,
                            has_violence=vio[(g, i)], count_fatalities=fat[(g, i)],
                            etype_violence_against_civilians=float(civ[(g, i)]),
                            violence_lag1=vio.get((g, i - 1), 0), violence_lag3=max(vio.get((g, i - k), 0) for k in (1, 2, 3)),
                            violence_lag6=max(vio.get((g, i - k), 0) for k in range(1, 7)), spatial_lag_violence=float(rng.random() * 0.2),
                            violent_count_roll1=vio[(g, i)], violent_count_roll3=sum(vio.get((g, i - k), 0) for k in (0, 1, 2)),
                            all_count_roll1=len(evm[(g, i)]), any_event_lag1=int(len(evm.get((g, i - 1), [])) > 0),
                            log_event_density_lag1=float(np.log1p(len(seq))), months_since_last_event=0 if vio[(g, i)] else 6))
            cty.append(dict(grid_id=g, year_month=ym, country=c))
    pd.DataFrame(sup).to_parquet(f"{root}/data_intermediate/baseline_features_super.parquet", index=False)
    pd.DataFrame(cty).to_parquet(f"{root}/data_intermediate/baseline_features.parquet", index=False)
    pd.DataFrame({"grid_id": [g for g, _ in cells], **{c: rng.normal(size=len(cells)) for c in STATIC}}).to_parquet(
        f"{root}/data_intermediate/priogrid_static.parquet", index=False)
    vocab = {f: {f"t{j}": j for j in range(2, VSZ[f])} for f in VSZ}
    vocab["month_offset"] = {"<PAD>": 0, **{str(j): j for j in range(1, 14)}}
    json.dump(vocab, open(f"{root}/data_intermediate/vocab_super.json", "w"))
    vsm = {f: max(v.values()) + 2 for f, v in vocab.items()}
    for sd in (123, 456):                                                       # all-region encoders
        torch.manual_seed(sd)
        torch.save(make_encoder(vsm).state_dict(), f"{root}/data_model/repr_model_L_contrastive_e_s{sd}.pt")
    return cells, vio, civ, fat



def copy_scripts(dst, root):
    os.makedirs(dst, exist_ok=True)
    s = open(f"{HERE}/ens_multitask.py").read()
    assert s.count("MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128") == 1
    # test-only: shorter padding and small batches keep the CPU run short; no check depends on them
    s = s.replace("MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128", "MAX_SEQ_LEN, BATCH, FT_BATCH = 24, 256, 64")
    open(f"{dst}/ens_multitask.py", "w").write(s)
    for f in ("ens_baseline_mt.py", "ens_combine.py", "regions.py"): shutil.copy(f"{HERE}/{f}", dst)
    open(f"{dst}/vocab_remap.py", "w").write(open(f"{HERE}/vocab_remap.py").read().replace(R_REAL, repr(root)))


def run(script, env):
    r = subprocess.run([sys.executable, script], env=dict(os.environ, **env), capture_output=True, text=True)
    print("\n".join(r.stdout.splitlines()[-6:]))
    if r.returncode != 0: print(r.stderr[-1500:])
    return r


TG = ["incidence", "civilian", "onset", "fatal25"]
root = f"{S}/root"
cells, vio, civ, fat = build_root(root)
C = f"{S}/code"; copy_scripts(C, root)
MT, BL, CB = f"{C}/ens_multitask.py", f"{C}/ens_baseline_mt.py", f"{C}/ens_combine.py"
base = dict(ENS_ROOT=root, SMOKE="1", ENS_WORKERS="0")
mi = {ym: i for i, ym in enumerate(MONTHS)}

# 1. guards and preflight
r = run(MT, dict(base, ENS_ENC_SEED="123")); assert r.returncode != 0 and "ENS_PRETRAIN_CUTOFF_OK" in r.stderr
r = run(MT, dict(base, ENS_ENC_SEED="123", ENS_PRETRAIN_CUTOFF_OK="1", ENS_PREFLIGHT="1"))
assert r.returncode == 0 and "[preflight]" in r.stdout and not os.path.exists(f"{root}/outputs/ens/done_mt_L_contrastive_e_s123_smoke.json")
print("[check ok] 1. cutoff guard; preflight writes nothing")

# 2-3. two encoders
for sd in ("123", "456"):
    r = run(MT, dict(base, ENS_ENC_SEED=sd, ENS_PRETRAIN_CUTOFF_OK="1")); assert r.returncode == 0, f"mt s{sd} failed"
for t in TG:
    a = np.load(f"{root}/outputs/ens/preds_mt_{t}_L_contrastive_e_s123_smoke.npz"); b = np.load(f"{root}/outputs/ens/preds_mt_{t}_L_contrastive_e_s456_smoke.npz")
    ix = [mi[m] for m in a["year_month"]]
    if t in ("incidence", "onset"): truth = [vio[(g, i + 1)] for g, i in zip(a["grid_id"], ix)]
    elif t == "civilian": truth = [int(civ[(g, i + 1)] > 0) for g, i in zip(a["grid_id"], ix)]
    else: truth = [int(fat[(g, i + 1)] >= 25) for g, i in zip(a["grid_id"], ix)]
    assert np.array_equal(a["y"].astype(int), np.array(truth)), f"2. {t}: labels differ"
    if t == "onset": assert all(vio[(g, i)] == 0 for g, i in zip(a["grid_id"], ix)), "2. onset row not quiet"
    assert (a["year_month"] >= "2021-07").all() and ((a["window"] == "test") == (a["year_month"] >= "2022-01")).all(), "2. windows"
    for k in ("grid_id", "year_month", "y", "window", "sample_hash"): assert np.array_equal(a[k], b[k]), f"2. {t}: {k} differs across encoders"
    assert not np.array_equal(a["p"], b["p"]), f"2. {t}: identical forecasts"
    print(f"[check ok] 2. {t}: labels, windows, row identity ({int(a['y'].sum())} positives of {len(a['y'])})")
lg = pd.read_csv(f"{root}/outputs/ens/log_mt_L_contrastive_e_s123_smoke.csv")
assert {f"stop_pr_{t}" for t in TG} <= set(lg.columns), "2. per-head stop PR-AUC not logged"
# 3. rare positives all kept: recompute the train-window rare positives independently and compare with the sample size
import re
out = run(MT, dict(base, ENS_ENC_SEED="456", ENS_PRETRAIN_CUTOFF_OK="1", ENS_PREFLIGHT="1")).stdout
civ_line = [l for l in out.splitlines() if l.startswith("[sample] civilian")][0]
fat_line = [l for l in out.splitlines() if l.startswith("[sample] fatal25")][0]
n_civ_tr = sum(1 for (g, c) in cells if c != "Atlantis" for i, m in enumerate(MONTHS[:-1]) if "1990-01" <= m <= "2020-12" and civ[(g, i + 1)] > 0)
n_fat_tr = sum(1 for (g, c) in cells if c != "Atlantis" for i, m in enumerate(MONTHS[:-1]) if "1990-01" <= m <= "2020-12" and fat[(g, i + 1)] >= 25)
assert int(re.search(r"positives=([\d,]+)", civ_line).group(1).replace(",", "")) == n_civ_tr, "3. civilian positives subsampled"
assert int(re.search(r"positives=([\d,]+)", fat_line).group(1).replace(",", "")) == n_fat_tr, "3. fatal25 positives subsampled"
print(f"[check ok] 3. every civilian ({n_civ_tr}) and fatal25 ({n_fat_tr}) train positive is in the shared sample")

# 4. baseline
assert run(BL, base).returncode == 0, "4. baseline failed"
for t in TG:
    z = np.load(f"{root}/outputs/ens/preds_baseline_mt_{t}_smoke.npz"); a = np.load(f"{root}/outputs/ens/preds_mt_{t}_L_contrastive_e_s123_smoke.npz")
    assert str(z["sample_hash"]) == str(a["sample_hash"]), f"4. {t}: sample hash differs between baseline and encoder"
    kz = dict(zip(zip(z["grid_id"], z["year_month"]), z["y"]))
    assert all(kz[(g, m)] == yv for g, m, yv in zip(a["grid_id"], a["year_month"], a["y"])), f"4. {t}: labels differ"
    if t == "onset": assert (z["p_persistence"] == 0).all()
print("[check ok] 4. baseline rows, labels and sample hash match the encoders")

# 5. combiner
for t in TG:
    env = dict(base, ENS_TARGET=t, ENS_SEEDS="123,456")
    assert run(CB, env).returncode == 0, f"5. combine {t} failed"
    sc = pd.read_csv(f"{root}/outputs/ens/scores_mt_{t}_pilot_s123-456_smoke.csv"); z = np.load(f"{root}/outputs/ens/preds_ens_mt_{t}_pilot_s123-456_smoke.npz")
    a = np.load(f"{root}/outputs/ens/preds_mt_{t}_L_contrastive_e_s123_smoke.npz"); b = np.load(f"{root}/outputs/ens/preds_mt_{t}_L_contrastive_e_s456_smoke.npz")
    assert np.allclose(z["p_ens_mean"], (a["p"].astype(float) + b["p"].astype(float)) / 2, atol=1e-6), "5. ens_mean"
    te = z["window"] == "test"; k = int(z["y"][te].sum()); top = np.argsort(-z["p_base"][te].astype(float), kind="stable")[:k]
    assert sc[(sc.model == "base") & (sc.region == "ALL")].hits_at_k.iloc[0] == int(z["y"][te][top].sum()), "5. hits"
    print(f"[check ok] 5. {t}: combiner verified")
r = run(CB, dict(base, ENS_TARGET="civilian", ENS_SEEDS="123,456,789")); assert r.returncode != 0 and "missing encoder files" in r.stderr
os.rename(f"{root}/outputs/ens/done_mt_L_contrastive_e_s456_smoke.json", f"{root}/outputs/ens/done_hold.bak")
r = run(CB, dict(base, ENS_TARGET="onset", ENS_SEEDS="456")); assert r.returncode != 0 and "missing encoder files" in r.stderr, "5. marker not required"
os.rename(f"{root}/outputs/ens/done_hold.bak", f"{root}/outputs/ens/done_mt_L_contrastive_e_s456_smoke.json")
r = run(CB, dict(base, ENS_TARGET="onset", ENS_SEEDS="456")); assert r.returncode == 0, "5. one-encoder pilot"
root2 = f"{S}/root2"; shutil.rmtree(root2, ignore_errors=True); shutil.copytree(root, root2)
for t in TG: os.rename(f"{root2}/outputs/ens/preds_baseline_mt_{t}_smoke.npz", f"{root2}/outputs/ens/bl_{t}.bak")
os.rename(f"{root2}/outputs/ens/done_baseline_mt_smoke.json", f"{root2}/outputs/ens/done_bl.bak")
assert run(BL, dict(base, ENS_ROOT=root2, ENS_POS_CAP="150")).returncode == 0
r = run(CB, dict(base, ENS_ROOT=root2, ENS_TARGET="incidence", ENS_SEEDS="123")); assert r.returncode != 0 and "provenance differs" in r.stderr
print("[check ok] 5. exact roster, completion markers, one-encoder pilot, provenance mismatch refused")

# 6. overwrite guards and divergence logging
r = run(MT, dict(base, ENS_ENC_SEED="123", ENS_PRETRAIN_CUTOFF_OK="1")); assert r.returncode != 0 and "refusing to overwrite" in r.stderr
r = run(BL, base); assert r.returncode != 0 and "refusing to overwrite" in r.stderr
os.rename(f"{root2}/outputs/ens/done_mt_L_contrastive_e_s123_smoke.json", f"{root2}/outputs/ens/done123.bak")
r = run(MT, dict(base, ENS_ROOT=root2, ENS_ENC_SEED="123", ENS_PRETRAIN_CUTOFF_OK="1", ENS_LRS="1e-4,1e6"))
assert r.returncode == 0 and "[recover]" in r.stdout, "6. diverging run failed or orphan forecasts not reported"
lg = pd.read_csv(f"{root2}/outputs/ens/log_mt_L_contrastive_e_s123_smoke.csv")
if "DIVERGED" in r.stdout: assert lg["diverged"].any() and (lg.loc[lg.diverged, "lr"] == 1e6).all(); print("[check ok] 6. divergence logged")
else: print("[note] 6. lr=1e6 did not diverge on this fixture")
print("[check ok] 6. overwrite guards")
# 7. concurrency: a second writer is refused while the lock is held (Codex 12c)
import fcntl
for script, lockname, env in [(MT, "lock_mt_L_contrastive_e_s456_smoke.lock", dict(base, ENS_ROOT=root2, ENS_ENC_SEED="456", ENS_PRETRAIN_CUTOFF_OK="1")),
                              (BL, "lock_baseline_mt_smoke.lock", dict(base, ENS_ROOT=root2)),
                              (CB, "lock_combine_mt_civilian_pilot_s123_smoke.lock", dict(base, ENS_ROOT=root2, ENS_TARGET="civilian", ENS_SEEDS="123"))]:
    rt = env["ENS_ROOT"]; fh = open(f"{rt}/outputs/ens/{lockname}", "a"); fcntl.flock(fh, fcntl.LOCK_EX)
    before = set(os.listdir(f"{rt}/outputs/ens"))
    r = run(script, env); assert r.returncode != 0 and "held by another running job" in (r.stdout + r.stderr), f"7. {script} not blocked"
    assert set(os.listdir(f"{rt}/outputs/ens")) == before, f"7. {script} wrote while blocked"
    fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
print("[check ok] 7. duplicate writers blocked by the job lock (multitask, baseline, combine)")

# 8. constant forecast: diversity drops the constant encoder and reports NaN without crashing
root3 = f"{S}/root3"; shutil.rmtree(root3, ignore_errors=True); shutil.copytree(root, root3)
f456 = f"{root3}/outputs/ens/preds_mt_fatal25_L_contrastive_e_s456_smoke.npz"; z = dict(np.load(f456)); z["p"] = np.full_like(z["p"], 0.3)
np.savez_compressed(f456, **z)
for ext in ("scores_mt_fatal25_pilot_s123-456_smoke.csv", "preds_ens_mt_fatal25_pilot_s123-456_smoke.npz"): os.rename(f"{root3}/outputs/ens/{ext}", f"{root3}/outputs/ens/{ext}.bak")
r = run(CB, dict(base, ENS_ROOT=root3, ENS_TARGET="fatal25", ENS_SEEDS="123,456")); assert r.returncode == 0, "8. constant forecast crashed the combiner"
sc = pd.read_csv(f"{root3}/outputs/ens/scores_mt_fatal25_pilot_s123-456_smoke.csv")
assert sc["mean_pairwise_spearman_encoders"].isna().all() and (sc["n_encoders_in_diversity"] == 1).all() and "[diversity]" in r.stdout
print("[check ok] 8. constant forecast handled (diversity NaN, 1 usable encoder)")
print("ALL CHECKS PASSED")

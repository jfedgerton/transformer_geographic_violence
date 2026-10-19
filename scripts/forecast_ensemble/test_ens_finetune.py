"""Synthetic test for ens_finetune.py (stage 1 of the forecasting ensemble).
Usage: python3 test_ens_finetune.py <scratch_dir>
Checks
  1. refuses to run without ENS_PRETRAIN_CUTOFF_OK=1
  2. every target: saved labels equal an independent recomputation from the synthetic events
  3. onset rows all have no violent event in month t; fatal25 uses next-month fatalities
  4. only stack (2021-07..2021-12) and test (>= 2022-01) rows are scored; train/stop windows never appear
  5. two different encoders on the same target score identical rows with identical labels
  6. refuses to overwrite a completed job
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
    s = open(f"{HERE}/ens_finetune.py").read()
    assert s.count("MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128") == 1
    # test-only: shorter padding and small batches keep the CPU run short; no check depends on them
    s = s.replace("MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128", "MAX_SEQ_LEN, BATCH, FT_BATCH = 24, 256, 64")
    open(f"{dst}/ens_finetune.py", "w").write(s)
    shutil.copy(f"{HERE}/regions.py", dst)
    open(f"{dst}/vocab_remap.py", "w").write(open(f"{HERE}/vocab_remap.py").read().replace(R_REAL, repr(root)))
    return f"{dst}/ens_finetune.py"


def run(script, env):
    r = subprocess.run([sys.executable, script], env=dict(os.environ, **env), capture_output=True, text=True)
    print("\n".join(r.stdout.splitlines()[-8:]))
    if r.returncode != 0: print(r.stderr[-2000:])
    return r


root = f"{S}/root"
cells, vio, civ, fat = build_root(root)
scr = copy_scripts(f"{S}/code", root)
base = dict(ENS_ROOT=root, SMOKE="1", ENS_WORKERS="0", ENS_POS_CAP="200")

# 1. cutoff guard
r = run(scr, dict(base, ENS_ENC_SEED="123", ENS_TARGET="incidence"))
assert r.returncode != 0 and "ENS_PRETRAIN_CUTOFF_OK" in r.stderr
print("[check ok] 1. refuses to run without the pretraining-cutoff confirmation")

mi = {ym: i for i, ym in enumerate(MONTHS)}
for tgt in ("incidence", "civilian", "onset", "fatal25"):
    for sd in ("123", "456"):
        r = run(scr, dict(base, ENS_ENC_SEED=sd, ENS_TARGET=tgt, ENS_PRETRAIN_CUTOFF_OK="1"))
        assert r.returncode == 0, f"{tgt} s{sd} failed"
    a = np.load(f"{root}/outputs/ens/preds_ft_{tgt}_L_contrastive_e_s123_smoke.npz")
    b = np.load(f"{root}/outputs/ens/preds_ft_{tgt}_L_contrastive_e_s456_smoke.npz")
    ix = [mi[m] for m in a["year_month"]]
    if tgt in ("incidence", "onset"): truth = [vio[(g, i + 1)] for g, i in zip(a["grid_id"], ix)]
    elif tgt == "civilian": truth = [int(civ[(g, i + 1)] > 0) for g, i in zip(a["grid_id"], ix)]
    else: truth = [int(fat[(g, i + 1)] >= 25) for g, i in zip(a["grid_id"], ix)]
    assert np.array_equal(a["y"].astype(int), np.array(truth)), f"2. {tgt}: labels differ from recomputation"
    if tgt == "onset": assert all(vio[(g, i)] == 0 for g, i in zip(a["grid_id"], ix)), "3. onset row with violence in month t"
    assert (a["year_month"] >= "2021-07").all() and set(np.unique(a["window"])) <= {"stack", "test"}, "4. train/stop rows scored"
    assert ((a["window"] == "test") == (a["year_month"] >= "2022-01")).all(), "4. window labels wrong"
    assert "other" not in set(a["region"]), "4. unmapped country kept"
    for k in ("grid_id", "year_month", "y", "window"): assert np.array_equal(a[k], b[k]), f"5. {tgt}: encoders scored different rows ({k})"
    assert not np.array_equal(a["p"], b["p"]), f"5. {tgt}: two encoders gave identical forecasts"
    print(f"[check ok] 2-5. {tgt}: labels, windows and row identity verified ({int(a['y'].sum())} positives of {len(a['y'])})")

r = run(scr, dict(base, ENS_ENC_SEED="123", ENS_TARGET="incidence", ENS_PRETRAIN_CUTOFF_OK="1"))
assert r.returncode != 0 and "refusing to overwrite" in r.stderr
print("[check ok] 6. refuses to overwrite a completed job")
r = run(scr, dict(base, ENS_ENC_SEED="456", ENS_TARGET="incidence", ENS_PRETRAIN_CUTOFF_OK="1", ENS_PREFLIGHT="1"))
assert r.returncode == 0 and "[preflight]" in r.stdout, "6b. preflight failed"
print("[check ok] 6b. preflight reports memory and exits without writing")

# 7. baseline + combiner on a two-encoder pilot roster
for f in ("ens_baseline.py", "ens_combine.py"): shutil.copy(f"{HERE}/{f}", f"{S}/code/{f}")
for tgt in ("incidence", "civilian", "onset", "fatal25"):
    env = dict(base, ENS_TARGET=tgt, ENS_SEEDS="123,456")
    assert run(f"{S}/code/ens_baseline.py", env).returncode == 0, f"7. baseline {tgt} failed"
    assert run(f"{S}/code/ens_combine.py", env).returncode == 0, f"7. combine {tgt} failed"
    sc = pd.read_csv(f"{root}/outputs/ens/scores_{tgt}_pilot_s123-456_smoke.csv"); z = np.load(f"{root}/outputs/ens/preds_ens_{tgt}_pilot_s123-456_smoke.npz")
    assert not os.path.exists(f"{root}/outputs/ens/scores_{tgt}_smoke.csv"), "7. pilot took the production name"
    assert {"enc_s123", "enc_s456", "ens_mean", "ens_stack", "persistence", "base"} <= set(sc.model), "7. models missing"
    te = z["window"] == "test"
    a = np.load(f"{root}/outputs/ens/preds_ft_{tgt}_L_contrastive_e_s123_smoke.npz"); b = np.load(f"{root}/outputs/ens/preds_ft_{tgt}_L_contrastive_e_s456_smoke.npz")
    assert np.allclose(z["p_ens_mean"], (a["p"].astype(float) + b["p"].astype(float)) / 2, atol=1e-6), "7. ens_mean is not the encoder mean"
    k = int(z["y"][te].sum()); top = np.argsort(-z["p_base"][te].astype(float), kind="stable")[:k]
    assert sc[(sc.model == "base") & (sc.region == "ALL")].hits_at_k.iloc[0] == int(z["y"][te][top].sum()), "7. hits_at_k mismatch"
    if tgt == "onset": assert (z["p_persistence"] == 0).all(), "7. onset persistence should be constant 0"
    if tgt == "fatal25":
        mi_ = {ym: i for i, ym in enumerate(MONTHS)}
        assert np.allclose(z["p_persistence"], [fat[(g, mi_[m])] for g, m in zip(z["grid_id"], z["year_month"])]), "7. fatal25 persistence is not month-t fatalities"
    assert str(a["sample_hash"]) == str(np.load(f"{root}/outputs/ens/preds_baseline_{tgt}_smoke.npz")["sample_hash"]), "7. sample hash differs"
    print(f"[check ok] 7. {tgt}: pilot naming, provenance, combiners and hit counts verified")

# 8. guards (Codex 12)
r = run(f"{S}/code/ens_combine.py", dict(base, ENS_TARGET="incidence", ENS_SEEDS="123,456"))
assert r.returncode != 0 and "refusing to overwrite" in r.stderr, "8. overwrite guard"
r = run(f"{S}/code/ens_combine.py", dict(base, ENS_TARGET="civilian", ENS_SEEDS="123,456,789"))
assert r.returncode != 0 and "missing encoder files" in r.stderr, "8. incomplete roster accepted"
r = run(f"{S}/code/ens_combine.py", dict(base, ENS_TARGET="civilian", ENS_SEEDS="123"))
assert r.returncode == 0, "8. one-encoder pilot crashed"
assert pd.read_csv(f"{root}/outputs/ens/scores_civilian_pilot_s123_smoke.csv")["mean_pairwise_spearman_encoders"].isna().all()
print("[check ok] 8. overwrite guard, exact roster, one-encoder pilot")
root2 = f"{S}/root2"; shutil.rmtree(root2, ignore_errors=True); shutil.copytree(root, root2)
os.rename(f"{root2}/outputs/ens/preds_baseline_onset_smoke.npz", f"{root2}/outputs/ens/preds_baseline_onset_smoke.npz.bak")
assert run(f"{S}/code/ens_baseline.py", dict(base, ENS_ROOT=root2, ENS_TARGET="onset", ENS_POS_CAP="150")).returncode == 0
r = run(f"{S}/code/ens_combine.py", dict(base, ENS_ROOT=root2, ENS_TARGET="onset", ENS_SEEDS="123"))
assert r.returncode != 0 and "provenance differs" in r.stderr, "8. mismatched training sample accepted"
print("[check ok] 8. a baseline built on a different training sample is refused")
os.rename(f"{root2}/outputs/ens/preds_ft_civilian_L_contrastive_e_s123_smoke.npz", f"{root2}/outputs/ens/old.npz.bak")
r = run(scr, dict(base, ENS_ROOT=root2, ENS_ENC_SEED="123", ENS_TARGET="civilian", ENS_PRETRAIN_CUTOFF_OK="1", ENS_LRS="1e-4,1e6", ENS_MAX_EP="2"))
assert r.returncode == 0, "8. run with a diverging learning rate failed"
lg = pd.read_csv(f"{root2}/outputs/ens/log_ft_civilian_L_contrastive_e_s123_smoke.csv")
if "DIVERGED" in r.stdout:
    assert lg["diverged"].any() and (lg.loc[lg.diverged, "lr"] == 1e6).all(), "8. divergence missing from the published log"
    print("[check ok] 8. final-learning-rate divergence is in the published log")
else:
    print("[note] 8. lr=1e6 did not diverge on this fixture; divergence logging not exercised")
print("ALL CHECKS PASSED")

"""build_seen_tokens.py -- token ids each LORO conflict encoder saw in pretraining (input to vocab_remap.py).

Reproduces the row selection of train_repr_obj.py (REPR_VARIANT=holdout, REPR_LORO=<region>, REPR_MAX_ROWS=0,
REPR_PANEL / REPR_VOCAB at their defaults, which is how launch_loro_converge.sbatch runs it):
  baseline_features.parquet, split == "train", country not in the held-out region, no country_{HOLDOUT_COUNTRIES} == 1
  tokens = every field id in the last MAX_SEQ_LEN (200) events of each such row.
One streaming pass over the panel accumulates tokens per country; each region's set is the union over the countries
it does not hold out. Labels are never read.
Also writes, per region, the share of non-empty test-period cell-months (split == "test", 2022-01 on) whose input window
contains at least one actor token unseen in pretraining, for the appendix.
Outputs: data_intermediate/seen_tokens_loro_{region}.json (refuses to overwrite), outputs/vocab_unseen_counts.csv
"""
import os, sys, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF, REGIONS

PANEL = f"{R}/data_intermediate/baseline_features.parquet"
VOCAB = f"{R}/data_intermediate/vocab.json"
MAX_SEQ_LEN = 200
FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type", "actor2_type", "month_offset"]
# copied verbatim from train_repr_obj.py (HOLDOUT_COUNTRIES); these rows are dropped from every encoder's pretraining
HOLDOUT_COUNTRIES = ["Sudan", "Somalia", "Uganda", "Kenya", "Ethiopia", "South Africa",
                     "Iraq", "Tunisia", "Iran", "Algeria", "Morocco",
                     "Pakistan", "Bangladesh", "Nepal", "Sri Lanka",
                     "Myanmar", "Philippines", "Thailand", "South Korea",
                     "Ukraine", "France", "Germany",
                     "Mexico", "Venezuela", "Colombia", "Chile"]
OUTS = {rg: f"{R}/data_intermediate/seen_tokens_loro_{rg}.json" for rg in REGIONS}
for f in OUTS.values(): assert not os.path.exists(f), f"refusing to overwrite {f}"

vocab = json.load(open(VOCAB))
vsz = [max(vocab[f].values()) + 1 for f in FIELDS]
pf = pq.ParquetFile(PANEL)
names = pf.schema_arrow.names
hcols = [f"country_{c}" for c in HOLDOUT_COUNTRIES if f"country_{c}" in names]
print(f"[cfg] rows={pf.metadata.num_rows:,} groups={pf.metadata.num_row_groups} holdout cols={len(hcols)}", flush=True)
assert len(hcols) == len(HOLDOUT_COUNTRIES), f"missing holdout cols: {set(HOLDOUT_COUNTRIES) - {c[8:] for c in hcols}}"

acc = {}                      # country -> list of per-field boolean arrays (train rows, holdout-flag rows excluded)
test_ev = {}                  # country -> list of per-field unique ids, one entry per non-empty test row
n_rows = n_ev = n_drop = 0
t0 = time.time()
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["country", "split", "year_month", "event_sequence"] + hcols).to_pandas()
    hflag = (t[hcols].to_numpy() == 1).any(axis=1)          # same predicate as train_repr_obj.py (NaN -> kept)
    for c, sp, ym, s, hf in zip(t["country"].values, t["split"].values, t["year_month"].values,
                                t["event_sequence"].values, hflag):
        if not isinstance(s, str) or len(s) <= 2: continue
        if sp == "train":
            if hf: n_drop += 1; continue
            seq = json.loads(s)[-MAX_SEQ_LEN:]
            a = acc.get(c)
            if a is None: a = acc[c] = [np.zeros(v, dtype=bool) for v in vsz]
            arr = np.asarray(seq, dtype=np.int64)
            assert arr.ndim == 2 and arr.shape[1] >= len(FIELDS), f"bad event shape {arr.shape}"
            for k in range(len(FIELDS)): a[k][arr[:, k]] = True
            n_rows += 1; n_ev += len(seq)
        elif str(ym) >= "2022-01" and sp == "test":
            arr = np.asarray(json.loads(s)[-MAX_SEQ_LEN:], dtype=np.int64)
            test_ev.setdefault(c, []).append([np.unique(arr[:, k]) for k in range(len(FIELDS))])
    print(f"[group {g}] train rows used={n_rows:,} events={n_ev:,} holdout-flag rows dropped={n_drop:,} "
          f"{time.time()-t0:.0f}s", flush=True)

unknown = sorted(set(acc) - set(REGION_OF))
print(f"[warn] countries without a region (kept in every pool, as in train_repr_obj.py): {unknown}", flush=True)
rows = []
for rg in REGIONS:
    held = {c for c, r in REGION_OF.items() if r == rg}
    pool = [a for c, a in acc.items() if c not in held]
    assert pool, f"empty pretraining pool for {rg}"
    toks = {}
    for k, f in enumerate(FIELDS):
        ok = np.zeros(vsz[k], dtype=bool)
        for a in pool: ok |= a[k]
        toks[f] = np.where(ok)[0].tolist()
    assert 1 in toks["actor1"] and 1 in toks["actor2"], f"{rg}: <UNK> never seen in pretraining"
    json.dump(dict(region=rg, rule="split==train, country not held out, no holdout-country flag, last 200 events",
                   panel=PANEL, tokens=toks), open(OUTS[rg], "w"))
    ev = [e for c in held for e in test_ev.get(c, [])]
    r = dict(region=rg, test_rows_nonempty=len(ev))
    anyu = np.zeros(len(ev), dtype=bool)
    for k, f in enumerate(FIELDS):
        if f == "month_offset": continue
        ok = np.zeros(vsz[k], dtype=bool); ok[toks[f]] = True; ok[0] = True      # PAD is never a data token
        u = np.array([(~ok[e[k]]).any() for e in ev], dtype=bool)
        r[f"share_rows_{f}_unseen"] = float(u.mean()) if len(ev) else np.nan
        r[f"{f}_ids_unseen"] = int((~ok).sum())
        r[f"{f}_unk_seen"] = bool(ok[1])
        if f in ("actor1", "actor2"): anyu |= u
        elif u.any() and not ok[1]:
            print(f"[WARN] {rg} {f}: test rows hold ids unseen in pretraining and <UNK> was never trained; "
                  f"vocab_remap.py maps them to an untrained <UNK>", flush=True)
    r["share_rows_any_actor_unseen"] = float(anyu.mean()) if len(ev) else np.nan
    rows.append(r)
    print(f"[{rg}] {rows[-1]}", flush=True)
pd.DataFrame(rows).to_csv(f"{R}/outputs/vocab_unseen_counts.csv", index=False)
print("[done]", flush=True)

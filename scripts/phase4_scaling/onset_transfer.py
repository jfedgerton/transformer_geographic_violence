"""onset_transfer.py -- new-war demo: frozen LORO embedding + a few months of local data (Ukraine 2022, Sudan 2023).

Question: when a war starts, does the frozen pretrained embedding help a forecaster trained on only the first 1-3
months of that war in that country?

One job = one country, all three encoder seeds, all three adaptation windows.
  country   Ukraine: onset month 2022-02 (24 Feb 2022), held-out-region encoder = europe
            Sudan:   onset month 2023-04 (15 Apr 2023), held-out-region encoder = africa
  cells     PRIO-GRID cells whose most common non-missing country label is the target country
            (labels come from baseline_features.parquet; see Codex finding F09 on border cells). The label uses the
            cell's full 1990-2025 history: a fixed geographic cohort, not a forecast-time choice (Codex 7, O01).
            Ties go to the alphabetically first country (O02). Rows used for training and evaluation must also carry
            a country label in the held-out region, the same row filter as the main evaluation; month-t and 12-month
            violence histories use every row of the cell.
  window w  1, 2 or 3 months. Training rows are forecast origins t with outcome month t+1 in [M0, M0+w-1], i.e.
            origins M0-1 .. M0+w-2. Every training label is observed by the end of month M0+w-1.
  eval      the SAME rows for every w: origins M0+2 .. last origin with an outcome (outcomes M0+3 onward), so the
            largest training window (w=3, last outcome M0+2) never overlaps the evaluation outcomes.
  arms      persistence_t      violence in month t (no training)
            base               XGBoost on the share-grid features, trained on the window rows
            emb_base           same + frozen embedding from the held-out-region encoder (seed 123/456/789)
            region_base,       reference forecasts already saved by the main evaluation (trained on the region's
            region_emb_base    1990-2020 history, no onset data); sliced to the same eval rows
  subsets   all eval rows; "new" = eval rows whose cell had no violence in months t-11..t (12-month history required)
  metrics   PR-AUC, ROC-AUC, Brier, and hits at an equal alert budget (top-k, k = violent cell-months in the subset;
            a retrospective ranking diagnostic, not an operational alert rule)
XGBoost uses the share-grid settings with seed fixed at 123, so base is identical across encoder seeds.
Vocabulary: main specification (VOCAB_REMAP=1), tokens unseen in pretraining -> <UNK>, via vocab_remap.py.
Outputs (outputs/onset/, never overwritten; .part until the job finishes):
  metrics_onset_{country}.csv   one row per model x window x encoder seed x subset
  preds_onset_{country}.npz      eval-row forecasts for every model x window x seed, plus row identity
Env: ONSET_COUNTRY (Ukraine|Sudan), ONSET_ROOT (defaults to the Roar project root; the synthetic test overrides it).
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = os.environ.get("ONSET_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
COUNTRY = os.environ.get("ONSET_COUNTRY")
CASES = {"Ukraine": ("2022-02", "europe"), "Sudan": ("2023-04", "africa")}
assert COUNTRY in CASES, f"ONSET_COUNTRY must be one of {list(CASES)}, got {COUNTRY}"
M0, RG = CASES[COUNTRY]
SEEDS = [123, 456, 789]
WINDOWS = [1, 2, 3]
XGB_SEED = 123
MIN_POS = 5                      # minimum positives AND negatives to fit or score
EMBED_DIM, N_LAYERS, N_HEADS = 256, 6, 8
MAX_SEQ_LEN, BATCH = 200, 512
SUPER_FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type", "actor2_type",
                "month_offset", "fatalities", "deaths_civilians"]
SEL = [0, 1, 2, 3, 4, 5, 6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
BASE_COLS = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocab_remap
vocab_remap.R = R                # same root as everything else (only differs in the synthetic test)
from vocab_remap import remap_enabled, remap_tables
from regions import REGION_OF
assert REGION_OF.get(COUNTRY) == RG, f"{COUNTRY} is in region {REGION_OF.get(COUNTRY)}, expected {RG}"
VOCAB_REMAP = remap_enabled()
assert VOCAB_REMAP, "onset demo runs the main specification only (VOCAB_REMAP=1)"

OD = f"{R}/outputs/onset"; os.makedirs(OD, exist_ok=True)
OUT_M = f"{OD}/metrics_onset_{COUNTRY}.csv"; OUT_P = f"{OD}/preds_onset_{COUNTRY}.npz"
assert not os.path.exists(OUT_M), f"refusing to overwrite completed job {OUT_M}"
TMP_M, TMP_P = OUT_M + ".part", OUT_P + ".part.npz"


def mi(s):
    """'2022-02' -> integer month index."""
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m) - 1


def ms(i):
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


m0 = mi(M0)
print(f"[cfg] country={COUNTRY} onset={M0} region={RG} seeds={SEEDS} windows={WINDOWS} dev={DEVICE}", flush=True)

# ---- 1. cells of the target country (modal non-missing label) ----
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
cm["grid_id"] = cm["grid_id"].astype(np.int64)
cm["ym"] = cm["year_month"].astype(str).str[:7]
cnt = cm.dropna(subset=["country"]).groupby(["grid_id", "country"]).size().rename("n").reset_index()
# tie policy (Codex 7, O02): highest count wins; ties broken alphabetically by country name, independent of row order
cnt = cnt.sort_values(["grid_id", "n", "country"], ascending=[True, False, True])
lab = cnt.drop_duplicates("grid_id").set_index("grid_id")["country"]
top2 = cnt.groupby("grid_id")["n"].apply(lambda x: len(x) > 1 and x.iloc[0] == x.iloc[1])
cells = np.sort(lab.index[lab == COUNTRY].to_numpy().astype(np.int64))
n_ties = int(top2.reindex(cells).fillna(False).sum())
n_mixed = int(cm.dropna(subset=["country"]).groupby("grid_id")["country"].nunique().loc[cells].gt(1).sum())
print(f"[cells] {len(cells)} cells labelled {COUNTRY}; {n_mixed} also carry another country label; "
      f"{n_ties} decided by the alphabetical tie rule", flush=True)
assert 50 <= len(cells) <= 2000, f"implausible cell count {len(cells)} for {COUNTRY}"
row_country = cm[cm["grid_id"].isin(cells)][["grid_id", "ym", "country"]]
del cm, lab, cnt

# ---- 2. panel rows for those cells ----
cols = ["grid_id", "year_month", "event_sequence", "violence_next_month"] + BASE_COLS
d = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet", columns=cols,
                    filters=[("grid_id", "in", cells.tolist())])
d["grid_id"] = d["grid_id"].astype(np.int64)
d["ym"] = d["year_month"].astype(str).str[:7]
assert not d.duplicated(["grid_id", "ym"]).any(), "duplicate grid-months in panel"
assert set(d["grid_id"].unique()) <= set(cells.tolist())
d["mi"] = d["ym"].map(mi)
d = d.sort_values(["grid_id", "mi"]).reset_index(drop=True)   # canonical row order (Codex 7, O02)

# outcome lookup: nxt[(g, m)] = violence in month m+1 (the row's own target)
nxt = {(g, m): v for g, m, v in zip(d["grid_id"].to_numpy(), d["mi"].to_numpy(), d["violence_next_month"].to_numpy())
       if not pd.isna(v)}


def violent(g, m):
    """Violence in month m for cell g, read from the row of month m-1; None when that row is absent or unlabelled."""
    return nxt.get((g, m - 1))


g_arr, m_arr = d["grid_id"].to_numpy(), d["mi"].to_numpy()
vt_list = [violent(g, m) for g, m in zip(g_arr, m_arr)]
d["violence_t"] = np.array([np.nan if v is None else float(v) for v in vt_list])   # NaN = month t unknown
# cross-check against the builder's violence_lag1 on the NEXT row: lag1 at (g, t+1) is violence in month t
lag1 = {(g, m): v for g, m, v in zip(g_arr, m_arr, d["violence_lag1"].to_numpy())}
chk = [(vt, lag1[(g, m + 1)]) for g, m, vt in zip(g_arr, m_arr, vt_list)
       if vt is not None and (g, m + 1) in lag1 and not np.isnan(lag1[(g, m + 1)])]
assert len(chk) > 0 and all(float(a) == float(b) for a, b in chk), "month-t violence disagrees with next row's violence_lag1"
print(f"[check] month-t violence matches next-row violence_lag1 on {len(chk):,} rows", flush=True)


def quiet12(g, t):
    """1 if no violence in months t-11..t, 0 if any, None if any of those months is unknown."""
    vals = [violent(g, t - k) for k in range(12)]
    if any(v is None for v in vals): return None
    return int(sum(vals) == 0)


d = d[d["violence_next_month"].notna()].reset_index(drop=True)
d["y"] = d["violence_next_month"].astype(int)
# same row filter as the main evaluation and share grid: the row's own country label must be in the held-out region
# (rows are used for training and evaluation only if they pass; violence history above uses every row of the cell)
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
n_before = len(d)
d = d.merge(row_country, on=["grid_id", "ym"], how="left")
assert len(d) == n_before, "country merge changed row count"
d = d[d["country"].isin(REGION_COUNTRIES)].sort_values(["grid_id", "mi"]).reset_index(drop=True)
print(f"[rows] {len(d):,} of {n_before:,} cell-months kept by the region filter used in the main evaluation", flush=True)

# ---- 3. rows for training windows and the common evaluation set ----
ev_first = m0 + 2
ev_mask = d["mi"].to_numpy() >= ev_first
assert ev_mask.sum() > 0
ev_idx = np.where(ev_mask)[0]
ev_idx = ev_idx[np.lexsort((d["grid_id"].to_numpy()[ev_idx], d["mi"].to_numpy()[ev_idx]))]   # month, then cell
train_idx = {}
for w in WINDOWS:
    lo, hi = m0 - 1, m0 + w - 2                      # origins whose outcome month is in [M0, M0+w-1]
    ix = np.where((d["mi"].to_numpy() >= lo) & (d["mi"].to_numpy() <= hi))[0]
    assert hi < ev_first, "training origins overlap evaluation origins"
    if len(ix) == 0:                                  # Codex 7, O04: empty window is reported, not a crash
        print(f"[window w={w}] no training rows", flush=True); train_idx[w] = ix; continue
    assert (d["mi"].to_numpy()[ix] + 1).max() <= m0 + w - 1, "a training label falls after the window"
    train_idx[w] = ix
    print(f"[window w={w}] origins {ms(lo)}..{ms(hi)} rows={len(ix)} pos={int(d['y'].to_numpy()[ix].sum())}", flush=True)
yev = d["y"].to_numpy()[ev_idx]
pers = d["violence_t"].to_numpy()[ev_idx].astype(float)
assert not np.isnan(pers).any(), "month-t violence missing on some eval rows"
q12 = np.array([quiet12(g, m) for g, m in zip(d["grid_id"].to_numpy()[ev_idx], d["mi"].to_numpy()[ev_idx])], dtype=object)
new_mask = np.array([v == 1 for v in q12]); known = np.array([v is not None for v in q12])
print(f"[eval] origins {ms(ev_first)}..{ms(int(d['mi'].max()))} rows={len(ev_idx):,} pos={yev.mean():.4f}; "
      f"new cells rows={new_mask.sum():,} pos={int(yev[new_mask].sum())}; history unknown={int((~known).sum())}", flush=True)

# ---- 4. features ----
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
assert ps["grid_id"].is_unique
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
n0 = len(d); d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "prio merge changed row count"
FEATS = BASE_COLS + static_cols
X = d[FEATS].fillna(0.0).to_numpy(np.float32)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}

# ---- 5. encoder (copied from share_grid_conflict.py) ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}


class Encoder(nn.Module):
    def __init__(self, vsm, d_, n_heads, n_layers, fields, static_dim, dropout=0.1):
        super().__init__()
        self.fields = fields
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d_, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d_, d_); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_))
        self.prio_proj = nn.Linear(static_dim, d_)
        layer = nn.TransformerEncoderLayer(d_model=d_, nhead=n_heads, dim_feedforward=d_ * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_)

    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1) + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)


def parse_seq(s):
    if isinstance(s, str): seq = json.loads(s)
    elif isinstance(s, (list, tuple, np.ndarray)): seq = [list(ev) for ev in s]
    elif s is None or (isinstance(s, float) and np.isnan(s)): seq = []
    else: raise TypeError(f"unsupported event_sequence type {type(s)}")
    return seq[-MAX_SEQ_LEN:]


class SeqDS(Dataset):
    def __init__(self, seq_strings, gids, lut):
        self.seqs = [parse_seq(s) for s in seq_strings]; self.gids = gids; self.lut = lut

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(self.seqs[i]):
            for f in range(NUM_FIELDS): t[j, f] = self.lut[f][ev[SEL[f]]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        return t, m, st


def embed(seed, rows):
    tag = f"L_contrastive_e_loro_{RG}_s{seed}"
    ckpt = f"{R}/data_model/repr_model_{tag}.pt"
    assert os.path.exists(ckpt), f"missing encoder {ckpt}"
    lut = remap_tables(tag, FIELDS, vsm, True)
    torch.manual_seed(seed)
    enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, STATIC_DIM).to(DEVICE)
    miss, _ = enc.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
    assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
    enc.eval(); Z = []
    with torch.no_grad():
        for t, m, st in DataLoader(SeqDS(d["event_sequence"].values[rows], d["grid_id"].values[rows], lut),
                                   batch_size=BATCH, shuffle=False, num_workers=0):
            Z.append(enc.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(Z); assert Z.shape == (len(rows), EMBED_DIM) and np.isfinite(Z).all()
    return Z


def xgb_p(Xtr, ytr, Xte):
    b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                   "colsample_bytree": 0.8, "seed": XGB_SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
    return b.predict(xgb.DMatrix(Xte))


def topk_hits(p, yy):
    k = int(yy.sum())
    if k == 0: return 0
    return int(yy[np.argsort(-p, kind="stable")[:k]].sum())


rows, preds = [], {}
SUBSETS = {"all": np.ones(len(ev_idx), bool), "new": new_mask}


def emit(model, w, seed, p, n_train=0, pos_train=0):
    p = np.asarray(p, float)
    assert len(p) == len(ev_idx) and np.isfinite(p).all(), (model, w, seed)
    preds[f"{model}|{w}|{seed}"] = p.astype(np.float32)
    for sub, mk in SUBSETS.items():
        yy, pp = yev[mk], p[mk]; npos = int(yy.sum())
        ok = npos >= MIN_POS and len(yy) - npos >= MIN_POS
        rows.append(dict(country=COUNTRY, onset=M0, region=RG, model=model, window=w, enc_seed=seed, subset=sub,
                         n_train=n_train, pos_train=pos_train, n_eval=int(len(yy)), pos_eval=npos,
                         pr=float(average_precision_score(yy, pp)) if ok else np.nan,
                         roc=float(roc_auc_score(yy, pp)) if ok else np.nan,
                         brier=float(np.mean((np.clip(pp, 0, 1) - yy) ** 2)) if len(yy) else np.nan,
                         hits_at_k=topk_hits(pp, yy) if ok else np.nan))
    pd.DataFrame(rows).to_csv(TMP_M, index=False)
    r = rows[-2]
    print(f"  [{model}] w={w} seed={seed} n_train={n_train} PR(all)={r['pr']:.4f} hits={r['hits_at_k']}", flush=True)


# persistence (no training, no window, no seed)
emit("persistence_t", 0, 0, pers)

# region-trained reference forecasts from the main evaluation
gid_ev = d["grid_id"].to_numpy()[ev_idx]; ym_ev = d["ym"].to_numpy()[ev_idx]
for seed in SEEDS:
    z = np.load(f"{R}/outputs/preds_region_L_contrastive_e_loro_{RG}_s{seed}.npz", allow_pickle=True)
    key = pd.Series(np.arange(len(z["y"])), index=pd.MultiIndex.from_arrays(
        [z["grid_id"].astype(np.int64), np.asarray(z["year_month"]).astype(str)]))
    pos = key.reindex(pd.MultiIndex.from_arrays([gid_ev, ym_ev])).to_numpy()
    cover = ~np.isnan(pos)
    assert cover.all(), (f"[region ref s{seed}] {int((~cover).sum())} eval rows missing from the main-evaluation "
                         "forecasts; the onset cohort should be a subset of the evaluated rows")
    pos = pos.astype(int)
    assert (np.asarray(z["y"]).astype(int)[pos] == yev).all(), "main-evaluation labels disagree with this panel"
    emit("region_base", 0, seed, z["base"][pos]); emit("region_emb_base", 0, seed, z["emb_base"][pos])

# window-trained arms
need = np.unique(np.concatenate([ev_idx] + list(train_idx.values())))
posn = -np.ones(len(d), np.int64); posn[need] = np.arange(len(need))
base_cache = {}
for seed in SEEDS:
    t0 = time.time(); Z = embed(seed, need)
    print(f"[emb s{seed}] {Z.shape} in {time.time() - t0:.0f}s", flush=True)
    Zev = Z[posn[ev_idx]]
    for w in WINDOWS:
        ix = train_idx[w]; ytr = d["y"].to_numpy()[ix]; npos = int(ytr.sum())
        if npos < MIN_POS or len(ix) - npos < MIN_POS:
            print(f"  [skip] w={w}: {npos} positives / {len(ix) - npos} negatives in training rows", flush=True); continue
        if w not in base_cache: base_cache[w] = xgb_p(X[ix], ytr, X[ev_idx])
        emit("base", w, seed, base_cache[w], len(ix), npos)
        emit("emb_base", w, seed, xgb_p(np.concatenate([Z[posn[ix]], X[ix]], 1), ytr, np.concatenate([Zev, X[ev_idx]], 1)),
             len(ix), npos)

# ---- 6. publish ----
np.savez_compressed(TMP_P, __grid_id=gid_ev, __year_month=ym_ev.astype("U7"), __y=yev.astype(np.int8),
                    __new=new_mask, __cells=cells, __n_mixed=np.int64(n_mixed), __n_ties=np.int64(n_ties), **preds)
chk = np.load(TMP_P); assert all(k in chk.files for k in preds)
os.replace(TMP_P, OUT_P); os.replace(TMP_M, OUT_M)
m = pd.DataFrame(rows)
summ = m[m.model.isin(["base", "emb_base"])].pivot_table(index=["window", "subset"], columns="model",
                                                          values=["pr", "hits_at_k"], aggfunc="mean")
print("\n[summary] means over encoder seeds\n", summ.round(4).to_string(), flush=True)
print(f"[done] {OUT_M} ({len(rows)} rows), {OUT_P}", flush=True)

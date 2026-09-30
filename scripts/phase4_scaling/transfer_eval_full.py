"""transfer_eval_full.py — three-assessment transfer eval for ONE grid encoder.
=============================================================================
Env EVAL_TAG = M_{obj}_{feat}_s{seed}  -> loads data_model/repr_model_{EVAL_TAG}.pt
(FEATSET/PRIO-aware; strict=False so only the encoder body+embeddings load).

For each of the 26 held-out countries, score the FROZEN embedding on:
  Mode A  = NON-TEMPORAL spatial transfer (time-pooled 5-fold CV)
  Mode B  = TEMPORAL forecasting (train <=2020-12, test >=2022-01)
across 4 targets:
  incidence  (violence_next_month, PR-AUC)         + onset (lag1==0) + NON-CONTIGUOUS (lag1==0 & spatial_lag==0) hard cut
  fatalities (any fatal next month, PR-AUC)
  rare_event (best-F1 on incidence)
  severity   (next-month fatality count, CRPS)
and 4 model variants per cell:
  persistence | base (spatial-lag + PRIO) | emb (embedding only) | emb+base
Plus a Mode-B PURE-FORECAST REFERENCE on a sample of TRAINING-region cells.

Output: outputs/eval_full_{EVAL_TAG}.csv  (long format: tag,unit,mode,target,model,score,n)
Run one per checkpoint via the array launcher.
"""
import os, json, time, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import KFold
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
TAG = os.environ["EVAL_TAG"]
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
OUT = f"{R}/outputs/eval_full_{TAG}.csv"
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
CAP = TAG.split("_")[0]   # capacity prefix from tag, e.g. XL_contrastive_a_s123 -> XL
assert CAP in CAPACITY, f"unrecognized capacity prefix in EVAL_TAG={TAG} (got {CAP})"
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[CAP]
MAX_SEQ_LEN, BATCH = 200, 512
TRAIN_END, TEST_START = "2020-12", "2022-01"
SEED = int(re.search(r"_s(\d+)", TAG).group(1)) if re.search(r"_s(\d+)", TAG) else 123
OBJ = TAG.split("_")[1]
FEAT = TAG.split("_")[2]
SUPER_FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type",
                "actor2_type", "month_offset", "fatalities", "deaths_civilians"]
FEATSET_SEL = {"a": [0,1,2,3,4,5,6], "b": [0,1,2,3,4,5,6,7], "c": [0,1,2,3,4,5,6,8],
               "e": [0,1,2,3,4,5,6], "f": [0,1,2,3,4,5,6,7,8]}
FEATSET_PRIO = {"a": False, "b": False, "c": False, "e": True, "f": True}
SEL = FEATSET_SEL[FEAT]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
USE_PRIO = FEATSET_PRIO[FEAT]
HOLDOUT_COUNTRIES = ["Sudan","Somalia","Uganda","Kenya","Ethiopia","South Africa","Iraq",
    "Tunisia","Iran","Algeria","Morocco","Pakistan","Bangladesh","Nepal","Sri Lanka",
    "Myanmar","Philippines","Thailand","South Korea","Ukraine","France","Germany",
    "Mexico","Venezuela","Colombia","Chile"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
print(f"[cfg] TAG={TAG} OBJ={OBJ} FEAT={FEAT} fields={FIELDS} prio={USE_PRIO} dev={DEVICE}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}  # +1 mask slot (matches trainer)

prio_lookup, STATIC_DIM = None, 1
if USE_PRIO:
    ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
    static_cols = [c for c in ps.columns if c != "grid_id"]
    STATIC_DIM = len(static_cols)
    prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32)
                   for r in ps.itertuples(index=False)}


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, use_prio, static_dim, dropout=0.1):
        super().__init__()
        self.fields, self.use_prio = fields, use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio:
            self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio:
            cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        out = self.transformer(e, src_key_padding_mask=~mwc)
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)


class SeqDS(Dataset):
    def __init__(self, seq_strings, gids):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]
        self.gids = gids
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[j, f] = ev[SEL[f]]
            m[j] = True
        if USE_PRIO:
            st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        else:
            st = torch.zeros(1, dtype=torch.float32)
        return t, m, st


def extract(model, seqs, gids):
    ld = DataLoader(SeqDS(seqs, gids), batch_size=BATCH, shuffle=False, num_workers=0)
    Z = []
    with torch.no_grad():
        for t, m, st in ld:
            Z.append(model.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, 0)


def crps_ens(draws, y):
    y = y[:, None]
    return (np.abs(draws - y).mean(1) - 0.5 * np.abs(draws[:, :, None] - draws[:, None, :]).mean((1, 2)))


def xgb_pr(Xtr, ytr, Xte, yte, objective="binary:logistic"):
    if ytr.sum() < 3 or len(np.unique(yte)) < 2:
        return None, None
    clf = xgb.train({"objective": objective, "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                     "colsample_bytree": 0.8, "seed": SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
    p = clf.predict(xgb.DMatrix(Xte))
    return p, average_precision_score(yte, p)


def best_f1(y, p):
    if p is None or len(np.unique(y)) < 2:
        return np.nan
    ths = np.quantile(p, np.linspace(0.5, 0.99, 20))
    return max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in ths)


# ---- load data ----
base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
cc = [f"country_{c}" for c in HOLDOUT_COUNTRIES]
need = ["grid_id", "year_month", "event_sequence", "violence_next_month", "etype_violence_against_civilians"] + base_cols + cc
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=[c for c in need if c])
tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")[
    ["grid_id", "year_month", "sb_fatalities", "sb_fat_next"]]
panel = panel.merge(tgt, on=["grid_id", "year_month"], how="left")
panel["fat_now"] = panel["sb_fatalities"].fillna(0.0)
panel["fat_next"] = panel["sb_fat_next"].fillna(0.0)
panel = panel.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
panel["civ_now"] = panel["etype_violence_against_civilians"].fillna(0.0)
panel["civ_next"] = panel.groupby("grid_id")["civ_now"].shift(-1).fillna(0.0)
if USE_PRIO:
    pst = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
    panel = panel.merge(pst, on="grid_id", how="left")
    prio_cols = [c for c in pst.columns if c != "grid_id"]
    panel[prio_cols] = panel[prio_cols].fillna(0.0)
else:
    prio_cols = []
BASE_FEATS = base_cols + prio_cols
panel[BASE_FEATS] = panel[BASE_FEATS].fillna(0.0)

enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
missing, unexpected = enc.load_state_dict(sd, strict=False)
enc.eval()
print(f"[model] loaded {CKPT}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)

rows = []


def score_block(unit, mode, d, tr, te, Z):
    """d=subframe, tr/te boolean masks into d, Z=embeddings for d."""
    y_inc = d["violence_next_month"].to_numpy().astype(int)
    lag1 = np.nan_to_num(d["violence_lag1"].to_numpy())
    slag = np.nan_to_num(d["spatial_lag_violence"].to_numpy())
    fnext = d["fat_next"].to_numpy().astype(float)
    fnow = d["fat_now"].to_numpy().astype(float)
    cnow = d["civ_now"].to_numpy().astype(float)
    cnext = d["civ_next"].to_numpy().astype(float)
    B = d[BASE_FEATS].to_numpy()
    EB = np.concatenate([Z, B], axis=1)
    def add(target, model, score, n):
        rows.append(dict(tag=TAG, obj=OBJ, feat=FEAT, unit=unit, mode=mode,
                         target=target, model=model, score=score, n=int(n)))
    # ---- incidence + onset + non-contiguous (PR-AUC) ----
    for tname, sub in [("incidence", np.ones(len(d), bool)),
                       ("onset", lag1 == 0),
                       ("noncontig_onset", (lag1 == 0) & (slag == 0))]:
        trm, tem = tr & sub, te & sub
        if tem.sum() < 30 or y_inc[tem].sum() < 5:
            continue
        add(tname, "persistence", average_precision_score(y_inc[tem], lag1[tem]) if len(np.unique(y_inc[tem])) > 1 else np.nan, tem.sum())
        pb, s = xgb_pr(B[trm], y_inc[trm], B[tem], y_inc[tem]); add(tname, "base", s, tem.sum())
        pe, s = xgb_pr(Z[trm], y_inc[trm], Z[tem], y_inc[tem]); add(tname, "emb", s, tem.sum())
        peb, s = xgb_pr(EB[trm], y_inc[trm], EB[tem], y_inc[tem]); add(tname, "emb_base", s, tem.sum())
        if tname == "incidence":
            add("rare_event_f1", "persistence", best_f1(y_inc[tem], lag1[tem]), tem.sum())
            add("rare_event_f1", "base", best_f1(y_inc[tem], pb), tem.sum())
            add("rare_event_f1", "emb", best_f1(y_inc[tem], pe), tem.sum())
            add("rare_event_f1", "emb_base", best_f1(y_inc[tem], peb), tem.sum())
    # ---- fatalities (any fatal next month, PR-AUC) ----
    yf = (fnext > 0).astype(int)
    if te.sum() >= 30 and yf[te].sum() >= 5 and yf[tr].sum() >= 3:
        add("fatalities", "persistence", average_precision_score(yf[te], fnow[te]) if len(np.unique(yf[te])) > 1 else np.nan, te.sum())
        _, s = xgb_pr(B[tr], yf[tr], B[te], yf[te]); add("fatalities", "base", s, te.sum())
        _, s = xgb_pr(Z[tr], yf[tr], Z[te], yf[te]); add("fatalities", "emb", s, te.sum())
        _, s = xgb_pr(EB[tr], yf[tr], EB[te], yf[te]); add("fatalities", "emb_base", s, te.sum())
    # ---- civilian-targeted violence (any violence-against-civilians event next month, PR-AUC) ----
    yc = (cnext > 0).astype(int)
    if te.sum() >= 30 and yc[te].sum() >= 5 and yc[tr].sum() >= 3:
        add("civilian", "persistence", average_precision_score(yc[te], cnow[te]) if len(np.unique(yc[te])) > 1 else np.nan, te.sum())
        _, s = xgb_pr(B[tr], yc[tr], B[te], yc[te]); add("civilian", "base", s, te.sum())
        _, s = xgb_pr(Z[tr], yc[tr], Z[te], yc[te]); add("civilian", "emb", s, te.sum())
        _, s = xgb_pr(EB[tr], yc[tr], EB[te], yc[te]); add("civilian", "emb_base", s, te.sum())
    # ---- severity (CRPS, lower better) ----
    if te.sum() >= 30:
        def crps_for(X):
            reg = xgb.train({"objective": "count:poisson", "max_depth": 6, "eta": 0.1,
                             "subsample": 0.8, "seed": SEED}, xgb.DMatrix(X[tr], label=fnext[tr]), 200)
            lam = np.clip(reg.predict(xgb.DMatrix(X[te])), 1e-6, None)
            return crps_ens(rng.poisson(lam[:, None], size=(len(lam), 100)).astype(float), fnext[te]).mean()
        add("severity_crps", "persistence", np.abs(fnow[te] - fnext[te]).mean(), te.sum())
        add("severity_crps", "base", crps_for(B), te.sum())
        add("severity_crps", "emb", crps_for(Z), te.sum())
        add("severity_crps", "emb_base", crps_for(EB), te.sum())


for uname in HOLDOUT_COUNTRIES:
    col = f"country_{uname}"
    if col not in panel.columns:
        print(f"[WARN] {col} missing", flush=True); continue
    d = panel[(panel[col] == 1) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
    if len(d) < 200:
        print(f"[unit {uname}] SKIP n={len(d)}", flush=True); continue
    t0 = time.time()
    Z = extract(enc, d["event_sequence"].values, d["grid_id"].values)
    ym = d["year_month"].to_numpy()
    # Mode B: temporal
    trb = ym <= TRAIN_END; teb = ym >= TEST_START
    if trb.sum() >= 100 and teb.sum() >= 30:
        score_block(uname.replace(" ", "_"), "B_temporal", d, trb, teb, Z)
    # Mode A: time-pooled 5-fold CV (aggregate folds by re-scoring each as test)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    idx = np.arange(len(d))
    for k, (trk, tek) in enumerate(kf.split(idx)):
        trm = np.zeros(len(d), bool); trm[trk] = True
        tem = np.zeros(len(d), bool); tem[tek] = True
        score_block(uname.replace(" ", "_"), f"A_pooled_f{k}", d, trm, tem, Z)
    print(f"[unit {uname}] done n={len(d)} {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)

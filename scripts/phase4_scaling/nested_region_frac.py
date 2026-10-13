"""transfer_eval_region_ft.py -- LORO low-data fine-tune curve (replaces Table 2's June design).
For held-out region RG: load repr_model_L_contrastive_e_loro_{RG}_s123.pt (pretrained on the
other five regions), fine-tune encoder+head on frac in {1,5,10,25,100}% of RG's TRAIN-period
labels vs an identical from-scratch model; score PR-AUC (and ROC) on RG's TEST period (Mode B).
Mirrors transfer_eval.py Mechanism B exactly (stratified frac subsample, FT_MAX_ROWS cap,
6 epochs, lr 1e-4 pretrained / 5e-4 scratch, BCE pos_weight, grad clip 1.0) but with the
Gen-4 Encoder (featset e, PRIO static covariates) and the LORO split from transfer_eval_region.py.
Env: EVAL_REGION (required), EVAL_TAG (default L_contrastive_e_loro_{RG}_s123), EVAL_FRACS,
EVAL_FT_EPOCHS, EVAL_FT_MAX_ROWS, EVAL_SEED.
Out: outputs/ft_region_{TAG}.csv
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
RG = os.environ["EVAL_REGION"]
SEED = int(os.environ.get("EVAL_SEED", "123"))
TAG = os.environ.get("EVAL_TAG", f"L_contrastive_e_loro_{RG}_s{SEED}")
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
OUT = f"{R}/outputs/ft_region_{TAG}.csv"
FRACS = [float(x) for x in os.environ.get("EVAL_FRACS", "0.01,0.05,0.10,0.25,1.0").split(",")]
FT_EPOCHS = int(os.environ.get("EVAL_FT_EPOCHS", "6"))
FT_MAX_ROWS = int(os.environ.get("EVAL_FT_MAX_ROWS", "150000"))
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[TAG.split("_")[0]]
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
TRAIN_END, TEST_START = "2020-12", "2022-01"
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL); USE_PRIO = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF, REGIONS
assert RG in REGIONS, f"bad EVAL_REGION {RG}"
assert os.path.exists(CKPT), f"missing checkpoint {CKPT}"
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
print(f"[cfg] RG={RG} TAG={TAG} fracs={FRACS} ep={FT_EPOCHS} cap={FT_MAX_ROWS} dev={DEVICE}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}

class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, use_prio, static_dim, dropout=0.1):
        super().__init__()
        self.fields, self.use_prio = fields, use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio: self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio: cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

# ======== nested GBM at matched fractions ========
# Everything above this line is transfer_eval_region_ft.py verbatim (config, vocab, PRIO lookup, Encoder).
# Question: does adding the FROZEN LORO embedding to the hand-crafted GBM improve it, and does the gain
# grow as target-region training data shrink? Same rows as ft_region_* and gbm_region_frac_* (same rng calls).
# Feature order [Z, B] and XGBoost settings copied from transfer_eval_region.py (Table 3).
import xgboost as xgb
OUT = f"{R}/outputs/nested_region_frac_{RG}_s{SEED}.csv"
REF = f"{R}/outputs/gbm_region_frac_{RG}_s{SEED}.csv"
assert os.path.exists(REF), f"missing reference {REF}"

class SeqDS(Dataset):
    def __init__(self, seq_strings, gids):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]
        self.gids = gids
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS): t[j, f] = ev[SEL[f]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        return t, m, st

def extract(model, seqs, gids):
    Z = []
    with torch.no_grad():
        for t, m, st in DataLoader(SeqDS(seqs, gids), batch_size=BATCH, shuffle=False, num_workers=0):
            Z.append(model.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, 0)

base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "event_sequence", "violence_next_month"] + base_cols)
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
d = panel[panel["country"].isin(REGION_COUNTRIES) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
del panel, cm
n0 = len(d)
assert ps["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "prio merge changed row count"
FEATS = base_cols + static_cols
d[FEATS] = d[FEATS].fillna(0.0)
X = d[FEATS].to_numpy(np.float32)
ym = d["year_month"].to_numpy(); trm = ym <= TRAIN_END; tem = ym >= TEST_START
y = d["violence_next_month"].to_numpy().astype(int); yte = y[tem]
seqs = d["event_sequence"].values; gids = d["grid_id"].values
print(f"[region {RG}] rows={len(d):,} train={trm.sum():,} test={tem.sum():,} test_pos={yte.mean():.4f} feats={len(FEATS)}", flush=True)

ref = pd.read_csv(REF)
assert int(ref["n_train"].iloc[0]) == int(trm.sum()) and int(ref["n_test"].iloc[0]) == int(tem.sum()), "panel differs from GBM reference"
persist = np.nan_to_num(d["violence_lag1"].to_numpy())[tem]
assert abs(average_precision_score(yte, persist) - float(ref.loc[ref["model"] == "persistence", "pr"].iloc[0])) < 1e-9, "persistence PR mismatch"
print("[check] panel matches GBM reference", flush=True)

# ---- identical subsample draws ----
tr_idx = np.where(trm)[0]; pos_idx = tr_idx[y[tr_idx] == 1]; neg_idx = tr_idx[y[tr_idx] == 0]
draws = {}
for frac in FRACS:
    npos = int(round(len(pos_idx) * frac)); nneg = int(round(len(neg_idx) * frac))
    if npos < 5: continue
    idx = np.concatenate([rng.choice(pos_idx, size=npos, replace=False), rng.choice(neg_idx, size=nneg, replace=False)])
    if len(idx) > FT_MAX_ROWS: idx = rng.choice(idx, size=FT_MAX_ROWS, replace=False)
    rng.shuffle(idx)
    n_ref = int(ref.loc[(ref["model"] == "gbm_base") & (np.isclose(ref["frac"], frac)), "n_ft"].iloc[0])
    assert len(idx) == n_ref, f"frac {frac}: n {len(idx)} vs reference {n_ref}"
    draws[frac] = idx

# ---- frozen embeddings for only the rows we need ----
te_idx = np.where(tem)[0]
need = np.unique(np.concatenate(list(draws.values()) + [te_idx]))
enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
miss, unexp = enc.load_state_dict(sd, strict=False)
assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
enc.eval()
t0 = time.time(); Zneed = extract(enc, seqs[need], gids[need])
assert Zneed.shape == (len(need), EMBED_DIM) and np.isfinite(Zneed).all(), Zneed.shape
print(f"[emb] {Zneed.shape} in {time.time()-t0:.0f}s; unexpected ckpt keys={len(unexp)}", flush=True)
pos = -np.ones(len(d), dtype=np.int64); pos[need] = np.arange(len(need))
def Zof(ix):
    assert (pos[ix] >= 0).all(); return Zneed[pos[ix]]

def xgb_pr(Xtr, ytr, Xte):
    clf = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                     "colsample_bytree": 0.8, "seed": SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
    p = clf.predict(xgb.DMatrix(Xte))
    return roc_auc_score(yte, p), average_precision_score(yte, p)

rows = []
base = dict(tag=TAG, region=RG, seed=SEED, n_train=int(trm.sum()), n_test=int(tem.sum()), test_pos=float(yte.mean()))
Zte, Bte = Zof(te_idx), X[te_idx]
for frac, idx in draws.items():
    Ztr, Btr, ytr = Zof(idx), X[idx], y[idx]
    res = {}
    for name, A, At in [("base", Btr, Bte), ("emb", Ztr, Zte),
                        ("emb_base", np.concatenate([Ztr, Btr], axis=1), np.concatenate([Zte, Bte], axis=1))]:
        roc, pr = xgb_pr(A, ytr, At); res[name] = pr
        rows.append(dict(base, model=name, frac=frac, n_ft=int(len(idx)), roc=roc, pr=pr))
    ref_pr = float(ref.loc[(ref["model"] == "gbm_base") & (np.isclose(ref["frac"], frac)), "pr"].iloc[0])
    assert abs(res["base"] - ref_pr) < 5e-3, f"base arm does not reproduce gbm_region_frac: {res['base']} vs {ref_pr}"
    print(f"  frac={frac} n={len(idx)} PR base {res['base']:.4f} emb {res['emb']:.4f} emb_base {res['emb_base']:.4f} d(emb_base-base) {res['emb_base']-res['base']:+.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)

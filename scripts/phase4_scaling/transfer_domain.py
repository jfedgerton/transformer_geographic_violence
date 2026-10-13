"""transfer_domain.py -- appendix crime / dengue analogue of Table 2 (seed 123 first).

DOMAIN=crime|dengue, CITY=<index 0-4 into HELDOUT>. For one held-out city:
  target train pool = city rows with t <  TEST_START      (its own early months)
  target test       = city rows with t >= TEST_START      (last 24 months with a target)
  shares            = 1, 5, 10, 25 percent of the pool, and the full pool capped at 150,000 rows
Arms (same definitions as the conflict tables):
  persistence   lag1 on test rows
  base          XGBoost on baseline features
  emb_base      XGBoost on baseline features + frozen 256-d embedding from pretrain_domain.py
  ft_pretrained encoder initialised from pretrain_domain.py, fine-tuned 6 epochs, lr 1e-4
  ft_scratch    same network from random values, same rows, epochs and lr
XGBoost settings fixed: depth 6, eta 0.1, subsample 0.8, colsample 0.8, 200 rounds.
Output: outputs/domain_{DOMAIN}_transfer_s{SEED}.csv (appends one row per city x share x arm).
"""
import os, json, time, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
E = os.environ.get
DOMAIN = E("DOMAIN", "dengue"); SEED = int(E("SEED", "123")); SMOKE = E("SMOKE", "0") == "1"
CITY_I = int(E("CITY", E("SLURM_ARRAY_TASK_ID", "0")))
assert DOMAIN in ("crime", "dengue")
if DOMAIN == "crime":
    PANEL = f"{R}/data_intermediate/crime_grid150m_violent_wide.parquet"
    VOCAB = f"{R}/data_intermediate/crime_vocab_grid150m_violent_wide.json"
    FIELDS = ["offense_type", "offense_group", "offense_against"]
    YCOL = "crime_next_month"
    HELDOUT = ["Boston", "St Louis", "Seattle", "Austin", "Los Angeles"]
else:
    PANEL = f"{R}/data_intermediate/dengue_br_muni_v1.parquet"
    VOCAB = f"{R}/data_intermediate/dengue_vocab_br_muni_v1.json"
    FIELDS = ["severity"]
    YCOL = "dengue_next_period"
    HELDOUT = None                         # dengue: one IBGE macro-region per job (first code digit)
REGION = [1, 2, 3, 4, 5][CITY_I] if DOMAIN == "dengue" else None
TAG = f"_r{REGION}" if DOMAIN == "dengue" else ""   # 1=N 2=NE 3=SE 4=S 5=CO
CITY = f"region{REGION}" if DOMAIN == "dengue" else HELDOUT[CITY_I]
FEAT = ["lag1", "lag3", "lag6", "roll3", "months_since_last", "spatial_lag"]
D, NL, NH, DROP = 256, 6, 8, 0.1
MAXLEN = 200; CAP = 150000; TEST_CAP = 200000
FT_EPOCHS = 6; FT_LR = 1e-4; BATCH = 256
SHARES = [0.01, 0.05, 0.10, 0.25, 1.0]
CKPT = f"{R}/data_model/repr_model_L_contrastive_{DOMAIN}_dom{TAG}_s{SEED}.pt"
OUT = E("OUT", f"{R}/outputs/domain_{DOMAIN}_transfer_s{SEED}.csv")

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] DOMAIN={DOMAIN} CITY={CITY} SEED={SEED} SMOKE={SMOKE} dev={DEV}", flush=True)
assert os.path.exists(CKPT), f"missing pretrained encoder {CKPT}"

voc = json.load(open(VOCAB))
VS = [len(voc[f]) + 1 for f in FIELDS] + [201]
NF = len(VS)

# ---- data: TEST_START uses the same rule as pretrain_domain.py (whole panel) ----
tcol = pd.read_parquet(PANEL, columns=["t", YCOL])
tmax = int(tcol.loc[tcol[YCOL].notna(), "t"].max()); del tcol
TEST_START = tmax - (11 if DOMAIN == "crime" else 23)   # crime panel is only 35 months
df = pd.read_parquet(PANEL, columns=["city", "t", YCOL, "event_sequence"] + FEAT)
if DOMAIN == "dengue":
    df = df[(df["city"].astype("int64") // 1000000) == REGION].reset_index(drop=True)
else:
    df = df[df["city"] == CITY].reset_index(drop=True)
df = df[df[YCOL].notna()].reset_index(drop=True)
assert len(df) > 0, f"no rows for city {CITY}"
pool = df[df["t"] < TEST_START].reset_index(drop=True)
test = df[df["t"] >= TEST_START].reset_index(drop=True)
del df
if len(test) > TEST_CAP: test = test.sample(n=TEST_CAP, random_state=SEED).reset_index(drop=True)
if SMOKE:
    pool = pool.sample(n=min(20000, len(pool)), random_state=SEED).reset_index(drop=True)
    test = test.sample(n=min(5000, len(test)), random_state=SEED).reset_index(drop=True)
    SHARES = [0.05, 1.0]; FT_EPOCHS = 1
assert pool["t"].max() < TEST_START <= test["t"].min()
yte = test[YCOL].values.astype(int)
print(f"[data] tmax={tmax} TEST_START={TEST_START} pool={len(pool):,} test={len(test):,} "
      f"pos_pool={pool[YCOL].mean():.3f} pos_test={yte.mean():.3f}", flush=True)
assert 0 < yte.sum() < len(yte), "test set has one class only"

def write_row(d):
    d.update(domain=DOMAIN, city=str(CITY), seed=SEED, test_start=TEST_START, n_test=len(yte), pos_test=round(float(yte.mean()), 4))
    pd.DataFrame([d]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"[RESULT] {d}", flush=True)

def parse(s):
    if isinstance(s, str): return json.loads(s)[-MAXLEN:]
    return [list(x) for x in s][-MAXLEN:] if s is not None else []

class DS(Dataset):
    def __init__(self, d):
        self.seq = [parse(s) for s in d["event_sequence"].values]
        self.y = d[YCOL].values.astype(np.float32)
        assert all(len(ev) == NF for sq in self.seq[:2000] for ev in sq), f"event length != {NF}"
    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        t = torch.zeros(MAXLEN, NF, dtype=torch.long); m = torch.zeros(MAXLEN, dtype=torch.bool)
        for j, ev in enumerate(self.seq[i]):
            for k in range(NF - 1): t[j, k] = int(ev[k])
            t[j, NF - 1] = min(int(ev[NF - 1]), 199) + 1
            m[j] = True
        return t, m, torch.tensor(self.y[i])
def coll(b): return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), torch.stack([x[2] for x in b])

class Enc(nn.Module):   # same layout as pretrain_domain.py
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, D, padding_idx=0) for v in VS])
        self.field_proj = nn.Linear(NF * D, D); self.pd = nn.Dropout(DROP)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D)); nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(D, NH, D * 4, DROP, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
    def cls_vec(self, seq, mask):
        e = self.pd(self.field_proj(torch.cat([self.embs[i](seq[:, :, i]) for i in range(NF)], -1)))
        B = seq.size(0); cls = self.cls_token.expand(B, -1, -1)
        e = torch.cat([cls, e], 1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        return self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
    def forward(self, seq, mask): return self.head(self.cls_vec(seq, mask)).squeeze(-1)

def load_pretrained(m):
    sd = torch.load(CKPT, map_location="cpu")
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert set(miss) == {"head.weight", "head.bias"} and not unexp, f"load mismatch miss={miss} unexp={unexp}"
    return m

@torch.no_grad()
def embed(m, d):
    m.eval(); out = []
    for seq, mask, _ in DataLoader(DS(d), batch_size=1024, collate_fn=coll, num_workers=2):
        out.append(m.cls_vec(seq.to(DEV), mask.to(DEV)).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def predict(m, d):
    m.eval(); out = []
    for seq, mask, _ in DataLoader(DS(d), batch_size=1024, collate_fn=coll, num_workers=2):
        out.append(torch.sigmoid(m(seq.to(DEV), mask.to(DEV))).cpu().numpy())
    return np.concatenate(out)

def fine_tune(d, pretrained):
    torch.manual_seed(SEED)
    m = Enc().to(DEV)
    if pretrained: load_pretrained(m)
    pos = float(d[YCOL].mean()); pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR)
    dl = DataLoader(DS(d), batch_size=BATCH, shuffle=True, collate_fn=coll, num_workers=2)
    for ep in range(1, FT_EPOCHS + 1):
        m.train(); tot = 0.0; nb = 0
        for seq, mask, y in dl:
            loss = lossf(m(seq.to(DEV), mask.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        assert math.isfinite(tot)
        print(f"  [ft {'pre' if pretrained else 'scr'} ep{ep}] loss={tot/max(nb,1):.4f}", flush=True)
    return predict(m, test)

def xgb_ap(Xtr, ytr, Xte):
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="aucpr", n_jobs=4, random_state=SEED)
    clf.fit(Xtr, ytr)
    return average_precision_score(yte, clf.predict_proba(Xte)[:, 1])

# ---- persistence (share-independent) ----
write_row(dict(share="all", arm="persistence", n_train=0, pos_train=float("nan"),
               test_ap=round(average_precision_score(yte, test["lag1"].values.astype(float)), 4)))

# ---- frozen embeddings computed once from the pretrained encoder ----
frozen = load_pretrained(Enc().to(DEV))
Z_pool = embed(frozen, pool); Z_test = embed(frozen, test)
assert Z_pool.shape == (len(pool), D) and np.isfinite(Z_pool).all() and np.isfinite(Z_test).all()
del frozen
Xte_b = test[FEAT].values.astype(float); Xte_e = np.hstack([Xte_b, Z_test])

for sh in SHARES:
    n = min(CAP, len(pool)) if sh == 1.0 else max(1, int(round(sh * len(pool))))
    idx = np.sort(np.random.RandomState(SEED).choice(len(pool), size=n, replace=False))
    d = pool.iloc[idx].reset_index(drop=True); ytr = d[YCOL].values.astype(int)
    lab = "cap" if sh == 1.0 else f"{sh:g}"
    base = dict(share=lab, n_train=len(d), pos_train=round(float(ytr.mean()), 4))
    if ytr.sum() < 5 or ytr.sum() == len(ytr):
        print(f"[skip] share={lab}: {ytr.sum()} positives in {len(d)} rows", flush=True); continue
    t0 = time.time()
    Xb = d[FEAT].values.astype(float); Xe = np.hstack([Xb, Z_pool[idx]])
    write_row(dict(base, arm="base", test_ap=round(xgb_ap(Xb, ytr, Xte_b), 4)))
    write_row(dict(base, arm="emb_base", test_ap=round(xgb_ap(Xe, ytr, Xte_e), 4)))
    write_row(dict(base, arm="ft_pretrained", test_ap=round(average_precision_score(yte, fine_tune(d, True)), 4)))
    write_row(dict(base, arm="ft_scratch", test_ap=round(average_precision_score(yte, fine_tune(d, False)), 4)))
    print(f"[share {lab}] done in {time.time()-t0:.0f}s", flush=True)
print("[done]", flush=True)

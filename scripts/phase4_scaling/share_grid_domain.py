"""share_grid_domain.py -- crime / dengue appendix analogue of share_grid_conflict.py (same design).

One job = DOMAIN x held-out unit (CITY index 0-4) x encoder seed SEED x ARM:
  ARM=nested    emb_base vs base (XGBoost, fixed settings, seed 123), all shares x draws
  ARM=finetune  ft_pretrained vs ft_scratch, ONE share, all draws, lr x wd sweep, early stopping on val
Split per held-out unit (VALCUT and TEST_START use the same rule as pretrain_domain.py):
  train pool  t <  VALCUT        val pool  VALCUT <= t < TEST_START        test  t >= TEST_START
Shares, draws, sweep, stopping, val floor and outputs exactly as in share_grid_conflict.py.
Encoder: data_model/repr_model_L_contrastive_{DOMAIN}_dom{TAG}_s{SEED}.pt from pretrain_domain.py.
Outputs: outputs/share_grid/{metrics,preds,sweep}_{DOMAIN}_{ARM}_{unit}_s{SEED}[_f{share}].{csv,npz}
"""
import os, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
E = os.environ.get
DOMAIN = E("DOMAIN", "crime"); SEED = int(E("SEED", "123")); ARM = E("GRID_ARM", "nested")
SMOKE = E("SMOKE", "0") == "1"
CITY_I = int(E("CITY", "0"))
assert DOMAIN in ("crime", "dengue") and ARM in ("nested", "finetune") and 0 <= CITY_I <= 4
if DOMAIN == "crime":
    PANEL = f"{R}/data_intermediate/crime_grid150m_violent_wide.parquet"
    VOCAB = f"{R}/data_intermediate/crime_vocab_grid150m_violent_wide.json"
    FIELDS = ["offense_type", "offense_group", "offense_against"]; YCOL = "crime_next_month"
    HELDOUT = ["Boston", "St Louis", "Seattle", "Austin", "Los Angeles"]
    UNIT = HELDOUT[CITY_I]; TAG = ""
else:
    PANEL = f"{R}/data_intermediate/dengue_br_muni_v1.parquet"
    VOCAB = f"{R}/data_intermediate/dengue_vocab_br_muni_v1.json"
    FIELDS = ["severity"]; YCOL = "dengue_next_period"
    REGION = [1, 2, 3, 4, 5][CITY_I]; UNIT = f"region{REGION}"; TAG = f"_r{REGION}"
FEAT = ["lag1", "lag3", "lag6", "roll3", "months_since_last", "spatial_lag"]
D, NL, NH, DROP = 256, 6, 8, 0.1
MAXLEN = 200; BATCH = 1024; FT_BATCH = 256
CKPT = f"{R}/data_model/repr_model_L_contrastive_{DOMAIN}_dom{TAG}_s{SEED}.pt"
SHARES = [float(x) for x in E("GRID_SHARES", "0.0025,0.005,0.0075,0.01,0.025,0.05,0.10").split(",")]
DRAWS = [int(x) for x in E("GRID_DRAWS", "0,1,2,3,4").split(",")]
LRS = [float(x) for x in E("GRID_LRS", "3e-5,1e-4,3e-4,1e-3").split(",")]
WDS = [float(x) for x in E("GRID_WDS", "0,0.01").split(",")]
SWEEP = E("GRID_SWEEP", "draw0")
MAX_EP = int(E("GRID_MAX_EP", "50")); PATIENCE = int(E("GRID_PATIENCE", "5")); MIN_DELTA = 1e-4
VAL_MIN_POS = 10; MIN_TRAIN_POS = 5; DRAW_BASE = 123; XGB_SEED = 123; TEST_CAP = 200000
assert SWEEP in ("draw0", "all")
if ARM == "finetune": assert len(SHARES) == 1, "finetune jobs take exactly one share"
if SMOKE:
    SHARES = SHARES[:1] if ARM == "finetune" else [0.01, 0.05]
    DRAWS = [0, 1]; LRS = [1e-4, 1e-3]; WDS = [0.0]; MAX_EP = 3; PATIENCE = 2
SUF = "_smoke" if SMOKE else ""
OD = f"{R}/outputs/share_grid"; os.makedirs(OD, exist_ok=True)
UNAME = UNIT.replace(" ", "_")
JOB = f"{DOMAIN}_{ARM}_{UNAME}_s{SEED}" + (f"_f{SHARES[0]:g}" if ARM == "finetune" else "") + SUF
OUT_M = f"{OD}/metrics_{JOB}.csv"; OUT_P = f"{OD}/preds_{JOB}.npz"; OUT_S = f"{OD}/sweep_{JOB}.csv"
assert not os.path.exists(OUT_M), f"refusing to overwrite completed job {OUT_M}"   # metrics file = completion marker
for f in (OUT_P, OUT_S):                       # orphans from an interrupted publish (metrics never renamed) are replaced
    if os.path.exists(f): print(f"[recover] {f} exists without {OUT_M}; it will be overwritten", flush=True)
TMP_M, TMP_P, TMP_S = OUT_M + ".part", OUT_P + ".part.npz", OUT_S + ".part"   # interrupted runs leave only these
assert os.path.exists(CKPT), f"missing encoder {CKPT}"
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] JOB={JOB} shares={SHARES} draws={DRAWS} lrs={LRS} wds={WDS} sweep={SWEEP} dev={DEV}", flush=True)

voc = json.load(open(VOCAB)); VS = [len(voc[f]) + 1 for f in FIELDS] + [201]; NF = len(VS)

# ---- data: TEST_START and VALCUT use the same rule as pretrain_domain.py (whole panel) ----
tcol = pd.read_parquet(PANEL, columns=["t", YCOL]); tmax = int(tcol.loc[tcol[YCOL].notna(), "t"].max()); del tcol
TEST_START = tmax - (11 if DOMAIN == "crime" else 23)
VALCUT = TEST_START - (3 if DOMAIN == "crime" else 12)
df = pd.read_parquet(PANEL, columns=["city", "t", YCOL, "event_sequence"] + FEAT)
if DOMAIN == "dengue":
    df = df[df["city"].astype(str) != "nan"]
    df = df[(df["city"].astype("int64") // 1000000) == REGION]
else:
    df = df[df["city"] == UNIT]
d = df[df[YCOL].notna()].reset_index(drop=True); del df
assert len(d) > 0, f"no rows for {UNIT}"
tt = d["t"].to_numpy(); trm = tt < VALCUT; vam = (tt >= VALCUT) & (tt < TEST_START); tem = tt >= TEST_START
y = d[YCOL].to_numpy().astype(int); X = d[FEAT].to_numpy(np.float32); seqs = d["event_sequence"].values
te_idx = np.where(tem)[0]
if len(te_idx) > TEST_CAP: te_idx = np.sort(np.random.default_rng(DRAW_BASE).choice(te_idx, TEST_CAP, replace=False))
if SMOKE: te_idx = np.sort(np.random.default_rng(0).choice(te_idx, size=min(5000, len(te_idx)), replace=False))
yte = y[te_idx]
assert trm.sum() > 100 and vam.sum() > 10 and 0 < yte.sum() < len(yte)
print(f"[data] {UNIT} tmax={tmax} VALCUT={VALCUT} TEST_START={TEST_START} train={trm.sum():,} val={vam.sum():,} "
      f"test={len(te_idx):,} pos_train={y[trm].mean():.3f} pos_test={yte.mean():.3f}", flush=True)
TEST_ID = dict(__t=tt[te_idx].astype(np.int64), __row=te_idx.astype(np.int64), __y=yte.astype(np.int8))

def parse(s):
    if isinstance(s, str): return json.loads(s)[-MAXLEN:]
    if isinstance(s, (list, tuple, np.ndarray)): return [list(x) for x in s][-MAXLEN:]
    if s is None or (isinstance(s, float) and np.isnan(s)): return []
    raise TypeError(f"unsupported event_sequence type {type(s)}")
class DS(Dataset):
    def __init__(self, idx):
        self.seq = [parse(s) for s in seqs[idx]]; self.y = y[idx].astype(np.float32)
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

class Enc(nn.Module):   # same layout as pretrain_domain.py / transfer_domain.py
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
def new_model(pretrained):
    m = Enc().to(DEV)
    if pretrained:
        miss, unexp = m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
        assert set(miss) == {"head.weight", "head.bias"} and not unexp, f"load mismatch miss={miss} unexp={unexp}"
    return m

def draw_rows(mask, share, k, purpose):
    rng = np.random.default_rng([DRAW_BASE, 100 * (DOMAIN == "dengue") + CITY_I, int(round(share * 1e5)), k, purpose])
    pool = np.where(mask)[0]; pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
    npos = int(round(len(pos) * share)); nneg = int(round(len(neg) * share))
    if purpose == 1:                                 # val floor of VAL_MIN_POS positives, base rate kept
        if len(pos) < VAL_MIN_POS: return None
        if npos < VAL_MIN_POS: npos = VAL_MIN_POS; nneg = min(len(neg), int(round(npos * len(neg) / len(pos))))
    elif npos < MIN_TRAIN_POS: return None
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]))

def scores(yt, p):
    return dict(pr=float(average_precision_score(yt, p)), roc=float(roc_auc_score(yt, p)),
                brier=float(np.mean((np.clip(p, 0, 1) - yt) ** 2)))
rows, preds, sweep_rows = [], {}, []
def emit(model, share, k, idx, p, **extra):
    assert len(p) == len(yte) and np.isfinite(p).all(), (model, share, k)
    r = dict(domain=DOMAIN, unit=UNIT, enc_seed=SEED, arm=ARM, model=model, share=share, draw=k,
             n_train=0 if idx is None else int(len(idx)), pos_train=np.nan if idx is None else int(y[idx].sum()),
             n_test=int(len(yte)), test_pos=float(yte.mean()), **scores(yte, p), **extra)
    rows.append(r); preds[f"{model}|{share:g}|{k}"] = p.astype(np.float32)
    pd.DataFrame(rows).to_csv(TMP_M, index=False)
    print(f"  [{model}] share={share:g} draw={k} n={r['n_train']} PR={r['pr']:.4f}", flush=True)

emit("persistence", 0.0, -1, None, X[te_idx, FEAT.index("lag1")].astype(float))

@torch.no_grad()
def run(m, ds, emb):
    m.eval(); out = []
    for seq, mask, _ in DataLoader(ds, batch_size=BATCH, collate_fn=coll, num_workers=0):
        seq, mask = seq.to(DEV), mask.to(DEV)
        out.append((m.cls_vec(seq, mask) if emb else torch.sigmoid(m(seq, mask))).float().cpu().numpy())
    return np.concatenate(out)

if ARM == "nested":
    draws = {(s, k): draw_rows(trm, s, k, 0) for s in SHARES for k in DRAWS}
    need = np.unique(np.concatenate([v for v in draws.values() if v is not None] + [te_idx]))
    Z = run(new_model(True), DS(need), True); assert Z.shape == (len(need), D) and np.isfinite(Z).all()
    posn = -np.ones(len(d), np.int64); posn[need] = np.arange(len(need))
    def Zof(ix): assert (posn[ix] >= 0).all(); return Z[posn[ix]]
    def xgb_p(Xtr, ytr, Xte):
        b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                       "colsample_bytree": 0.8, "seed": XGB_SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
        return b.predict(xgb.DMatrix(Xte))
    Zte, Bte = Zof(te_idx), X[te_idx]
    for (s, k), idx in draws.items():
        if idx is None: print(f"  [skip] share={s:g} draw={k}: too few positives", flush=True); continue
        emit("base", s, k, idx, xgb_p(X[idx], y[idx], Bte))
        emit("emb_base", s, k, idx, xgb_p(np.concatenate([Zof(idx), X[idx]], 1), y[idx], np.concatenate([Zte, Bte], 1)))
else:
    s = SHARES[0]; ds_te = DS(te_idx)
    def train_one(pretrained, lr, wd, tr, va, k):
        torch.manual_seed(SEED * 1000 + k)
        m = new_model(pretrained)
        pos = float(max(y[tr].sum(), 1.0)); neg = float(len(tr)) - pos
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(neg / pos, 1.0), device=DEV))
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        dl = DataLoader(DS(tr), batch_size=FT_BATCH, shuffle=True, collate_fn=coll, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED * 1000 + k))
        ds_va = DS(va); yva = y[va]
        best, best_ep, best_sd, bad = -1.0, 0, None, 0
        for ep in range(1, MAX_EP + 1):
            m.train()
            for seq, mask, yy in dl:
                loss = lossf(m(seq.to(DEV), mask.to(DEV)), yy.to(DEV))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            assert torch.isfinite(loss), "non-finite training loss"
            v = float(average_precision_score(yva, run(m, ds_va, False)))
            improved = v > best + MIN_DELTA                    # MIN_DELTA governs patience only
            if v > best: best, best_ep, best_sd = v, ep, {kk: t_.detach().cpu().clone() for kk, t_ in m.state_dict().items()}
            bad = 0 if improved else bad + 1
            if bad >= PATIENCE: break
        m.load_state_dict(best_sd)
        return m, best, best_ep, ep
    chosen = {}
    for k in DRAWS:
        tr = draw_rows(trm, s, k, 0); va = draw_rows(vam, s, k, 1)
        if tr is None or va is None: print(f"  [skip] share={s:g} draw={k}: too few positives", flush=True); continue
        for arm_name, pretrained in (("ft_pretrained", True), ("ft_scratch", False)):
            grid = [(lr, wd) for lr in LRS for wd in WDS] if (SWEEP == "all" or arm_name not in chosen) else [chosen[arm_name]]
            best = None
            for lr, wd in grid:
                t0 = time.time()
                m, vpr, bep, lep = train_one(pretrained, lr, wd, tr, va, k)
                sweep_rows.append(dict(domain=DOMAIN, unit=UNIT, enc_seed=SEED, model=arm_name, share=s, draw=k,
                                       lr=lr, wd=wd, val_pr=vpr, best_epoch=bep, last_epoch=lep, n_train=len(tr),
                                       n_val=len(va), val_pos=int(y[va].sum()), secs=round(time.time() - t0)))
                pd.DataFrame(sweep_rows).to_csv(TMP_S, index=False)
                print(f"    {arm_name} draw={k} lr={lr:g} wd={wd:g} valPR={vpr:.4f} best_ep={bep} stop_ep={lep} "
                      f"{time.time()-t0:.0f}s", flush=True)
                if best is None or vpr > best[0]: best = (vpr, lr, wd, bep, m)
                del m                                           # winner (if any) is held only by `best`
            if arm_name not in chosen: chosen[arm_name] = (best[1], best[2])
            emit(arm_name, s, k, tr, run(best[4], ds_te, False), lr=best[1], wd=best[2], best_epoch=best[3],
                 val_pr=best[0], n_val=len(va))
            del best
            if torch.cuda.is_available(): torch.cuda.empty_cache()

np.savez_compressed(TMP_P, **TEST_ID, **preds)
chk = np.load(TMP_P); assert all(k in chk.files for k in preds)
if os.path.exists(TMP_S): os.replace(TMP_S, OUT_S)
os.replace(TMP_P, OUT_P); os.replace(TMP_M, OUT_M)
print(f"[done] {OUT_M} ({len(rows)} rows), {OUT_P}", flush=True)

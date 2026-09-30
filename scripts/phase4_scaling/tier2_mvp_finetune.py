"""tier-2 MVP cross-attention (Option A cache) + LEARNED broad-pool selection + 2-hop + empty-cell fix.
=============================================================================================
DRAFT — must be SMOKE-tested (SMOKE=1) once the scale-up winner checkpoint exists, before the
real 5-seed run. Built 2026-06-25, parameterized; not yet run.

Unfreezes the encoder (warm-started from a scale-up contrastive checkpoint) and trains
  encoder (trainable)  ->  spatial cross-attention  ->  forecasting head
end-to-end on violence_next_month (BCE). NEIGHBOR EMBEDDINGS ARE CACHED ONCE PER EPOCH
(from encode_all at epoch start) and looked up during training (Option A) -- the focal cell is
still encoded live so encoder grads flow through it; neighbor embeddings are epoch-frozen features.

MODE (REPR2_MODE):
  nospatial   = encoder + head only (control: does fine-tuning alone help?)
  geo         = k-NN candidate set by grid centroid (physical neighbors)     [control]
  emb         = k-NN candidate set in the (current) embedding space          [hypothesis: learned/nonlocal geography]
  both        = union of geo + emb candidate sets (K/2 each, deduped)        [does learned add ON TOP of physical?]
Same SpatialXAttn (content q.k + learned dlat/dlon/dist bias) in all spatial modes; MODE only sets the candidate pool.
Compare hard (noncontig) cases vs base (xgb, unchanged) and vs nospatial.
WIN = emb (or both) beats nospatial AND base on noncontig; both>geo => learned geography is complementary.

Env: CKPT (warm-start .pt), REPR2_CAP (S/M/L/XL), REPR2_FEAT (a/e), REPR2_MODE, REPR2_SEED,
     REGION (africa|global), SMOKE, REPR2_K (default 20). Tractability: K small, neg subsample.
"""
import os, json, time, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, f1_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEED = int(os.environ.get("REPR2_SEED", "123"))
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("SMOKE", "0") == "1"
REGION = os.environ.get("REGION", "africa")
MODE = os.environ.get("REPR2_MODE", "emb")          # nospatial | geo | emb | both | learned
CAP = os.environ.get("REPR2_CAP", "L")
FEAT = os.environ.get("REPR2_FEAT", "a")
K = int(os.environ.get("REPR2_K", "20"))
HOPS = int(os.environ.get("REPR2_HOPS", "1"))
CKPT = os.environ["CKPT"]                            # warm-start checkpoint (required)
CAPACITY = {"S": (64,2,4), "M": (128,4,8), "L": (256,6,8), "XL": (512,8,16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[CAP]
MAX_SEQ_LEN = 200
BBOX = {"africa": dict(lat=(-37.,38.), lon=(-20.,55.)),
        "global": dict(lat=(-90.,90.), lon=(-180.,180.))}[REGION]
HOLDOUT_AFRICA = ["Sudan","Somalia","Uganda","Kenya","Ethiopia","South Africa"]
HOLDOUT_ALL = HOLDOUT_AFRICA + ["Iraq","Tunisia","Iran","Algeria","Morocco","Pakistan",
    "Bangladesh","Nepal","Sri Lanka","Myanmar","Philippines","Thailand","South Korea",
    "Ukraine","France","Germany","Mexico","Venezuela","Colombia","Chile"]
TEST_COUNTRIES = HOLDOUT_AFRICA if REGION == "africa" else HOLDOUT_ALL
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type",
                "month_offset","fatalities","deaths_civilians"]
FEATSET_SEL = {"a":[0,1,2,3,4,5,6], "e":[0,1,2,3,4,5,6]}
FEATSET_PRIO = {"a": False, "e": True}
SEL = FEATSET_SEL[FEAT]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
USE_PRIO = FEATSET_PRIO[FEAT]
print(f"[cfg] MODE={MODE} CAP={CAP} FEAT={FEAT} K={K} SMOKE={SMOKE} ckpt={os.path.basename(CKPT)}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 2 for k, v in vocabs.items()}
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32)
               for r in ps.itertuples(index=False)}


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, static_dim, use_prio, dropout=0.1):
        super().__init__(); self.fields = fields; self.use_prio = use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields)*d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio: self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], -1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio: cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], 1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        out = self.transformer(e, src_key_padding_mask=~mwc)
        z = self.ln(out[:, 0, :])
        return z   # MVP: no empty-cell zeroing -> feat-e empties differentiated by PRIO (de-pollutes emb-kNN)


class SpatialXAttn(nn.Module):
    def __init__(self, d, nh=4):
        super().__init__(); self.nh = nh; self.dh = d // nh
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, nh)); self.out = nn.Linear(d, d)
    def forward(self, zc, zn, rel, mask):
        B, Kk, d = zn.shape
        q = self.q(zc).view(B, self.nh, self.dh); k = self.k(zn).view(B, Kk, self.nh, self.dh)
        v = self.v(zn).view(B, Kk, self.nh, self.dh)
        logit = torch.einsum('bhd,bkhd->bhk', q, k) / (self.dh ** 0.5)
        logit = logit + self.bias(rel).permute(0, 2, 1)
        logit = logit.masked_fill(~mask.unsqueeze(1), -1e9)
        ctx = torch.einsum('bhk,bkhd->bhd', torch.softmax(logit, -1), v).reshape(B, d)
        return self.out(ctx)


class Model(nn.Module):
    def __init__(self, enc, d, spatial, hops=1):
        super().__init__(); self.enc = enc; self.spatial = spatial; self.hops = hops
        if spatial:
            self.attn = SpatialXAttn(d)
            if hops >= 2:
                self.proj1 = nn.Linear(d, d); self.ln1 = nn.LayerNorm(d); self.attn2 = SpatialXAttn(d)
        hd = 2*d if spatial else d
        self.head = nn.Sequential(nn.Linear(hd, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1))
    def forward(self, tgt, zn=None, rel=None, mask=None):
        zc = self.enc.cls_vec(*tgt)
        if not self.spatial:
            return self.head(zc).squeeze(-1)
        ctx = self.attn(zc, zn, rel, mask)   # hop-1 (zn = per-epoch cached neighbor embeddings, Option A)
        if self.hops >= 2:
            zc2 = self.ln1(zc + self.proj1(ctx))   # refine focal with hop-1 context
            ctx = self.attn2(zc2, zn, rel, mask)   # hop-2 read over same cached pool
        return self.head(torch.cat([zc, ctx], -1)).squeeze(-1)


# ---------------- data ----------------
base_cols = ["violence_lag1","violence_lag3","violence_lag6","spatial_lag_violence",
             "violent_count_roll1","violent_count_roll3","all_count_roll1","any_event_lag1",
             "log_event_density_lag1","months_since_last_event"]
need = (["grid_id","year_month","event_sequence","violence_next_month","country",
         "grid_lat_center","grid_lon_center"] + base_cols)
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=[c for c in dict.fromkeys(need)])
panel = panel[panel["violence_next_month"].notna()].copy()
la, lo = panel["grid_lat_center"], panel["grid_lon_center"]
panel = panel[(la.between(*BBOX["lat"])) & (lo.between(*BBOX["lon"]))].reset_index(drop=True)
panel = panel.merge(ps, on="grid_id", how="left")
panel[base_cols + static_cols] = panel[base_cols + static_cols].fillna(0.0)
if SMOKE:
    keep = sorted(panel["year_month"].unique())[-24:]
    panel = panel[panel["year_month"].isin(keep)].reset_index(drop=True)
N = len(panel)
SEQS = [json.loads(s) if isinstance(s, str) else (s or []) for s in panel["event_sequence"]]
GID = panel["grid_id"].to_numpy()
PRIO = np.stack([prio_lookup.get(int(g), np.zeros(STATIC_DIM, np.float32)) for g in GID])
lat = panel["grid_lat_center"].to_numpy(); lon = panel["grid_lon_center"].to_numpy()
y = panel["violence_next_month"].to_numpy().astype(np.float32)
lag1 = panel["violence_lag1"].to_numpy(); slag = panel["spatial_lag_violence"].to_numpy()
noncontig = (lag1 == 0) & (slag == 0); onset = (lag1 == 0)
is_hold = panel["country"].isin(HOLDOUT_ALL).to_numpy()
test_m = panel["country"].isin(TEST_COUNTRIES).to_numpy(); train_m = ~is_hold
BASE = panel[base_cols + static_cols].to_numpy(np.float32)
month_idx = panel.groupby("year_month").indices
print(f"[panel] N={N} train={train_m.sum()} test={test_m.sum()} pos_noncontig_test={int(y[test_m&noncontig].sum())}", flush=True)


def build_tensors(rows):
    B = len(rows)
    t = torch.zeros(B, MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
    m = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.bool)
    st = torch.from_numpy(PRIO[rows]) if USE_PRIO else torch.zeros(B, STATIC_DIM)
    for bi, r in enumerate(rows):
        s = SEQS[r][-MAX_SEQ_LEN:]
        for j, ev in enumerate(s):
            for f in range(NUM_FIELDS): t[bi, j, f] = ev[SEL[f]]
            m[bi, j] = True
    return t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)


@torch.no_grad()
def encode_all(model):
    model.eval(); Z = np.zeros((N, EMBED_DIM), np.float32)
    for i in range(0, N, 512):
        rows = np.arange(i, min(i+512, N))
        Z[rows] = model.enc.cls_vec(*build_tensors(rows)).cpu().numpy()
    return Z


def _knn_local(F, m, want):
    """local self-excluded neighbor indices [m, want] (0..m-1), -1 padded."""
    kk = min(want, m-1)
    C = NearestNeighbors(n_neighbors=kk+1).fit(F).kneighbors(F, return_distance=False)[:, 1:]
    if kk < want: C = np.concatenate([C, -np.ones((m, want-kk), int)], axis=1)
    return C

def build_index(Z):
    """Per-epoch candidate set (row -> K neighbor rows, -1 pad) + geo relpos. Returns NB[N,K],REL[N,K,3].
    MODE sets the candidate pool: geo=physical kNN, emb=embedding kNN, both=union (K/2 each, deduped)."""
    NB = np.full((N, K), -1, np.int64); REL = np.zeros((N, K, 3), np.float32)
    for ym, idx in month_idx.items():
        idx = np.asarray(idx); m = len(idx)
        if m < 2: continue
        GEO = np.column_stack([lat[idx], lon[idx]])
        if MODE == "geo":
            C = _knn_local(GEO, m, K)
        elif MODE == "emb":
            C = _knn_local(Z[idx], m, K)
        else:  # both/learned: union of source pools, deduped per row, capped at K cols
            if MODE == "learned":   # broad unprejudiced pool: physical + embedding + random
                kg = min(30, m-1); ke2 = min(30, m-1); kr = min(10, m-1)
                srcs = [_knn_local(GEO, m, kg), _knn_local(Z[idx], m, ke2),
                        np.array([rng.choice(m, size=kr, replace=False) for _ in range(m)])]
            else:  # both: physical + embedding (K/2 each)
                ke = K // 2; srcs = [_knn_local(Z[idx], m, ke), _knn_local(GEO, m, K - ke)]
            C = np.full((m, K), -1, int)
            for r in range(m):
                seen = []
                for src in srcs:
                    for c in src[r]:
                        if c >= 0 and c != r and c not in seen: seen.append(c)
                        if len(seen) == K: break
                    if len(seen) == K: break
                C[r, :len(seen)] = seen
        valid = C >= 0; gidx = idx[np.clip(C, 0, m-1)]
        NB[idx] = np.where(valid, gidx, -1)
        dlat = lat[gidx]-lat[idx][:, None]; dlon = lon[gidx]-lon[idx][:, None]
        REL[idx, :, 0] = np.where(valid, dlat, 0.0); REL[idx, :, 1] = np.where(valid, dlon, 0.0)
        REL[idx, :, 2] = np.where(valid, np.log1p(np.sqrt(dlat**2+dlon**2)), 0.0)
    return NB, REL


# ---------------- train ----------------
enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, STATIC_DIM, USE_PRIO).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
miss, unexp = enc.load_state_dict(sd, strict=False)
print(f"[warm-start] {CKPT} missing={len(miss)} unexpected={len(unexp)}", flush=True)
spatial = MODE in ("emb", "geo", "both", "learned")
model = Model(enc, EMBED_DIM, spatial, hops=HOPS).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)   # low LR for fine-tune
tr_pos = np.where(train_m & (y == 1))[0]; tr_neg = np.where(train_m & (y == 0))[0]
pos_w = torch.tensor([len(tr_neg)/max(len(tr_pos), 1)], device=DEVICE)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
BATCH = 64 if not SMOKE else 32
EPOCHS = 8 if not SMOKE else 2

for ep in range(EPOCHS):
    t0 = time.time()
    Zc = NB = REL = None
    if spatial:
        Zc = encode_all(model); NB, REL = build_index(Zc)   # cache per-epoch embeddings (Option A)
    model.train()
    neg_samp = rng.choice(tr_neg, size=min(len(tr_neg), 3*len(tr_pos)), replace=False)
    rows_ep = np.concatenate([tr_pos, neg_samp]); rng.shuffle(rows_ep)
    tot = 0.0
    for i in range(0, len(rows_ep), BATCH):
        b = rows_ep[i:i+BATCH]
        tgt = build_tensors(b); yb = torch.from_numpy(y[b]).to(DEVICE)
        if spatial:
            nb = NB[b]; nb_safe = np.where(nb < 0, 0, nb)
            zn = torch.from_numpy(Zc[nb_safe]).to(DEVICE)   # [B,K,D] cached neighbor embeddings
            rel = torch.from_numpy(REL[b]).to(DEVICE); mk = torch.from_numpy(nb >= 0).to(DEVICE)
            out = model(tgt, zn, rel, mk)
        else:
            out = model(tgt)
        loss = lossf(out, yb); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"[ep{ep}] loss={tot/(len(rows_ep)//BATCH+1):.4f} {time.time()-t0:.0f}s", flush=True)

# ---------------- eval ----------------
@torch.no_grad()
def predict_test():
    model.eval(); te = np.where(test_m)[0]; preds = np.zeros(len(te), np.float32)
    Zc = NB = REL = None
    if spatial: Zc = encode_all(model); NB, REL = build_index(Zc)
    for i in range(0, len(te), 256):
        b = te[i:i+256]; tgt = build_tensors(b)
        if spatial:
            nb = NB[b]; nb_safe = np.where(nb < 0, 0, nb)
            zn = torch.from_numpy(Zc[nb_safe]).to(DEVICE)
            out = model(tgt, zn, torch.from_numpy(REL[b]).to(DEVICE),
                        torch.from_numpy(nb >= 0).to(DEVICE))
        else:
            out = model(tgt)
        preds[i:i+len(b)] = torch.sigmoid(out).cpu().numpy()
    return te, preds

te, p_ft = predict_test()
clf = xgb.train({"objective":"binary:logistic","max_depth":6,"eta":0.1,"subsample":0.8,"seed":SEED},
                xgb.DMatrix(BASE[train_m], label=y[train_m]), 200)
p_base = clf.predict(xgb.DMatrix(BASE[test_m]))
yte = y[test_m].astype(int)
def bf1(yy, pp):
    if len(np.unique(yy)) < 2: return np.nan
    return max(f1_score(yy, (pp >= t).astype(int), zero_division=0)
               for t in np.quantile(pp, np.linspace(0.5, 0.99, 20)))
def rep(name, mask):
    yy = yte[mask]
    if yy.sum() < 5: print(f"  [{name}] too few pos"); return
    print(f"  [{name}] n={int(mask.sum())} pos={int(yy.sum())}", flush=True)
    print(f"      base        PR-AUC={average_precision_score(yy,p_base[mask]):.4f} bestF1={bf1(yy,p_base[mask]):.4f}", flush=True)
    print(f"      ft_{MODE:8s} PR-AUC={average_precision_score(yy,p_ft[mask]):.4f} bestF1={bf1(yy,p_ft[mask]):.4f}", flush=True)
print(f"\n===== TIER-2 RESULT  mode={MODE} cap={CAP} feat={FEAT} seed={SEED} =====", flush=True)
rep("all", np.ones(test_m.sum(), bool)); rep("onset", onset[test_m]); rep("noncontig (HARD)", noncontig[test_m])
np.save(f"{R}/outputs/mvp_pred_{MODE}_{CAP}_{FEAT}_s{SEED}.npy", p_ft)
print("[done]", flush=True)

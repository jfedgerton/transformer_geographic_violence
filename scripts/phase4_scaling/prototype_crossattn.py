"""tier-1 cell-cell cross-attention prototype (FROZEN stage-1 = contrastive·e).
=============================================================================
THE TEST: are a cell's useful "neighbors" its closest cells in the LEARNED
EMBEDDING space (similar dynamics/actors, possibly geographically distant) rather
than its geographic neighbors? Three retrieval modes for the attention candidate set:
  emb  = k-NN in the frozen stage-1 embedding space  (nonlocal-capable; the claim)
  geo  = k-NN by grid centroid (lat,lon)             (control: physical geography)
  rand = k random same-month cells                   (control: noise floor)
In ALL modes the geographic relative position (dlat,dlon,logdist) is fed as the
attention bias -> we can SEE when emb-mode attends across large distance (a corridor).

Compare on held-out countries' HARD (non-contiguous onset) cases vs:
  base = spatial-lag+PRIO (xgb),  emb = same frozen embedding, NO spatial context.
WIN: emb-mode spatial > geo-mode spatial > rand, AND beats base+emb on hard cases.
Frozen stage-1 => LOWER BOUND. Neighbors use z_{c',t} (events <= t); target t+1 => no leakage.
Scope: REGION env (africa default | global). SMOKE=1 limits months. Seed 123.
"""
import os, json, time, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, f1_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEED = 123
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("SMOKE", "0") == "1"
REGION = os.environ.get("REGION", "africa")
MODES = os.environ.get("MODES", "emb,geo,rand").split(",")
K = 50
CKPT = f"{R}/data_model/repr_model_M_contrastive_e_s123.pt"
EMBED_DIM, N_LAYERS, N_HEADS = 128, 4, 8
MAX_SEQ_LEN, BATCH = 200, 512
BOX = {"africa": dict(lat=(-37.0, 38.0), lon=(-20.0, 55.0)),
       "global": dict(lat=(-90.0, 90.0), lon=(-180.0, 180.0))}[REGION]
HOLDOUT_AFRICA = ["Sudan", "Somalia", "Uganda", "Kenya", "Ethiopia", "South Africa"]
HOLDOUT_ALL = HOLDOUT_AFRICA + ["Iraq","Tunisia","Iran","Algeria","Morocco","Pakistan",
    "Bangladesh","Nepal","Sri Lanka","Myanmar","Philippines","Thailand","South Korea",
    "Ukraine","France","Germany","Mexico","Venezuela","Colombia","Chile"]
TEST_COUNTRIES = HOLDOUT_AFRICA if REGION == "africa" else HOLDOUT_ALL
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type",
                "actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
print(f"[cfg] dev={DEVICE} SMOKE={SMOKE} REGION={REGION} MODES={MODES} K={K}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 2 for k, v in vocabs.items()}
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32)
               for r in ps.itertuples(index=False)}


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, static_dim, dropout=0.1):
        super().__init__()
        self.fields = fields
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], -1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1) + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], 1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        out = self.transformer(e, src_key_padding_mask=~mwc)
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)


class SeqDS(Dataset):
    def __init__(self, seqs, gids): self.seqs = seqs; self.gids = gids
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]; s = json.loads(s) if isinstance(s, str) else (s or [])
        s = s[-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(s):
            for f in range(NUM_FIELDS): t[j, f] = ev[SEL[f]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        return t, m, st


def extract(model, seqs, gids):
    ld = DataLoader(SeqDS(seqs, gids), batch_size=BATCH, shuffle=False, num_workers=0)
    Z = []
    with torch.no_grad():
        for t, m, st in ld:
            Z.append(model.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, 0).astype(np.float32)


# ---------------- load panel ----------------
base_cols = ["violence_lag1","violence_lag3","violence_lag6","spatial_lag_violence",
             "violent_count_roll1","violent_count_roll3","all_count_roll1","any_event_lag1",
             "log_event_density_lag1","months_since_last_event"]
need = (["grid_id","year_month","event_sequence","violence_next_month","country",
         "grid_lat_center","grid_lon_center"] + base_cols)   # PRIO statics merged below
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=[c for c in dict.fromkeys(need)])
panel = panel[panel["violence_next_month"].notna()].copy()
la, lo = panel["grid_lat_center"], panel["grid_lon_center"]
panel = panel[(la.between(*BOX["lat"])) & (lo.between(*BOX["lon"]))].reset_index(drop=True)
panel = panel.merge(ps, on="grid_id", how="left")
panel[base_cols + static_cols] = panel[base_cols + static_cols].fillna(0.0)
if SMOKE:
    keep = sorted(panel["year_month"].unique())[-24:]
    panel = panel[panel["year_month"].isin(keep)].reset_index(drop=True)
print(f"[panel] rows={len(panel)} months={panel['year_month'].nunique()} "
      f"cells={panel['grid_id'].nunique()}", flush=True)

# ---------------- frozen embeddings ----------------
enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, STATIC_DIM).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
miss, unexp = enc.load_state_dict(sd, strict=False); enc.eval()
print(f"[enc] loaded missing={len(miss)} unexpected={len(unexp)}", flush=True)
t0 = time.time()
Z = extract(enc, panel["event_sequence"].values, panel["grid_id"].values)
print(f"[emb] Z={Z.shape} in {time.time()-t0:.0f}s", flush=True)

lat = panel["grid_lat_center"].to_numpy(); lon = panel["grid_lon_center"].to_numpy()
month_idx = panel.groupby("year_month").indices

def build_neighbors(mode):
    """Return NB[N,K] (global row idx, -1 pad) and REL[N,K,3] (dlat,dlon,logdist, geographic always)."""
    N = len(panel); NB = np.full((N, K), -1, np.int64); REL = np.zeros((N, K, 3), np.float32)
    for ym, idx in month_idx.items():
        idx = np.asarray(idx); m = len(idx)
        if m < 2: continue
        kk = min(K, m - 1)
        if mode == "geo":
            F = np.column_stack([lat[idx], lon[idx]])
            C = NearestNeighbors(n_neighbors=kk+1).fit(F).kneighbors(F, return_distance=False)[:, 1:]
        elif mode == "emb":
            F = Z[idx]
            C = NearestNeighbors(n_neighbors=kk+1).fit(F).kneighbors(F, return_distance=False)[:, 1:]
        else:  # rand
            C = rng.integers(0, m, size=(m, kk))
            hit = C == np.arange(m)[:, None]; C[hit] = (C[hit] + 1) % m
        if kk < K:
            C = np.concatenate([C, -np.ones((m, K - kk), int)], axis=1)
        valid = C >= 0; Cc = np.clip(C, 0, m - 1); gidx = idx[Cc]
        NB[idx] = np.where(valid, gidx, -1)
        dlat = lat[gidx] - lat[idx][:, None]; dlon = lon[gidx] - lon[idx][:, None]
        dist = np.log1p(np.sqrt(dlat**2 + dlon**2))
        REL[idx, :, 0] = np.where(valid, dlat, 0.0)
        REL[idx, :, 1] = np.where(valid, dlon, 0.0)
        REL[idx, :, 2] = np.where(valid, dist, 0.0)
    return NB, REL

# ---------------- splits / targets ----------------
is_hold = panel["country"].isin(HOLDOUT_ALL).to_numpy()
test_m = panel["country"].isin(TEST_COUNTRIES).to_numpy()
train_m = ~is_hold
y = panel["violence_next_month"].to_numpy().astype(np.float32)
lag1 = panel["violence_lag1"].to_numpy(); slag = panel["spatial_lag_violence"].to_numpy()
onset = (lag1 == 0); noncontig = (lag1 == 0) & (slag == 0)
BASE = panel[base_cols + static_cols].to_numpy(np.float32)
print(f"[split] train={train_m.sum()} test={test_m.sum()} "
      f"test_noncontig={int((test_m&noncontig).sum())} pos_noncontig={int(y[test_m&noncontig].sum())}", flush=True)
Zt = torch.from_numpy(Z); yt = torch.from_numpy(y)

# ---------------- models ----------------
class SpatialXAttn(nn.Module):
    def __init__(self, d, nh=4):
        super().__init__(); self.nh = nh; self.dh = d // nh
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, nh))
        self.out = nn.Linear(d, d)
    def forward(self, zc, zn, rel, mask):
        B, Kk, d = zn.shape
        q = self.q(zc).view(B, self.nh, self.dh)
        k = self.k(zn).view(B, Kk, self.nh, self.dh)
        v = self.v(zn).view(B, Kk, self.nh, self.dh)
        logit = torch.einsum('bhd,bkhd->bhk', q, k) / (self.dh ** 0.5)
        logit = logit + self.bias(rel).permute(0, 2, 1)
        logit = logit.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = torch.softmax(logit, -1)
        ctx = torch.einsum('bhk,bkhd->bhd', attn, v).reshape(B, d)
        return self.out(ctx), attn

class SpatialModel(nn.Module):
    def __init__(self, d):
        super().__init__(); self.attn = SpatialXAttn(d)
        self.head = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1))
    def forward(self, zc, zn, rel, mask):
        ctx, a = self.attn(zc, zn, rel, mask)
        return self.head(torch.cat([zc, ctx], -1)).squeeze(-1), a

class EmbModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1))
    def forward(self, zc): return self.head(zc).squeeze(-1)

def train_eval(spatial, NBt=None, RELt=None):
    torch.manual_seed(SEED)
    if spatial:
        nb_safe = NBt.clamp(min=0); nb_mask = (NBt >= 0)
        def gather(rows):
            idx = nb_safe[rows]; zn = Zt[idx] * nb_mask[rows].unsqueeze(-1)
            return zn, RELt[rows], nb_mask[rows]
    model = (SpatialModel if spatial else EmbModel)(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = np.where(train_m)[0]
    pos_w = torch.tensor([(yt[tr] == 0).sum() / max((yt[tr] == 1).sum(), 1)], device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    EP = 3 if SMOKE else 15
    for ep in range(EP):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 1024):
            bt = torch.from_numpy(perm[i:i+1024]); zc = Zt[bt].to(DEVICE); yb = yt[bt].to(DEVICE)
            opt.zero_grad()
            if spatial:
                zn, rel, mk = gather(bt)
                out, _ = model(zc, zn.to(DEVICE), rel.to(DEVICE), mk.to(DEVICE))
            else:
                out = model(zc)
            lossf(out, yb).backward(); opt.step()
    model.eval(); te = np.where(test_m)[0]; preds = np.zeros(len(te), np.float32)
    with torch.no_grad():
        for i in range(0, len(te), 4096):
            bt = torch.from_numpy(te[i:i+4096]); zc = Zt[bt].to(DEVICE)
            if spatial:
                zn, rel, mk = gather(bt)
                out, _ = model(zc, zn.to(DEVICE), rel.to(DEVICE), mk.to(DEVICE))
            else:
                out = model(zc)
            preds[i:i+len(bt)] = torch.sigmoid(out).cpu().numpy()
    return preds

def base_pred():
    clf = xgb.train({"objective":"binary:logistic","max_depth":6,"eta":0.1,
                     "subsample":0.8,"seed":SEED}, xgb.DMatrix(BASE[train_m], label=y[train_m]), 200)
    return clf.predict(xgb.DMatrix(BASE[test_m]))

# ---------------- run ----------------
results = {}
print("[train] base ...", flush=True); results["base"] = base_pred()
print("[train] emb (no spatial) ...", flush=True); results["emb"] = train_eval(False)
for mode in MODES:
    t1 = time.time(); NB, REL = build_neighbors(mode)
    mean_d = REL[NB >= 0][:, 2].mean()
    print(f"[nbr:{mode}] built in {time.time()-t1:.0f}s; mean log-geo-dist of selected nbrs={mean_d:.3f}", flush=True)
    results[f"spatial_{mode}"] = train_eval(True, torch.from_numpy(NB), torch.from_numpy(REL))
    print(f"[train] spatial_{mode} done", flush=True)

# ---------------- score on hard cases ----------------
yte = y[test_m].astype(int)
def bestf1(yy, pp):
    if len(np.unique(yy)) < 2: return np.nan
    ths = np.quantile(pp, np.linspace(0.5, 0.99, 20))
    return max(f1_score(yy, (pp >= t).astype(int), zero_division=0) for t in ths)
def report(name, mask):
    yy = yte[mask]
    if yy.sum() < 5 or len(np.unique(yy)) < 2:
        print(f"  [{name}] n={int(mask.sum())} pos={int(yy.sum())} -> too few", flush=True); return
    print(f"  [{name}] n={int(mask.sum())} pos={int(yy.sum())}", flush=True)
    for mn, pp in results.items():
        print(f"      {mn:16s} PR-AUC={average_precision_score(yy, pp[mask]):.4f}  bestF1={bestf1(yy, pp[mask]):.4f}", flush=True)

print("\n===== TIER-1 RESULT (held-out countries) =====", flush=True)
report("all", np.ones(test_m.sum(), bool))
report("onset", onset[test_m])
report("noncontig (HARD)", noncontig[test_m])
print("\n[done]", flush=True)

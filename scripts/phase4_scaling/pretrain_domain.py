"""pretrain_domain.py -- within-domain contrastive pretraining for the appendix
crime / dengue transfer test (mirrors the conflict LORO design).

DOMAIN=crime|dengue. Pretrains the same L-size encoder used in crime_finetune.py /
dengue_finetune.py on every NON-held-out city, using only months before the
held-out cities' test window:
    test window   = last 24 months with a target  (t >= TEST_START)
    early stop    = 12 months before TEST_START
    pretrain rows = t <  TEST_START - 12
Objective matches the conflict encoder: NT-Xent (tau=0.1), two views per unit made
by dropping 15% of records. Held-out cities are never loaded into pretraining.
Seed 123 by default. Output: data_model/repr_model_L_contrastive_{DOMAIN}_dom_s{SEED}.pt
"""
import os, json, time, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
E = os.environ.get
DOMAIN = E("DOMAIN", "dengue"); SEED = int(E("SEED", "123")); SMOKE = E("SMOKE", "0") == "1"
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
    HELDOUT = None                         # dengue: hold out one IBGE macro-region (first code digit)
REGION = [1, 2, 3, 4, 5][int(E("REGION", E("SLURM_ARRAY_TASK_ID", "0")))] if DOMAIN == "dengue" else None
TAG = f"_r{REGION}" if DOMAIN == "dengue" else ""   # 1=N 2=NE 3=SE 4=S 5=CO
D, NL, NH, DROP = 256, 6, 8, 0.1          # L capacity, same as conflict
MAXLEN = 200; TAU = 0.1; PDROP = 0.15
LR = 3e-4; BATCH = int(E("BATCH", "256")); EPOCHS = int(E("EPOCHS", "20")); PATIENCE = 3
MAX_PRE = int(E("MAX_PRE", "300000"))     # cap on pretraining units (speed)
OUT = f"{R}/data_model/repr_model_L_contrastive_{DOMAIN}_dom{TAG}_s{SEED}.pt"

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] DOMAIN={DOMAIN} REGION={REGION} SEED={SEED} SMOKE={SMOKE} dev={DEV} MAX_PRE={MAX_PRE}", flush=True)

voc = json.load(open(VOCAB))
VS = [len(voc[f]) + 1 for f in FIELDS] + [201]    # +pad; month offset shifted +1
NF = len(VS)

df = pd.read_parquet(PANEL, columns=["city", "t", YCOL, "event_sequence"])
df = df[df["city"].astype(str) != "nan"].reset_index(drop=True)
held = ((df["city"].astype("int64") // 1000000) == REGION).values if DOMAIN == "dengue" else df["city"].isin(HELDOUT).values
assert held.sum() > 0, "no held-out cities found -- check HELDOUT ids"
tmax = int(df.loc[df[YCOL].notna(), "t"].max())
TEST_START = tmax - (11 if DOMAIN == "crime" else 23)   # crime panel is only 35 months
VALCUT = TEST_START - (3 if DOMAIN == "crime" else 12)
pool = df[~held].reset_index(drop=True)
del df
assert held.sum() < len(held)
pre = pool[pool["t"] < VALCUT].reset_index(drop=True)
val = pool[(pool["t"] >= VALCUT) & (pool["t"] < TEST_START)].reset_index(drop=True)
assert pre["t"].max() < VALCUT and val["t"].max() < TEST_START

def nonempty(s):
    return isinstance(s, str) and len(s) > 2 or (not isinstance(s, str) and s is not None and len(s) > 0)
pre = pre[pre["event_sequence"].map(nonempty)].reset_index(drop=True)
val = val[val["event_sequence"].map(nonempty)].reset_index(drop=True)
nmax = 20000 if SMOKE else MAX_PRE
if len(pre) > nmax: pre = pre.sample(n=nmax, random_state=SEED).reset_index(drop=True)
if len(val) > nmax // 5: val = val.sample(n=nmax // 5, random_state=SEED).reset_index(drop=True)
if SMOKE: EPOCHS = 2
print(f"[data] tmax={tmax} TEST_START={TEST_START} VALCUT={VALCUT} pre={len(pre):,} val={len(val):,}", flush=True)
assert len(pre) > 1000 and len(val) > 200

def parse(s):
    seq = json.loads(s) if isinstance(s, str) else [list(x) for x in s]
    return seq[-MAXLEN:]

class DS(Dataset):
    def __init__(self, d):
        self.seq = [parse(s) for s in d["event_sequence"].values]
        assert all(len(ev) == NF for sq in self.seq[:2000] for ev in sq), f"event length != {NF}"
    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        t = torch.zeros(MAXLEN, NF, dtype=torch.long); m = torch.zeros(MAXLEN, dtype=torch.bool)
        for j, ev in enumerate(self.seq[i]):
            for k in range(NF - 1): t[j, k] = int(ev[k])
            t[j, NF - 1] = min(int(ev[NF - 1]), 199) + 1
            m[j] = True
        return t, m
def coll(b): return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])

class Enc(nn.Module):   # identical layout to DengueEnc / crime encoder in the *_finetune.py scripts
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

model = Enc().to(DEV)
proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 128)).to(DEV)   # SimCLR projection head, discarded after
opt = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=LR)

def view(mask):
    keep = (torch.rand_like(mask, dtype=torch.float) > PDROP) & mask
    empty = keep.sum(1) == 0                      # never drop every record
    keep[empty] = mask[empty]
    return keep

def nt_xent(z1, z2):
    z = F.normalize(torch.cat([z1, z2]), dim=1); n = z1.size(0)
    sim = z @ z.t() / TAU
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)

def run(loader, train):
    model.train(train); proj.train(train); tot = 0.0; nb = 0
    with torch.set_grad_enabled(train):
        for seq, mask in loader:
            seq, mask = seq.to(DEV), mask.to(DEV)
            if seq.size(0) < 4: continue
            loss = nt_xent(proj(model.cls_vec(seq, view(mask))), proj(model.cls_vec(seq, view(mask))))
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
    return tot / max(nb, 1)

tl = DataLoader(DS(pre), batch_size=BATCH, shuffle=True, collate_fn=coll, num_workers=2, drop_last=True)
vl = DataLoader(DS(val), batch_size=BATCH, shuffle=False, collate_fn=coll, num_workers=2, drop_last=True)
best = 1e9; wait = 0
for ep in range(1, EPOCHS + 1):
    t0 = time.time(); tr = run(tl, True); va = run(vl, False)
    assert math.isfinite(tr) and math.isfinite(va), "non-finite loss"
    print(f"[ep {ep:2d}] train={tr:.4f} val={va:.4f} {time.time()-t0:.0f}s", flush=True)
    if va < best - 1e-4:
        best = va; wait = 0
        torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("head.")}, OUT)
    else:
        wait += 1
        if wait >= PATIENCE: print(f"[stop] ep {ep} best_val={best:.4f}", flush=True); break
assert os.path.exists(OUT)
print(f"[done] saved {OUT} best_val={best:.4f} TEST_START={TEST_START}", flush=True)

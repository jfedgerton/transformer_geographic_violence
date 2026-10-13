"""pretrain_ces.py -- contrastive pretraining of the CES respondent encoder for one held-out Census division.

Mirrors pretrain_domain.py (same L encoder, NT-Xent tau=0.1, two views made by dropping 15% of tokens, SimCLR
projection head discarded after training, Adam lr 3e-4, best-validation weights saved without the head).
Differences: tokens are survey answers (item=value token, item), so there is no month-offset field.
Data the encoder sees: respondents from the OTHER eight divisions, years 2006-2019 (all respondents, including odd
years; no outcome is used). Early stopping on the other divisions' 2020 respondents. It never sees the held-out
division, any respondent from 2021 on, or any outcome.
Env: DIV (0-8), SEED (123/456/789), SMOKE. Output: data_model/repr_model_L_contrastive_ces_dom_d{DIV}_s{SEED}.pt
"""
import os, json, time, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
E = os.environ.get
DIV = int(E("DIV", "0")); SEED = int(E("SEED", "123")); SMOKE = E("SMOKE", "0") == "1"
DIVISIONS = ["New England", "Middle Atlantic", "East North Central", "West North Central", "South Atlantic",
             "East South Central", "West South Central", "Mountain", "Pacific"]
assert 0 <= DIV <= 8
HELD = DIVISIONS[DIV]
PANEL = f"{R}/data_intermediate/ces_tokens.parquet"; VOCAB = f"{R}/data_intermediate/ces_vocab.json"
D, NL, NH, DROP = 256, 6, 8, 0.1
MAXLEN = 40; TAU = 0.1; PDROP = 0.15
LR = 3e-4; BATCH = int(E("BATCH", "512")); EPOCHS = int(E("EPOCHS", "30")); PATIENCE = int(E("PATIENCE", "5"))
OUT = f"{R}/data_model/repr_model_L_contrastive_ces_dom_d{DIV}_s{SEED}" + ("_smoke" if SMOKE else "") + ".pt"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] held-out={HELD} SEED={SEED} SMOKE={SMOKE} dev={DEV}", flush=True)

voc = json.load(open(VOCAB)); VS = [len(voc["tok_id"]) + 1, len(voc["item_id"]) + 1]; NF = len(VS)
df = pd.read_parquet(PANEL, columns=["year", "division", "event_sequence"])
assert set(df["division"].unique()) == set(DIVISIONS)
pool = df[df["division"] != HELD]
pre = pool[pool["year"].between(2006, 2019)].reset_index(drop=True)
val = pool[pool["year"] == 2020].reset_index(drop=True)
assert (df.loc[df["division"] == HELD].shape[0] > 0) and pre["year"].max() <= 2019 and (val["year"] == 2020).all()
assert HELD not in set(pre["division"]) | set(val["division"])
del df, pool
if SMOKE:
    pre = pre.sample(n=20000, random_state=SEED).reset_index(drop=True); val = val.sample(n=4000, random_state=SEED).reset_index(drop=True)
    EPOCHS = 2
print(f"[data] pre={len(pre):,} (2006-2019) val={len(val):,} (2020), vocab={VS}", flush=True)

class DS(Dataset):
    def __init__(self, d):
        self.seq = [json.loads(s)[:MAXLEN] for s in d["event_sequence"].values]
        assert all(len(ev) == NF for sq in self.seq[:2000] for ev in sq)
    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        t = torch.zeros(MAXLEN, NF, dtype=torch.long); m = torch.zeros(MAXLEN, dtype=torch.bool)
        for j, ev in enumerate(self.seq[i]):
            t[j, 0] = int(ev[0]); t[j, 1] = int(ev[1]); m[j] = True
        return t, m
def coll(b): return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])

class Enc(nn.Module):   # identical layout to share_grid_ces.py (head unused here, dropped when saving)
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
proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 128)).to(DEV)
opt = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=LR)

def view(mask):
    keep = (torch.rand_like(mask, dtype=torch.float) > PDROP) & mask
    empty = keep.sum(1) == 0
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

g = torch.Generator().manual_seed(SEED)
tl = DataLoader(DS(pre), batch_size=BATCH, shuffle=True, collate_fn=coll, num_workers=2, drop_last=True, generator=g)
vl = DataLoader(DS(val), batch_size=BATCH, shuffle=False, collate_fn=coll, num_workers=2, drop_last=True)
best = 1e9; wait = 0; TMP = OUT + ".part"
for ep in range(1, EPOCHS + 1):
    t0 = time.time(); tr = run(tl, True); va = run(vl, False)
    assert math.isfinite(tr) and math.isfinite(va), "non-finite loss"
    print(f"[ep {ep:2d}] train={tr:.4f} val={va:.4f} {time.time()-t0:.0f}s", flush=True)
    improved = va < best - 1e-4                      # min-delta governs patience only
    if va < best:
        best = va
        torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("head.")}, TMP)
    wait = 0 if improved else wait + 1
    if wait >= PATIENCE: print(f"[stop] ep {ep} best_val={best:.4f}", flush=True); break
os.replace(TMP, OUT)
print(f"[done] saved {OUT} best_val={best:.4f} stopped_ep={ep} max_ep={EPOCHS}", flush=True)

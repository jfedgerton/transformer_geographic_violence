"""02_train_ucdp_repr.py -- mini MLM encoder on UCDP GED sequences (open-data HF arm).
5 fields: type_of_violence, side_a, side_b, deaths_bin, month_offset. Warmup+clip from
the start (lesson from the ACLED L/XL NaN bug). Seed 123 default.
Env: UCDP_TAG, UCDP_DIM, UCDP_LAYERS, UCDP_HEADS, UCDP_SEED, UCDP_EPOCHS, UCDP_PATIENCE,
UCDP_BATCH, UCDP_LR, UCDP_WARMUP, UCDP_CLIP, UCDP_MAX_ROWS (smoke).
Out: data_model/ucdp_repr_{TAG}.pt + logs/ucdp_log_{TAG}.csv
"""
import os, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
PANEL = f"{R}/data_intermediate/ucdp_panel.parquet"
VOCAB = f"{R}/data_intermediate/ucdp_vocab.json"
assert "acled" not in PANEL.lower() and "acled" not in VOCAB.lower(), f"Refusing to train the public/HF release model on a path containing acled: {PANEL} / {VOCAB}"
FIELDS = ["type_of_violence", "side_a", "side_b", "deaths_bin", "month_offset"]
NF = 5
MAX_SEQ_LEN = 200
TAG = os.environ.get("UCDP_TAG", "ucdp_S")
DIM = int(os.environ.get("UCDP_DIM", "64"))
NL = int(os.environ.get("UCDP_LAYERS", "2"))
NH = int(os.environ.get("UCDP_HEADS", "4"))
SEED = int(os.environ.get("UCDP_SEED", "123"))
EPOCHS = int(os.environ.get("UCDP_EPOCHS", "100"))
PATIENCE = int(os.environ.get("UCDP_PATIENCE", "12"))
BATCH = int(os.environ.get("UCDP_BATCH", "256"))
LR = float(os.environ.get("UCDP_LR", "1e-3"))
WARMUP = int(os.environ.get("UCDP_WARMUP", "200"))
CLIP = float(os.environ.get("UCDP_CLIP", "1.0"))
MAX_ROWS = int(os.environ.get("UCDP_MAX_ROWS", "0"))
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] TAG={TAG} dim={DIM} layers={NL} heads={NH} seed={SEED} lr={LR} "
      f"warmup={WARMUP} clip={CLIP} dev={DEVICE}", flush=True)

vocab = json.load(open(VOCAB))
vocab_sizes = {f: max(vocab[f].values()) + 1 for f in FIELDS}   # ids 1..K, 0 pad -> size K+1
MASK_IDX = {f: vocab_sizes[f] for f in FIELDS}
vsm = {f: vocab_sizes[f] + 1 for f in FIELDS}
print(f"[data] vocab sizes (no mask): {vocab_sizes}", flush=True)

panel = pd.read_parquet(PANEL, columns=["split", "event_sequence"])
if MAX_ROWS > 0 and len(panel) > MAX_ROWS:
    panel = panel.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
train_df = panel[panel["split"] == "train"].reset_index(drop=True)
val_df = panel[panel["split"] == "val"].reset_index(drop=True)
print(f"[data] train={len(train_df):,} val={len(val_df):,}", flush=True)
assert len(train_df) and len(val_df)

class DS(Dataset):
    def __init__(self, df):
        self.seqs = []
        self.lengths = []
        for s in df["event_sequence"].values:
            q = json.loads(s) if isinstance(s, str) else []
            q = q[-MAX_SEQ_LEN:]
            self.seqs.append(q)
            self.lengths.append(len(q))
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        q = self.seqs[i]
        t = torch.zeros(MAX_SEQ_LEN, NF, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(q):
            for f in range(NF):
                t[j, f] = ev[f]
            m[j] = True
        return t, m

def collate(batch, mask_prob=0.15, replace_prob=0.8, random_prob=0.1):
    seqs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    B, L, _ = seqs.shape
    targets = torch.full((B, L, NF), -100, dtype=torch.long)
    inputs = seqs.clone()
    for b in range(B):
        valid = torch.nonzero(masks[b], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        nmask = max(1, int(round(valid.numel() * mask_prob)))
        chosen = valid[torch.randperm(valid.numel())[:nmask]]
        for pos in chosen.tolist():
            for f in range(NF):
                targets[b, pos, f] = seqs[b, pos, f]
            r = torch.rand(1).item()
            if r < replace_prob:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = MASK_IDX[fld]
            elif r < replace_prob + random_prob:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = torch.randint(1, vocab_sizes[fld], (1,)).item()
    return inputs, masks, targets

class UcdpEncoder(nn.Module):
    def __init__(self, vsm, fields, d, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.fields = list(fields)
        self.embed_dim = d
        self.emb = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({f: nn.Linear(d, vsm[f] - 1) for f in fields})

    def _encode(self, seq, mask):
        B = seq.size(0)
        e = torch.cat([self.emb[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        e = torch.cat([cls, e], dim=1)
        cm = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mwc = torch.cat([cm, mask], dim=1)
        return self.transformer(e, src_key_padding_mask=~mwc)

    def encode_sequence(self, seq, mask):
        out = self._encode(seq, mask)
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(dim=1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

    def forward_mlm(self, seq, mask):
        oe = self._encode(seq, mask)[:, 1:, :]
        return {f: self.heads[f](oe) for f in self.fields}

train_ds, val_ds = DS(train_df), DS(val_df)
g = torch.Generator(); g.manual_seed(SEED)
w = 1.0 + np.log1p(np.asarray(train_ds.lengths, dtype=np.float64))
sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                num_samples=len(train_ds), replacement=True, generator=g)
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0,
                          pin_memory=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=collate)

model = UcdpEncoder(vsm, FIELDS, DIM, NH, NL).to(DEVICE)
print(f"[model] params={sum(p.numel() for p in model.parameters()):,}", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)
spe = max(1, len(train_loader)); tot = max(1, EPOCHS * spe)
def lrl(st):
    if st < WARMUP: return (st + 1) / WARMUP
    pr = (st - WARMUP) / max(1, tot - WARMUP)
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, pr)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lrl)

def run_epoch(loader, train):
    model.train() if train else model.eval()
    ls, npos = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, mask, targets in loader:
            inputs, mask, targets = inputs.to(DEVICE), mask.to(DEVICE), targets.to(DEVICE)
            logits = model.forward_mlm(inputs, mask)
            loss, npb = 0.0, 0
            for f, fld in enumerate(FIELDS):
                tgt = targets[:, :, f]
                l = F.cross_entropy(logits[fld].reshape(-1, logits[fld].size(-1)),
                                    tgt.reshape(-1), ignore_index=-100, reduction="sum")
                npb += (tgt != -100).sum().item()
                loss = loss + l
            if npb == 0:
                continue
            loss = loss / npb
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
                sched.step()
            ls += loss.item() * npb
            npos += npb
    return ls / max(npos, 1)

best, pat, log = float("inf"), 0, []
best_path = f"{R}/data_model/ucdp_repr_{TAG}.pt"
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    print(f"[epoch {ep:3d}] train={tr:.4f} val={va:.4f} {time.time()-t0:.0f}s", flush=True)
    log.append({"tag": TAG, "epoch": ep, "train_loss": tr, "val_loss": va})
    if va < best:
        best, pat = va, 0
        torch.save(model.state_dict(), best_path)
        print(f"[epoch {ep:3d}] new best -> {best_path}", flush=True)
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"[stop] early stop at {ep} (best={best:.4f})", flush=True)
            break
    pd.DataFrame(log).to_csv(f"{R}/logs/ucdp_log_{TAG}.csv", index=False)
print(f"[done] TAG={TAG} best_val={best:.4f} -> {best_path}", flush=True)

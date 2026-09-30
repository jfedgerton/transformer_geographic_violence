import os, sys, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
PANEL_PATH = f"{R}/data_intermediate/baseline_features.parquet"
VOCAB_PATH = f"{R}/data_intermediate/vocab.json"
MODEL_DIR  = f"{R}/data_model"
LOG_DIR    = f"{R}/logs"
FIELDS = ["event_type", "sub_event_type", "actor1", "actor2",
          "actor1_type", "actor2_type", "month_offset"]
NUM_FIELDS = 7
MAX_SEQ_LEN = 200
DROPOUT = 0.1
CAPACITY = {
    "S":  dict(embed_dim=64,  n_layers=2, n_heads=4),
    "M":  dict(embed_dim=128, n_layers=4, n_heads=8),
    "L":  dict(embed_dim=256, n_layers=6, n_heads=8),
    "XL": dict(embed_dim=512, n_layers=8, n_heads=16),
}
HOLDOUT_REGIONS = {"LatAm": (-56, 33, -118, -34), "ME": (12, 42, 34, 63)}
AFRICA_BBOX = (-36.0, 38.0, -19.0, 52.0)
HOLDOUT_COUNTRIES = ["Nigeria", "Democratic Republic of Congo", "Myanmar",
                     "Philippines", "Ukraine", "India", "Pakistan", "Indonesia"]
TAG      = os.environ["REPR_TAG"]
CAP      = os.environ.get("REPR_CAPACITY", "S")
VARIANT  = os.environ.get("REPR_VARIANT", "full")
SEED     = int(os.environ.get("REPR_SEED", "123"))
SAMPLE_W = os.environ.get("REPR_SAMPLE_WEIGHT", "conflict")
BUFFER   = float(os.environ.get("REPR_BUFFER_DEG", "0.0"))
MAX_EP   = int(os.environ.get("REPR_MAX_EPOCHS", "60"))
PATIENCE = int(os.environ.get("REPR_PATIENCE", "8"))
BATCH    = int(os.environ.get("REPR_BATCH", "256"))
LR       = float(os.environ.get("REPR_LR", "1e-3"))
MAX_ROWS = int(os.environ.get("REPR_MAX_ROWS", "0"))
assert CAP in CAPACITY, f"bad REPR_CAPACITY {CAP}"
assert VARIANT in ("full", "holdout", "africa"), f"bad REPR_VARIANT {VARIANT}"
DIMS = CAPACITY[CAP]
EMBED_DIM, N_LAYERS, N_HEADS = DIMS["embed_dim"], DIMS["n_layers"], DIMS["n_heads"]
print(f"[cfg] TAG={TAG} CAP={CAP} dims={DIMS} VARIANT={VARIANT} SEED={SEED} "
      f"sampler={SAMPLE_W} buffer={BUFFER} max_ep={MAX_EP} batch={BATCH} lr={LR} "
      f"max_rows={MAX_ROWS}", flush=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] device={DEVICE}", flush=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
with open(VOCAB_PATH) as f:
    vocabs = json.load(f)
vocab_sizes = {k: max(v.values()) + 1 for k, v in vocabs.items()}
for fld in FIELDS:
    if fld not in vocab_sizes:
        raise KeyError(f"vocab missing field '{fld}'")
MASK_IDX = {k: vocab_sizes[k] for k in vocab_sizes}
vsm = {k: vocab_sizes[k] + 1 for k in vocab_sizes}   # +1 mask slot
print(f"[data] vocab sizes (no mask): {vocab_sizes}", flush=True)
import pyarrow.parquet as pq
schema_names = pq.read_schema(PANEL_PATH).names
country_cols = []
if VARIANT == "holdout":
    for c in HOLDOUT_COUNTRIES:
        col = f"country_{c}"
        if col in schema_names:
            country_cols.append(col)
        else:
            print(f"[WARN] holdout country column missing, skipping: {col}", flush=True)
load_cols = ["grid_id", "year_month", "grid_lat_center", "grid_lon_center",
             "split", "event_sequence"] + country_cols
load_cols = [c for c in dict.fromkeys(load_cols)]  # dedupe, keep order
print(f"[data] loading {len(load_cols)} columns: {load_cols}", flush=True)
panel = pd.read_parquet(PANEL_PATH, columns=load_cols)
print(f"[data] panel rows (global): {len(panel):,}", flush=True)
lat = panel["grid_lat_center"].to_numpy()
lon = panel["grid_lon_center"].to_numpy()
if VARIANT == "africa":
    a0, a1, o0, o1 = AFRICA_BBOX
    keep = (lat >= a0) & (lat <= a1) & (lon >= o0) & (lon <= o1)
    panel = panel[keep].reset_index(drop=True)
    print(f"[filter] africa bbox -> {len(panel):,} rows", flush=True)
elif VARIANT == "holdout":
    drop = np.zeros(len(panel), dtype=bool)
    for name, (a0, a1, o0, o1) in HOLDOUT_REGIONS.items():
        b = BUFFER
        rdrop = ((lat >= a0 - b) & (lat <= a1 + b) &
                 (lon >= o0 - b) & (lon <= o1 + b))
        drop |= rdrop
        print(f"[filter] region holdout {name} (buf={b}) drops {rdrop.sum():,}", flush=True)
    for col in country_cols:
        cdrop = panel[col].to_numpy() == 1
        drop |= cdrop
        print(f"[filter] country holdout {col} drops {cdrop.sum():,}", flush=True)
    panel = panel[~drop].reset_index(drop=True)
    print(f"[filter] holdout variant -> {len(panel):,} rows remain", flush=True)
else:
    print(f"[filter] full variant -> all {len(panel):,} rows", flush=True)
if MAX_ROWS > 0 and len(panel) > MAX_ROWS:
    panel = panel.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
    print(f"[smoke] subsampled to {len(panel):,} rows", flush=True)
train_df = panel[panel["split"] == "train"].reset_index(drop=True)
val_df   = panel[panel["split"] == "val"].reset_index(drop=True)
print(f"[data] train={len(train_df):,}  val={len(val_df):,}", flush=True)
if len(train_df) == 0 or len(val_df) == 0:
    raise RuntimeError("empty train or val after filtering -- check variant/splits")
class EventSequenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = []
        self.lengths = []
        for s in df["event_sequence"].values:
            seq = json.loads(s) if isinstance(s, str) else []
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            self.sequences.append(seq)
            self.lengths.append(len(seq))
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for i, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[i, f] = ev[f]
            m[i] = True
        return t, m
def mlm_collate(batch, mask_prob=0.15, replace_prob=0.8, random_prob=0.1):
    seqs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    B, L, _ = seqs.shape
    targets = torch.full((B, L, NUM_FIELDS), -100, dtype=torch.long)
    inputs = seqs.clone()
    for b in range(B):
        valid = torch.nonzero(masks[b], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        n_mask = max(1, int(round(valid.numel() * mask_prob)))
        chosen = valid[torch.randperm(valid.numel())[:n_mask]]
        for pos in chosen.tolist():
            for f in range(NUM_FIELDS):
                targets[b, pos, f] = seqs[b, pos, f]
            r = torch.rand(1).item()
            if r < replace_prob:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = MASK_IDX[fld]
            elif r < replace_prob + random_prob:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = torch.randint(0, vocab_sizes[fld], (1,)).item()
    return inputs, masks, targets
class EventHistoryEncoder(nn.Module):
    def __init__(self, vsm, embed_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        d = embed_dim
        self.emb_event_type = nn.Embedding(vsm["event_type"], d, padding_idx=0)
        self.emb_sub_event_type = nn.Embedding(vsm["sub_event_type"], d, padding_idx=0)
        self.emb_actor1 = nn.Embedding(vsm["actor1"], d, padding_idx=0)
        self.emb_actor2 = nn.Embedding(vsm["actor2"], d, padding_idx=0)
        self.emb_actor1_type = nn.Embedding(vsm["actor1_type"], d, padding_idx=0)
        self.emb_actor2_type = nn.Embedding(vsm["actor2_type"], d, padding_idx=0)
        self.emb_month_offset = nn.Embedding(vsm["month_offset"], d, padding_idx=0)
        self.field_proj = nn.Linear(NUM_FIELDS * d, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.ln = nn.LayerNorm(embed_dim)
        self.head_event_type = nn.Linear(embed_dim, vsm["event_type"] - 1)
        self.head_sub_event_type = nn.Linear(embed_dim, vsm["sub_event_type"] - 1)
        self.head_actor1 = nn.Linear(embed_dim, vsm["actor1"] - 1)
        self.head_actor2 = nn.Linear(embed_dim, vsm["actor2"] - 1)
        self.head_actor1_type = nn.Linear(embed_dim, vsm["actor1_type"] - 1)
        self.head_actor2_type = nn.Linear(embed_dim, vsm["actor2_type"] - 1)
        self.head_month_offset = nn.Linear(embed_dim, vsm["month_offset"] - 1)
    def _encode(self, seq, mask):
        B = seq.size(0)
        e = torch.cat([
            self.emb_event_type(seq[:, :, 0]), self.emb_sub_event_type(seq[:, :, 1]),
            self.emb_actor1(seq[:, :, 2]), self.emb_actor2(seq[:, :, 3]),
            self.emb_actor1_type(seq[:, :, 4]), self.emb_actor2_type(seq[:, :, 5]),
            self.emb_month_offset(seq[:, :, 6]),
        ], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        e = torch.cat([cls, e], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mwc = torch.cat([cls_mask, mask], dim=1)
        out = self.transformer(e, src_key_padding_mask=~mwc)
        return out
    def forward_mlm(self, seq, mask):
        out = self._encode(seq, mask)
        oe = out[:, 1:, :]
        return {
            "event_type": self.head_event_type(oe),
            "sub_event_type": self.head_sub_event_type(oe),
            "actor1": self.head_actor1(oe),
            "actor2": self.head_actor2(oe),
            "actor1_type": self.head_actor1_type(oe),
            "actor2_type": self.head_actor2_type(oe),
            "month_offset": self.head_month_offset(oe),
        }
print("[data] building datasets (parsing sequences)...", flush=True)
t_build = time.time()
train_ds = EventSequenceDataset(train_df)
val_ds = EventSequenceDataset(val_df)
print(f"[data] datasets built in {time.time()-t_build:.0f}s", flush=True)
g = torch.Generator()
g.manual_seed(SEED)
if SAMPLE_W == "conflict":
    w = 1.0 + np.log1p(np.asarray(train_ds.lengths, dtype=np.float64))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(train_ds), replacement=True, generator=g)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=0, pin_memory=True, collate_fn=mlm_collate)
    print(f"[sampler] conflict-weighted (w in [{w.min():.2f},{w.max():.2f}])", flush=True)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=mlm_collate,
                              generator=g)
    print("[sampler] uniform shuffle", flush=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=mlm_collate)
model = EventHistoryEncoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] params={n_params:,}", flush=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
WARMUP = int(os.environ.get('REPR_WARMUP_STEPS','500'))
CLIP = float(os.environ.get('REPR_CLIP','1.0'))
_spe = max(1, len(train_loader)); _tot = max(1, MAX_EP*_spe)
def _lrl(st):
    if st < WARMUP: return (st+1)/WARMUP
    pr=(st-WARMUP)/max(1,_tot-WARMUP); return 0.1+0.45*(1.0+math.cos(math.pi*min(1.0,pr)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,_lrl)
print(f'[opt] warmup={WARMUP} clip={CLIP} tot={_tot}',flush=True)
best_path = os.path.join(MODEL_DIR, f"repr_model_{TAG}.pt")
ckpt_path = os.path.join(MODEL_DIR, f"ckpt_{TAG}.pt")
log_path = os.path.join(LOG_DIR, f"repr_scaling_log_{TAG}.csv")
start_epoch = 1
best_val = float("inf")
patience_ctr = 0
training_log = []
if os.path.exists(ckpt_path):
    print(f"[resume] found {ckpt_path}, loading...", flush=True)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    start_epoch = ck["epoch"] + 1
    best_val = ck["best_val"]
    patience_ctr = ck["patience_ctr"]
    training_log = ck.get("log", [])
    torch.set_rng_state(ck["torch_rng"].cpu() if hasattr(ck["torch_rng"], "cpu") else ck["torch_rng"])
    np.random.set_state(ck["numpy_rng"])
    print(f"[resume] resuming at epoch {start_epoch} (best_val={best_val:.4f}, "
          f"patience={patience_ctr})", flush=True)
def run_epoch(loader, train):
    model.train() if train else model.eval()
    loss_sum, n_pos = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, mask, targets in loader:
            inputs, mask, targets = inputs.to(DEVICE), mask.to(DEVICE), targets.to(DEVICE)
            logits = model.forward_mlm(inputs, mask)
            loss = 0.0
            npb = 0
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
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                scheduler.step()
            loss_sum += loss.item() * npb
            n_pos += npb
    return loss_sum / max(n_pos, 1)
print(f"[train] up to {MAX_EP} epochs, patience={PATIENCE}", flush=True)
for epoch in range(start_epoch, MAX_EP + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    dt = time.time() - t0
    print(f"[epoch {epoch:3d}] train_mlm={tr:.4f} val_mlm={va:.4f} {dt:.0f}s", flush=True)
    training_log.append({"tag": TAG, "epoch": epoch, "train_loss": tr,
                         "val_loss": va, "elapsed_sec": dt})
    improved = va < best_val
    if improved:
        best_val = va
        patience_ctr = 0
        torch.save(model.state_dict(), best_path)
        print(f"[epoch {epoch:3d}] new best -> {best_path}", flush=True)
    else:
        patience_ctr += 1
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_val": best_val, "patience_ctr": patience_ctr,
                "log": training_log, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state()}, ckpt_path)
    pd.DataFrame(training_log).to_csv(log_path, index=False)
    if patience_ctr >= PATIENCE:
        print(f"[train] early stopping at epoch {epoch} (best_val={best_val:.4f})", flush=True)
        break
print(f"[done] TAG={TAG} best_val={best_val:.4f} best_model={best_path}", flush=True)

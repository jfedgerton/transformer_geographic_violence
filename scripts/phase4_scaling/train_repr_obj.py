"""train_repr_obj.py
=====================
Unified multi-OBJECTIVE + multi-FEATSET representation trainer (M loss/feature grid).
Self-contained (re-implements warmup->cosine + clip-0.5).

OBJECTIVE (env REPR_OBJECTIVE): mlm | autoregressive | contrastive | tversky
FEATSET   (env REPR_FEATSET):   a | b | c | e | f
    a event-only(7)   b +fatalities(8)   c +civ-deaths(8)   e +PRIO   f all+PRIO
Field selection indexes into the 9-field superset tuple:
    0 event_type 1 sub_event_type 2 actor1 3 actor2 4 actor1_type 5 actor2_type
    6 month_offset 7 fatalities 8 deaths_civilians
PRIO (sets e,f): 34 z-scored static covariates per grid cell, projected to embed_dim
and ADDED to the CLS token (pooled objectives) / event embeddings (causal AR).
Nothing positional/geographic enters the event encoder; PRIO is a place-attribute
appendix arm. Defaults (FEATSET=a, vocab.json, baseline_features.parquet) reproduce
the event-only run.
"""
import os, sys, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
MODEL_DIR = f"{R}/data_model"
LOG_DIR   = f"{R}/logs"
MAX_SEQ_LEN = 200
DROPOUT = 0.1
CAPACITY = {
    "S":  dict(embed_dim=64,  n_layers=2, n_heads=4),
    "M":  dict(embed_dim=128, n_layers=4, n_heads=8),
    "L":  dict(embed_dim=256, n_layers=6, n_heads=8),
    "XL": dict(embed_dim=512, n_layers=8, n_heads=16),
}
HOLDOUT_REGIONS = {}  # country-level holdout for the M grid; region-LORO handled separately at L
AFRICA_BBOX = (-36.0, 38.0, -19.0, 52.0)
# Expanded 26-country holdout (2026-06-22, all dummy-verified). US/Canada deliberately
# NOT held out (kept in training). Stratified across region x intensity.
HOLDOUT_COUNTRIES = ["Sudan", "Somalia", "Uganda", "Kenya", "Ethiopia", "South Africa",
                     "Iraq", "Tunisia", "Iran", "Algeria", "Morocco",
                     "Pakistan", "Bangladesh", "Nepal", "Sri Lanka",
                     "Myanmar", "Philippines", "Thailand", "South Korea",
                     "Ukraine", "France", "Germany",
                     "Mexico", "Venezuela", "Colombia", "Chile"]
# 9-field superset order and per-FEATSET selection
SUPER_FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type",
                "actor2_type", "month_offset", "fatalities", "deaths_civilians"]
FEATSET_SEL = {"a": [0,1,2,3,4,5,6], "b": [0,1,2,3,4,5,6,7], "c": [0,1,2,3,4,5,6,8],
               "e": [0,1,2,3,4,5,6], "f": [0,1,2,3,4,5,6,7,8]}
FEATSET_PRIO = {"a": False, "b": False, "c": False, "e": True, "f": True}

E = os.environ.get
TAG       = os.environ["REPR_TAG"]
CAP       = E("REPR_CAPACITY", "M")
VARIANT = os.environ["REPR_VARIANT"]  # 2026-10-19: no default; every run must set it explicitly
SEED      = int(E("REPR_SEED", "123"))
SAMPLE_W  = E("REPR_SAMPLE_WEIGHT", "conflict")
BUFFER    = float(E("REPR_BUFFER_DEG", "0.0"))
MAX_EP    = int(E("REPR_MAX_EPOCHS", "30"))
PATIENCE  = int(E("REPR_PATIENCE", "15"))
BATCH     = int(E("REPR_BATCH", "256"))
LR        = float(E("REPR_LR", "5e-4"))
MAX_ROWS  = int(E("REPR_MAX_ROWS", "0"))
WARMUP    = int(E("REPR_WARMUP_STEPS", "2000"))
CLIP      = float(E("REPR_CLIP", "0.5"))
MIN_DELTA = float(E("REPR_MIN_DELTA", "1e-3"))  # val must improve by > this to reset patience
OBJECTIVE = E("REPR_OBJECTIVE", "mlm")
FEATSET   = E("REPR_FEATSET", "a")
PANEL_PATH = E("REPR_PANEL", f"{R}/data_intermediate/baseline_features.parquet")
VOCAB_PATH = E("REPR_VOCAB", f"{R}/data_intermediate/vocab.json")
PRIO_PATH  = E("REPR_PRIO", f"{R}/data_intermediate/priogrid_static.parquet")
LORO = E("REPR_LORO","")
LORO_COUNTRIES = []
if LORO:
    import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from regions import REGION_OF, REGIONS
    assert LORO in REGIONS, f"bad REPR_LORO {LORO}"
    LORO_COUNTRIES = [c for c,r in REGION_OF.items() if r == LORO]
    print(f"[loro] hold out region {LORO}: {len(LORO_COUNTRIES)} countries", flush=True)
TV_A = float(E("TVERSKY_ALPHA", "0.3")); TV_B = float(E("TVERSKY_BETA", "0.7"))
TEMP = float(E("CONTRAST_TEMP", "0.1")); AUG_P = float(E("CONTRAST_AUG_P", "0.15"))
assert CAP in CAPACITY and VARIANT in ("full","holdout","africa")
assert not (LORO and VARIANT != "full"), "2026-10-19: LORO runs must use REPR_VARIANT=full (only the target region is excluded)"
DRYRUN = E("REPR_DRYRUN", "0") == "1"
assert OBJECTIVE in ("mlm","autoregressive","contrastive","tversky")
assert FEATSET in FEATSET_SEL, f"bad FEATSET {FEATSET}"
SEL = FEATSET_SEL[FEATSET]
FIELDS = [SUPER_FIELDS[i] for i in SEL]
NUM_FIELDS = len(SEL)
USE_PRIO = FEATSET_PRIO[FEATSET]
DIMS = CAPACITY[CAP]
EMBED_DIM, N_LAYERS, N_HEADS = DIMS["embed_dim"], DIMS["n_layers"], DIMS["n_heads"]
print(f"[cfg] TAG={TAG} CAP={CAP} OBJ={OBJECTIVE} FEATSET={FEATSET} fields={FIELDS} "
      f"prio={USE_PRIO} VARIANT={VARIANT} SEED={SEED} lr={LR} max_ep={MAX_EP}", flush=True)

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

with open(VOCAB_PATH) as f:
    vocabs = json.load(f)
vocab_sizes = {k: max(v.values()) + 1 for k, v in vocabs.items()}
for fld in FIELDS:
    if fld not in vocab_sizes:
        raise KeyError(f"vocab missing field '{fld}' (use vocab_super.json for b/c/f)")
MASK_IDX = {k: vocab_sizes[k] for k in FIELDS}
vsm = {k: vocab_sizes[k] + 1 for k in FIELDS}
print(f"[data] selected vocab sizes: { {k: vocab_sizes[k] for k in FIELDS} }", flush=True)

# PRIO static lookup
prio_lookup = None; STATIC_DIM = 1
if USE_PRIO:
    ps = pd.read_parquet(PRIO_PATH)
    static_cols = [c for c in ps.columns if c != "grid_id"]
    STATIC_DIM = len(static_cols)
    prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], dtype=np.float32)
                   for r in ps.itertuples(index=False)}
    print(f"[prio] loaded {len(prio_lookup):,} cells x {STATIC_DIM} static vars", flush=True)

import pyarrow.parquet as pq
schema_names = pq.read_schema(PANEL_PATH).names
country_cols = []
if VARIANT == "holdout":
    for c in HOLDOUT_COUNTRIES:
        col = f"country_{c}"
        if col in schema_names:
            country_cols.append(col)
        else:
            print(f"[WARN] holdout country col missing: {col}", flush=True)
load_cols = ["grid_id", "year_month", "grid_lat_center", "grid_lon_center",
             "split", "event_sequence", "violence_next_month", "country"] + country_cols
load_cols = [c for c in dict.fromkeys(load_cols)]
panel = pd.read_parquet(PANEL_PATH, columns=load_cols)
print(f"[data] panel rows: {len(panel):,}", flush=True)
lat = panel["grid_lat_center"].to_numpy(); lon = panel["grid_lon_center"].to_numpy()
if VARIANT == "africa":
    a0, a1, o0, o1 = AFRICA_BBOX
    panel = panel[(lat >= a0)&(lat <= a1)&(lon >= o0)&(lon <= o1)].reset_index(drop=True)
elif VARIANT == "holdout":
    drop = np.zeros(len(panel), dtype=bool)
    for name, (a0, a1, o0, o1) in HOLDOUT_REGIONS.items():
        b = BUFFER
        drop |= ((lat >= a0-b)&(lat <= a1+b)&(lon >= o0-b)&(lon <= o1+b))
    for col in country_cols:
        drop |= panel[col].to_numpy() == 1
    panel = panel[~drop].reset_index(drop=True)
    print(f"[filter] holdout -> {len(panel):,} rows", flush=True)
# 2026-10-19: the target-region exclusion applies under every variant (it was nested inside the holdout branch).
if LORO:
    n_before = len(panel)
    panel = panel[~panel["country"].isin(LORO_COUNTRIES)].reset_index(drop=True)
    print(f"[filter] loro {LORO} -> dropped {n_before - len(panel):,} rows, {len(panel):,} remain", flush=True)
    assert not panel["country"].isin(LORO_COUNTRIES).any(), "target-region rows remain"
if MAX_ROWS > 0 and len(panel) > MAX_ROWS:
    panel = panel.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
    print(f"[smoke] subsampled to {len(panel):,}", flush=True)
train_df = panel[panel["split"] == "train"].reset_index(drop=True)
val_df   = panel[panel["split"] == "val"].reset_index(drop=True)
print(f"[data] train={len(train_df):,} val={len(val_df):,}", flush=True)
# 2026-10-19 manifest: record exactly what this encoder is pretrained on, and fail loudly on a wrong data scope.
import hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF as _RO
_ctry = panel["country"].astype(str).value_counts()
_present = set(_ctry.index)
_regions_present = sorted({_RO.get(c, "UNMAPPED") for c in _present})
manifest = dict(tag=TAG, variant=VARIANT, loro=LORO, capacity=CAP, objective=OBJECTIVE, featset=FEATSET,
                seed=SEED, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA, max_epochs=MAX_EP, max_rows=MAX_ROWS,
                panel=PANEL_PATH, vocab=VOCAB_PATH, n_rows=int(len(panel)), n_train=int(len(train_df)),
                n_val=int(len(val_df)),
                train_months=[str(train_df["year_month"].min()), str(train_df["year_month"].max())],
                val_months=[str(val_df["year_month"].min()), str(val_df["year_month"].max())],
                n_countries=int(len(_present)), regions_present=_regions_present,
                rows_by_region={r: int(sum(int(_ctry[c]) for c in _present if _RO.get(c, "UNMAPPED") == r)) for r in _regions_present},
                unmapped_countries=sorted(c for c in _present if c not in _RO),
                holdout26_present=sorted(c for c in HOLDOUT_COUNTRIES if c in _present),
                countries=sorted(_present),
                script_md5=hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest())
print("[manifest] " + json.dumps({k: v for k, v in manifest.items() if k != "countries"}), flush=True)
assert manifest["train_months"][1] <= "2020-12" and manifest["val_months"] == ["2021-01", "2021-12"], "time split changed"
if LORO:
    assert LORO not in _regions_present, f"target region {LORO} present in pretraining"
if VARIANT == "full":
    _expected26 = [c for c in HOLDOUT_COUNTRIES if not (LORO and _RO.get(c) == LORO)]
    _missing = [c for c in _expected26 if c not in _present]
    assert not _missing, f"full variant but these countries are missing: {_missing}"
_mpath = os.path.join(LOG_DIR if DRYRUN else MODEL_DIR, f"{'dryrun_' if DRYRUN else ''}repr_model_{TAG}.manifest.json")
with open(_mpath, "w") as _f:
    json.dump(manifest, _f, indent=1)
print(f"[manifest] wrote {_mpath}", flush=True)
if DRYRUN:
    print("[dryrun] counts and manifest only; exiting before training", flush=True)
    sys.exit(0)
assert not os.path.exists(os.path.join(MODEL_DIR, f"repr_model_{TAG}.pt")), "checkpoint exists; refusing to overwrite"
if len(train_df) == 0 or len(val_df) == 0:
    raise RuntimeError("empty train or val")


class SeqDS(Dataset):
    def __init__(self, df):
        self.seqs, self.lengths, self.labels, self.static = [], [], [], []
        for s, lb, gid in zip(df["event_sequence"].values,
                              df["violence_next_month"].values, df["grid_id"].values):
            seq = json.loads(s) if isinstance(s, str) else []
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            self.seqs.append(seq); self.lengths.append(len(seq))
            self.labels.append(float(lb) if lb == lb else 0.0)
            if USE_PRIO:
                self.static.append(prio_lookup.get(int(gid), np.zeros(STATIC_DIM, np.float32)))
            else:
                self.static.append(np.zeros(1, np.float32))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for i, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[i, f] = ev[SEL[f]]
            m[i] = True
        return (t, m, torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.from_numpy(self.static[idx]))


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]))


def apply_mlm_mask(seqs, masks):
    B, L, _ = seqs.shape
    targets = torch.full((B, L, NUM_FIELDS), -100, dtype=torch.long, device=seqs.device)
    inputs = seqs.clone()
    for b in range(B):
        valid = torch.nonzero(masks[b], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        n_mask = max(1, int(round(valid.numel() * 0.15)))
        chosen = valid[torch.randperm(valid.numel(), device=seqs.device)[:n_mask]]
        for pos in chosen.tolist():
            for f in range(NUM_FIELDS):
                targets[b, pos, f] = seqs[b, pos, f]
            r = torch.rand(1).item()
            if r < 0.8:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = MASK_IDX[fld]
            elif r < 0.9:
                for f, fld in enumerate(FIELDS):
                    inputs[b, pos, f] = torch.randint(0, vocab_sizes[fld], (1,)).item()
    return inputs, targets


def augment_dropout(masks, p):
    drop = (torch.rand_like(masks, dtype=torch.float) < p) & masks
    m2 = masks & ~drop
    empty = (m2.sum(dim=1) == 0) & (masks.sum(dim=1) > 0)
    if empty.any():
        for b in torch.nonzero(empty, as_tuple=False).squeeze(-1).tolist():
            m2[b, torch.nonzero(masks[b], as_tuple=False).squeeze(-1)[0]] = True
    return m2


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, dropout, fields, use_prio, static_dim):
        super().__init__()
        self.fields = fields
        self.use_prio = use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        if use_prio:
            self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({f: nn.Linear(d, vsm[f] - 1) for f in fields})
        self.proj_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # contrastive
        self.tversky_head = nn.Linear(d, 1)  # forecasting head

    def _embed(self, seq):
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        return self.proj_dropout(self.field_proj(e))

    def encode_cls(self, seq, mask, static):
        B = seq.size(0)
        e = self._embed(seq)
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio:
            cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        return self.transformer(e, src_key_padding_mask=~mwc)

    def forward_mlm(self, seq, mask, static):
        oe = self.encode_cls(seq, mask, static)[:, 1:, :]
        return {f: self.heads[f](oe) for f in self.fields}

    def forward_ar(self, seq, mask, static):
        e = self._embed(seq)
        if self.use_prio:
            e = e + self.prio_proj(static).unsqueeze(1)
        L = seq.size(1)
        am = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        out = self.transformer(e, mask=am, src_key_padding_mask=~mask)
        return {f: self.heads[f](out) for f in self.fields}

    def cls_vec(self, seq, mask, static):
        return self.ln(self.encode_cls(seq, mask, static)[:, 0, :])


def per_field_ce(logits, targets):
    loss, npos = 0.0, 0
    for f, fld in enumerate(FIELDS):
        tgt = targets[:, :, f]
        loss = loss + F.cross_entropy(logits[fld].reshape(-1, logits[fld].size(-1)),
                                      tgt.reshape(-1), ignore_index=-100, reduction="sum")
        npos += (tgt != -100).sum().item()
    return loss, npos


def nt_xent(z1, z2, temp):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0); n = z1.size(0)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-1e9)
    tgt = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return F.cross_entropy(sim, tgt)


def soft_tversky(logit, y, a, b, eps=1e-6):
    p = torch.sigmoid(logit)
    tp = (p*y).sum(); fp = (p*(1-y)).sum(); fn = ((1-p)*y).sum()
    return 1.0 - (tp + eps) / (tp + a*fp + b*fn + eps)


print("[data] building datasets...", flush=True)
t0 = time.time()
train_ds, val_ds = SeqDS(train_df), SeqDS(val_df)
print(f"[data] built in {time.time()-t0:.0f}s", flush=True)
g = torch.Generator(); g.manual_seed(SEED)
if SAMPLE_W == "conflict":
    w = 1.0 + np.log1p(np.asarray(train_ds.lengths, dtype=np.float64))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(train_ds), replacement=True, generator=g)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0,
                              pin_memory=True, collate_fn=collate)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=g, num_workers=0,
                              pin_memory=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=collate)

model = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
print(f"[model] params={sum(p.numel() for p in model.parameters()):,}", flush=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
total_steps = MAX_EP * max(1, len(train_loader))


def lr_lambda(step):
    if step < WARMUP:
        return float(step + 1) / float(max(1, WARMUP))
    prog = float(step - WARMUP) / float(max(1, total_steps - WARMUP))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
best_path = os.path.join(MODEL_DIR, f"repr_model_{TAG}.pt")
log_path = os.path.join(LOG_DIR, f"repr_obj_log_{TAG}.csv")
best_val = float("inf"); patience_ctr = 0; training_log = []


def run_epoch(loader, train):
    model.train() if train else model.eval()
    loss_sum, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, masks, labels, static in loader:
            seqs, masks = seqs.to(DEVICE), masks.to(DEVICE)
            labels, static = labels.to(DEVICE), static.to(DEVICE)
            if OBJECTIVE == "mlm":
                inputs, targets = apply_mlm_mask(seqs, masks)
                l, npos = per_field_ce(model.forward_mlm(inputs, masks, static), targets)
                if npos == 0:
                    continue
                loss = l / npos; bs = npos
            elif OBJECTIVE == "autoregressive":
                logits = model.forward_ar(seqs, masks, static)
                nxt_valid = masks[:, 1:]
                shifted = seqs[:, 1:, :].clone()
                shifted[~nxt_valid] = -100
                tgt = torch.full((seqs.size(0), seqs.size(1)-1, NUM_FIELDS), -100,
                                 dtype=torch.long, device=DEVICE)
                tgt[:] = shifted
                logits = {k: v[:, :-1, :] for k, v in logits.items()}
                l, npos = per_field_ce(logits, tgt)
                if npos == 0:
                    continue
                loss = l / npos; bs = npos
            elif OBJECTIVE == "contrastive":
                z1 = model.proj_head(model.cls_vec(seqs, augment_dropout(masks, AUG_P), static))
                z2 = model.proj_head(model.cls_vec(seqs, augment_dropout(masks, AUG_P), static))
                loss = nt_xent(z1, z2, TEMP); bs = seqs.size(0)
            else:  # tversky
                logit = model.tversky_head(model.cls_vec(seqs, masks, static)).squeeze(-1)
                loss = soft_tversky(logit, labels, TV_A, TV_B); bs = seqs.size(0)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step(); scheduler.step()
            loss_sum += loss.item() * bs; n += bs
    return loss_sum / max(n, 1)


print(f"[train] OBJ={OBJECTIVE} FEATSET={FEATSET} up to {MAX_EP} ep, patience={PATIENCE}", flush=True)
for epoch in range(1, MAX_EP + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    print(f"[epoch {epoch:3d}] train={tr:.4f} val={va:.4f} lr={scheduler.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
    training_log.append({"tag": TAG, "obj": OBJECTIVE, "featset": FEATSET, "epoch": epoch,
                         "train_loss": tr, "val_loss": va, "lr": scheduler.get_last_lr()[0]})
    if va < best_val - MIN_DELTA:
        best_val = va; patience_ctr = 0
        torch.save(model.state_dict(), best_path)
        print(f"[epoch {epoch:3d}] new best -> {best_path}", flush=True)
    else:
        patience_ctr += 1
    pd.DataFrame(training_log).to_csv(log_path, index=False)
    if patience_ctr >= PATIENCE:
        print(f"[train] early stop at epoch {epoch} (best_val={best_val:.4f})", flush=True)
        break
print(f"[done] TAG={TAG} OBJ={OBJECTIVE} FEATSET={FEATSET} best_val={best_val:.4f}", flush=True)

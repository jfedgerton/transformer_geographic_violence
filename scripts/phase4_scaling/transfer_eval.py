"""transfer_eval.py -- Phase 4 transfer skill per checkpoint.
Mechanism A: frozen embeddings -> xgb -> next-month incidence ROC/PR on held-out unit.
Mechanism B: low-data fine-tune curve -- fine-tune encoder on {1,5,10,25,100}% of the
held-out unit's train labels vs an identical from-scratch model. THE headline figure.
Units: 2 held-out regions (LatAm, ME) + 8 held-out countries. Baselines: persistence,
africa-specialist (flagged where africa saw the unit: Nigeria, DRC).
Env: EVAL_CAP S|M|L|XL, EVAL_SEED, EVAL_MECH A|B|both, EVAL_FRACS, EVAL_FT_EPOCHS,
EVAL_FT_MAX_ROWS, EVAL_UNIT_MAX_ROWS (smoke), EVAL_UNITS (comma list, smoke).
Out: outputs/transfer_eval_{CAP}_s{SEED}.csv  (rewritten after each unit)
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
CAP = os.environ.get("EVAL_CAP", "S")
SEED = int(os.environ.get("EVAL_SEED", "123"))
MECH = os.environ.get("EVAL_MECH", "both")
FRACS = [float(x) for x in os.environ.get("EVAL_FRACS", "0.01,0.05,0.10,0.25,1.0").split(",")]
FT_EPOCHS = int(os.environ.get("EVAL_FT_EPOCHS", "6"))
FT_MAX_ROWS = int(os.environ.get("EVAL_FT_MAX_ROWS", "150000"))
UNIT_MAX_ROWS = int(os.environ.get("EVAL_UNIT_MAX_ROWS", "0"))
ONLY_UNITS = [u for u in os.environ.get("EVAL_UNITS", "").split(",") if u]
BATCH = 512
FT_BATCH = 128
MAX_SEQ_LEN = 200
NUM_FIELDS = 7
FIELDS = ["event_type", "sub_event_type", "actor1", "actor2",
          "actor1_type", "actor2_type", "month_offset"]
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
HOLDOUT_REGIONS = {"LatAm": (-56, 33, -118, -34), "ME": (12, 42, 34, 63)}
HOLDOUT_COUNTRIES = ["Nigeria", "Democratic Republic of Congo", "Myanmar",
                     "Philippines", "Ukraine", "India", "Pakistan", "Indonesia"]
AFRICA_SAW = {"Nigeria", "Democratic Republic of Congo"}
TRAIN_END = "2020-12"
TEST_START = "2022-01"
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[CAP]
CKPT_HOLDOUT = f"{R}/data_model/repr_model_{CAP}_holdout_s{SEED}.pt"
CKPT_AFRICA = f"{R}/data_model/repr_model_{CAP}_africa_s{SEED}.pt"
OUT = f"{R}/outputs/transfer_eval_{CAP}_s{SEED}.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
print(f"[cfg] CAP={CAP} SEED={SEED} MECH={MECH} fracs={FRACS} ft_ep={FT_EPOCHS} dev={DEVICE}", flush=True)
assert os.path.exists(CKPT_HOLDOUT), f"missing checkpoint {CKPT_HOLDOUT}"
HAVE_AFRICA = os.path.exists(CKPT_AFRICA)

vocabs = json.load(open(f"{R}/data_intermediate/vocab.json"))
vocab_sizes = {k: max(v.values()) + 1 for k, v in vocabs.items()}
vsm = {k: vocab_sizes[k] + 1 for k in vocab_sizes}

class EventHistoryEncoder(nn.Module):
    def __init__(self, vsm, embed_dim, n_heads, n_layers, dropout=0.1):
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
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=embed_dim * 4, dropout=dropout,
                                           batch_first=True)
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

    def encode_sequence(self, seq, mask):
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
        cm = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mwc = torch.cat([cm, mask], dim=1)
        out = self.transformer(e, src_key_padding_mask=~mwc)
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(dim=1) == 0).unsqueeze(1)
        z = torch.where(pad, torch.zeros_like(z), z)
        return z

class SeqDS(Dataset):
    def __init__(self, seq_strings, labels=None):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[j, f] = ev[f]
            m[j] = True
        y = 0.0 if self.labels is None else self.labels[i]
        return t, m, torch.tensor(y)

def load_enc(ckpt):
    m = EventHistoryEncoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def extract(model, seq_strings):
    ld = DataLoader(SeqDS(seq_strings), batch_size=BATCH, shuffle=False, num_workers=0)
    Z = []
    with torch.no_grad():
        for t, m, _ in ld:
            Z.append(model.encode_sequence(t.to(DEVICE), m.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, axis=0)

def xgb_eval(Xtr, ytr, Xte, yte):
    mdl = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
                     "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED},
                    xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
    p = mdl.predict(xgb.DMatrix(Xte))
    return roc_auc_score(yte, p), average_precision_score(yte, p)

class Clf(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.embed_dim, 1)
    def forward(self, seq, mask):
        return self.head(self.enc.encode_sequence(seq, mask)).squeeze(-1)

def finetune_eval(seqs_tr, y_tr, seqs_te, y_te, ckpt, lr):
    enc = EventHistoryEncoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS).to(DEVICE)
    if ckpt is not None:
        enc.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    clf = Clf(enc).to(DEVICE)
    pos = float(max(y_tr.sum(), 1.0))
    neg = float(len(y_tr)) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(neg / pos, 1.0)], device=DEVICE))
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    ld = DataLoader(SeqDS(seqs_tr, y_tr), batch_size=FT_BATCH, shuffle=True, num_workers=0)
    clf.train()
    for ep in range(FT_EPOCHS):
        for t, m, y in ld:
            t, m, y = t.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            loss = crit(clf(t, m), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
    clf.eval()
    preds = []
    ld_te = DataLoader(SeqDS(seqs_te), batch_size=BATCH, shuffle=False, num_workers=0)
    with torch.no_grad():
        for t, m, _ in ld_te:
            preds.append(clf(t.to(DEVICE), m.to(DEVICE)).cpu().numpy())
    p = np.concatenate(preds)
    roc, pr = roc_auc_score(y_te, p), average_precision_score(y_te, p)
    del clf, enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return roc, pr

# ---- panel ----
country_cols = [f"country_{c}" for c in HOLDOUT_COUNTRIES]
cols = ["grid_id", "year_month", "grid_lat_center", "grid_lon_center",
        "event_sequence", "violence_next_month", "violence_lag1"] + country_cols
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=cols)
print(f"[data] panel rows: {len(panel):,}", flush=True)
lat = panel["grid_lat_center"].to_numpy()
lon = panel["grid_lon_center"].to_numpy()

units = {}
for nm, (a0, a1, o0, o1) in HOLDOUT_REGIONS.items():
    units[nm] = ("region", (lat >= a0) & (lat <= a1) & (lon >= o0) & (lon <= o1))
for c in HOLDOUT_COUNTRIES:
    units[c.replace(" ", "_")] = ("country", panel[f"country_{c}"].to_numpy() == 1)

enc_h = load_enc(CKPT_HOLDOUT)
enc_a = load_enc(CKPT_AFRICA) if HAVE_AFRICA else None
print(f"[model] holdout enc loaded; africa enc: {HAVE_AFRICA}", flush=True)

rows = []
for uname, (utype, sel) in units.items():
    if ONLY_UNITS and uname not in ONLY_UNITS:
        continue
    t0 = time.time()
    d = panel[sel & panel["violence_next_month"].notna()].copy()
    if UNIT_MAX_ROWS > 0 and len(d) > UNIT_MAX_ROWS:
        d = d.sample(n=UNIT_MAX_ROWS, random_state=SEED)
    d = d.sort_values("year_month").reset_index(drop=True)
    trm = (d["year_month"] <= TRAIN_END).to_numpy()
    tem = (d["year_month"] >= TEST_START).to_numpy()
    y = d["violence_next_month"].to_numpy().astype(int)
    ytr, yte = y[trm], y[tem]
    seqs = d["event_sequence"].values
    base = dict(cap=CAP, seed=SEED, unit=uname, unit_type=utype,
                n_train=int(trm.sum()), n_test=int(tem.sum()),
                test_pos=float(yte.mean()) if tem.sum() else np.nan,
                africa_saw_unit=(uname.replace("_", " ") in AFRICA_SAW))
    if trm.sum() < 100 or tem.sum() < 100 or ytr.sum() < 10 or yte.sum() < 10:
        rows.append(dict(base, mech="skip", model="too_few_rows", frac=np.nan,
                         roc=np.nan, pr=np.nan))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[unit {uname}] SKIPPED (too few rows/positives)", flush=True)
        continue
    print(f"[unit {uname}] rows={len(d):,} train={trm.sum():,} test={tem.sum():,} "
          f"pos={yte.mean():.4f}", flush=True)
    if MECH in ("A", "both"):
        zh = extract(enc_h, seqs)
        roc, pr = xgb_eval(zh[trm], ytr, zh[tem], yte)
        rows.append(dict(base, mech="A", model="frozen_holdout", frac=1.0, roc=roc, pr=pr))
        print(f"  frozen_holdout ROC {roc:.4f} PR {pr:.4f}", flush=True)
        if enc_a is not None:
            za = extract(enc_a, seqs)
            roc, pr = xgb_eval(za[trm], ytr, za[tem], yte)
            rows.append(dict(base, mech="A", model="frozen_africa", frac=1.0, roc=roc, pr=pr))
            print(f"  frozen_africa  ROC {roc:.4f} PR {pr:.4f}", flush=True)
        persist = np.nan_to_num(d["violence_lag1"].to_numpy())[tem]
        rows.append(dict(base, mech="A", model="persistence", frac=1.0,
                         roc=roc_auc_score(yte, persist),
                         pr=average_precision_score(yte, persist)))
    if MECH in ("B", "both"):
        tr_idx = np.where(trm)[0]
        pos_idx = tr_idx[y[tr_idx] == 1]
        neg_idx = tr_idx[y[tr_idx] == 0]
        seqs_te = seqs[tem]
        for frac in FRACS:
            npos = int(round(len(pos_idx) * frac))
            nneg = int(round(len(neg_idx) * frac))
            if npos < 5:
                rows.append(dict(base, mech="B", model="too_few_pos", frac=frac,
                                 roc=np.nan, pr=np.nan))
                continue
            sp = rng.choice(pos_idx, size=npos, replace=False)
            sn = rng.choice(neg_idx, size=nneg, replace=False)
            idx = np.concatenate([sp, sn])
            if len(idx) > FT_MAX_ROWS:
                idx = rng.choice(idx, size=FT_MAX_ROWS, replace=False)
            rng.shuffle(idx)
            s_tr, y_s = seqs[idx], y[idx]
            for mdl, ck, lr in [("ft_pretrained", CKPT_HOLDOUT, 1e-4),
                                ("ft_scratch", None, 5e-4)]:
                roc, pr = finetune_eval(s_tr, y_s, seqs_te, yte, ck, lr)
                rows.append(dict(base, mech="B", model=mdl, frac=frac, roc=roc, pr=pr))
                print(f"  {mdl} frac={frac} n={len(idx)} ROC {roc:.4f} PR {pr:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[unit {uname}] done in {time.time()-t0:.0f}s -> {OUT}", flush=True)

print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)

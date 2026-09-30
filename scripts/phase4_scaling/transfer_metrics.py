"""transfer_metrics.py — 3-metric transfer eval of a frozen event-only encoder.

For each held-out unit (2 regions + 8 countries), extract the FROZEN encoder's
embedding and evaluate transfer on three targets:
  1. INCIDENCE  : violence next month (binary)            -> PR-AUC      vs persistence
  2. ONSET(lag1): violence next month where lag1==0       -> PR-AUC      vs base-rate
  3. SEVERITY   : next-month fatality count (sb_fatalities)-> CRPS        vs persistence / own-mean

This is the v1 (M-control) validation. nbr-12 baseline + non-contiguous onset are
deferred to v2 (need the neighbor array from count_crps_v3). Self-contained CRPS
(empirical, Poisson predictive draws). Encoder mirrors train_repr_scaling.py (7-field).

Env: EVAL_CKPT (default M-control pilot s123), EVAL_CAP (default M), EVAL_SEED (123),
     EVAL_OUT (default outputs/transfer_metrics_s{SEED}.csv).
Run via sbatch (GPU). One job per seed.
"""
import os, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
CAP   = os.environ.get("EVAL_CAP", "M")
SEED  = int(os.environ.get("EVAL_SEED", "123"))
CKPT  = os.environ.get("EVAL_CKPT", f"{R}/data_model/repr_model_M_holdout_pilot_lr5e4_s{SEED}.pt")
OUT   = os.environ.get("EVAL_OUT", f"{R}/outputs/transfer_metrics_s{SEED}.csv")
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[CAP]
MAX_SEQ_LEN, NUM_FIELDS, BATCH = 200, 7, 512
TRAIN_END, TEST_START = "2020-12", "2022-01"
HOLDOUT_REGIONS = {"LatAm": (-56, 33, -118, -34), "ME": (12, 42, 34, 63)}
HOLDOUT_COUNTRIES = ["Nigeria", "Democratic Republic of Congo", "Myanmar", "Philippines",
                     "Ukraine", "India", "Pakistan", "Indonesia"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
print(f"[cfg] CAP={CAP} SEED={SEED} ckpt={CKPT} dev={DEVICE}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab.json"))
vsm = {k: max(v.values()) + 2 for k, v in vocabs.items()}


class EventHistoryEncoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = d
        self.emb_event_type = nn.Embedding(vsm["event_type"], d, padding_idx=0)
        self.emb_sub_event_type = nn.Embedding(vsm["sub_event_type"], d, padding_idx=0)
        self.emb_actor1 = nn.Embedding(vsm["actor1"], d, padding_idx=0)
        self.emb_actor2 = nn.Embedding(vsm["actor2"], d, padding_idx=0)
        self.emb_actor1_type = nn.Embedding(vsm["actor1_type"], d, padding_idx=0)
        self.emb_actor2_type = nn.Embedding(vsm["actor2_type"], d, padding_idx=0)
        self.emb_month_offset = nn.Embedding(vsm["month_offset"], d, padding_idx=0)
        self.field_proj = nn.Linear(NUM_FIELDS * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        # MLM heads exist in the checkpoint; define so load_state_dict matches (unused here).
        self.head_event_type = nn.Linear(d, vsm["event_type"] - 1)
        self.head_sub_event_type = nn.Linear(d, vsm["sub_event_type"] - 1)
        self.head_actor1 = nn.Linear(d, vsm["actor1"] - 1)
        self.head_actor2 = nn.Linear(d, vsm["actor2"] - 1)
        self.head_actor1_type = nn.Linear(d, vsm["actor1_type"] - 1)
        self.head_actor2_type = nn.Linear(d, vsm["actor2_type"] - 1)
        self.head_month_offset = nn.Linear(d, vsm["month_offset"] - 1)

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
        out = self.transformer(e, src_key_padding_mask=~torch.cat([cm, mask], dim=1))
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(dim=1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)


class SeqDS(Dataset):
    def __init__(self, seq_strings):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[j, f] = ev[f]
            m[j] = True
        return t, m


def extract(model, seqs):
    ld = DataLoader(SeqDS(seqs), batch_size=BATCH, shuffle=False, num_workers=0)
    Z = []
    with torch.no_grad():
        for t, m in ld:
            Z.append(model.encode_sequence(t.to(DEVICE), m.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, axis=0)


def crps_ens(draws, y, max_pairs=200):
    # draws: (N,S) samples; y: (N,). Empirical CRPS = E|X-y| - 0.5 E|X-X'|.
    y = y[:, None]
    term1 = np.abs(draws - y).mean(axis=1)
    term2 = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(1, 2))
    return term1 - 0.5 * term2


# ---- data ----
country_cols = [f"country_{c}" for c in HOLDOUT_COUNTRIES]
cols = ["grid_id", "year_month", "grid_lat_center", "grid_lon_center", "event_sequence",
        "violence_next_month", "violence_lag1"] + country_cols
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=cols)
# next-month fatality target: pgm_fatalities_target has precomputed sb_fat_next
tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")[
    ["grid_id", "year_month", "sb_fatalities", "sb_fat_next"]]
panel = panel.merge(tgt, on=["grid_id", "year_month"], how="left")
panel["fat_now"] = panel["sb_fatalities"].fillna(0.0)
panel["fat_next"] = panel["sb_fat_next"].fillna(0.0)
print(f"[data] fatality-target join coverage = {float(panel['sb_fat_next'].notna().mean()):.3f}", flush=True)
lat, lon = panel["grid_lat_center"].to_numpy(), panel["grid_lon_center"].to_numpy()

units = {}
for nm, (a0, a1, o0, o1) in HOLDOUT_REGIONS.items():
    units[nm] = (lat >= a0) & (lat <= a1) & (lon >= o0) & (lon <= o1)
for c in HOLDOUT_COUNTRIES:
    units[c.replace(" ", "_")] = panel[f"country_{c}"].to_numpy() == 1

enc = EventHistoryEncoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS).to(DEVICE)
enc.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True)); enc.eval()
print("[model] encoder loaded", flush=True)

rows = []
for uname, sel in units.items():
    d = panel[sel & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
    trm = (d["year_month"] <= TRAIN_END).to_numpy()
    tem = (d["year_month"] >= TEST_START).to_numpy()
    if trm.sum() < 200 or tem.sum() < 100:
        print(f"[unit {uname}] SKIP (too few rows)", flush=True); continue
    Z = extract(enc, d["event_sequence"].values)
    Ztr, Zte = Z[trm], Z[tem]
    vnm = d["violence_next_month"].to_numpy().astype(int)
    lag1 = np.nan_to_num(d["violence_lag1"].to_numpy())
    fnext = d["fat_next"].to_numpy().astype(float)
    fnow = d["fat_now"].to_numpy().astype(float)
    t0 = time.time()

    # 1. INCIDENCE (PR-AUC)
    clf = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                     "colsample_bytree": 0.8, "seed": SEED}, xgb.DMatrix(Ztr, label=vnm[trm]), 300)
    p_inc = clf.predict(xgb.DMatrix(Zte))
    inc_pr = average_precision_score(vnm[tem], p_inc)
    inc_persist = average_precision_score(vnm[tem], lag1[tem])
    rows.append(dict(unit=uname, metric="incidence", model="embedding", score=inc_pr, n=int(tem.sum())))
    rows.append(dict(unit=uname, metric="incidence", model="persistence", score=inc_persist, n=int(tem.sum())))

    # 2. ONSET lag-1 (PR-AUC on the subset with no violence last month)
    onset = tem & (lag1 == 0)
    if onset.sum() > 50 and vnm[onset].sum() >= 5:
        p_ons = clf.predict(xgb.DMatrix(Z[onset]))
        rows.append(dict(unit=uname, metric="onset_lag1", model="embedding",
                         score=average_precision_score(vnm[onset], p_ons), n=int(onset.sum())))
        rows.append(dict(unit=uname, metric="onset_lag1", model="baserate",
                         score=float(vnm[onset].mean()), n=int(onset.sum())))

    # 3. SEVERITY (CRPS) — Poisson predictive draws from xgb count model
    reg = xgb.train({"objective": "count:poisson", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                     "colsample_bytree": 0.8, "seed": SEED}, xgb.DMatrix(Ztr, label=fnext[trm]), 300)
    lam = np.clip(reg.predict(xgb.DMatrix(Zte)), 1e-6, None)
    draws = rng.poisson(lam[:, None], size=(len(lam), 200)).astype(float)
    crps_emb = crps_ens(draws, fnext[tem]).mean()
    crps_persist = np.abs(fnow[tem] - fnext[tem]).mean()                       # last month as point forecast
    own_mean = fnext[trm].mean()
    crps_own = crps_ens(rng.poisson(max(own_mean, 1e-6), size=(tem.sum(), 200)).astype(float), fnext[tem]).mean()
    rows.append(dict(unit=uname, metric="severity_crps", model="embedding", score=crps_emb, n=int(tem.sum())))
    rows.append(dict(unit=uname, metric="severity_crps", model="persistence", score=crps_persist, n=int(tem.sum())))
    rows.append(dict(unit=uname, metric="severity_crps", model="own_mean", score=crps_own, n=int(tem.sum())))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[unit {uname}] inc_pr={inc_pr:.3f}(persist {inc_persist:.3f}) "
          f"crps_emb={crps_emb:.3f}(persist {crps_persist:.3f}) {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[done] wrote {OUT}", flush=True)

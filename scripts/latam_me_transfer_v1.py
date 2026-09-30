"""
latam_me_transfer_v1.py — Zero-shot CROSS-REGION transfer probe.

Question (paper point 3): does the AFRICA-pretrained encoder (phase2a, region-filtered
to Africa, never saw Latin America or the Middle East) produce cell-month embeddings that
predict next-month violence INCIDENCE in LatAm and ME — above a conflict-history baseline
and naive persistence — and how close does it get to the GLOBAL (in-distribution) encoder?

This is frozen-encoder transfer (mechanism A), NO fine-tune. It is the cheap signal check.
The win condition here is NOT beating neighbors-12; it is:
  (a) africa_emb  >  persistence  and  >  chance              -> encoder carries cross-region signal
  (b) africa+base >  baseline                                  -> embeddings ADD over conflict history
  (c) africa_emb  ~  global_emb                                -> transfer ~ as good as in-distribution

Encoders share one vocab (502 actors + mask = 503 rows); Africa is 64-dim, global 128-dim.
Shared structural fields (event_type, sub_event_type, actor1_type, actor2_type, month_offset)
are well-trained and globally consistent; actor-identity rows for non-African actors are
mostly untrained, so the Africa encoder must transfer via the structural/temporal fields.

Run (SLURM, A100):
  sbatch --job-name=tgv_xregion --account=jfe4_cr_default --partition=standard \
    --gres=gpu:a100:1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
    --output=logs/xregion_%j.out --error=logs/xregion_%j.err \
    --wrap="module load anaconda && source venv/bin/activate && python scripts/latam_me_transfer_v1.py"
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
SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIELDS = ["event_type", "sub_event_type", "actor1", "actor2",
          "actor1_type", "actor2_type", "month_offset"]
NUM_FIELDS = 7
MAX_SEQ_LEN = 200
BATCH = 512
np.random.seed(SEED)
torch.manual_seed(SEED)

REGIONS = {"LatAm": (-56, 33, -118, -34), "ME": (12, 42, 34, 63)}
TRAIN_END = "2020-12"
TEST_START = "2022-01"

# ---- vocab ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab.json"))
vocab_sizes = {k: max(v.values()) + 1 for k, v in vocabs.items()}
vsm = {k: vocab_sizes[k] + 1 for k in vocab_sizes}   # +1 mask slot

# ---- encoder (exact architecture from scripts/phase2a/06_train_representation_model.py) ----
class EventHistoryEncoder(nn.Module):
    def __init__(self, vsm, embed_dim, n_heads=4, n_layers=2, dropout=0.1):
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(embed_dim)
        # MLM heads exist in the checkpoint; define so load_state_dict is strict-clean
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
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_with_cls = torch.cat([cls_mask, mask], dim=1)
        out = self.transformer(e, src_key_padding_mask=~mask_with_cls)
        z = self.ln(out[:, 0, :])
        z[mask.sum(dim=1) == 0] = 0.0
        return z


class SeqDS(Dataset):
    def __init__(self, seq_strings):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for i, ev in enumerate(seq):
            for f in range(NUM_FIELDS):
                t[i, f] = ev[f]
            m[i] = True
        return t, m


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def extract(seq_strings, ckpt, embed_dim):
    model = EventHistoryEncoder(vsm, embed_dim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    loader = DataLoader(SeqDS(seq_strings), batch_size=BATCH, shuffle=False,
                        num_workers=0, collate_fn=collate)
    Z = []
    with torch.no_grad():
        for seq, mask in loader:
            Z.append(model.encode_sequence(seq.to(DEVICE), mask.to(DEVICE)).cpu().numpy())
    return np.concatenate(Z, axis=0)


def fit_eval(Xtr, ytr, Xte, yte):
    m = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
                   "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED},
                  xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
    p = m.predict(xgb.DMatrix(Xte))
    return roc_auc_score(yte, p), average_precision_score(yte, p)


# ---- load panel (only needed columns) ----
META = {"grid_id", "year_month", "month_idx", "grid_lat_center", "grid_lon_center",
        "country", "split", "event_sequence", "seq_length"}
TARGETS = {"violence_next_month", "violent_count_next_month", "count_all_next_month"}
import pyarrow.parquet as pq
allcols = pq.read_schema(f"{R}/data_intermediate/baseline_features.parquet").names
bcols = [c for c in allcols if c not in META and c not in TARGETS and not c.endswith("_next_month")]
load_cols = sorted(set(["grid_id", "year_month", "grid_lat_center", "grid_lon_center",
                        "event_sequence", "violence_next_month"] + bcols))
bf = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=load_cols)
# numeric baseline features only
bcols = [c for c in bcols if pd.api.types.is_numeric_dtype(bf[c])]
print(f"baseline feature cols: {len(bcols)}")

rows = []
for name, (a0, a1, o0, o1) in REGIONS.items():
    t0 = time.time()
    d = bf[(bf.grid_lat_center >= a0) & (bf.grid_lat_center <= a1) &
           (bf.grid_lon_center >= o0) & (bf.grid_lon_center <= o1)].copy()
    d = d[d.violence_next_month.notna()].copy()          # drop final month (NaN next-month target)
    d = d.sort_values("year_month").reset_index(drop=True)
    tr_m = (d.year_month <= TRAIN_END).to_numpy()
    te_m = (d.year_month >= TEST_START).to_numpy()
    y = d.violence_next_month.to_numpy().astype(int)
    ytr, yte = y[tr_m], y[te_m]
    print(f"\n=== {name}: rows {len(d)}  train {tr_m.sum()}  test {te_m.sum()}  "
          f"test_pos {yte.mean():.4f} ===", flush=True)

    z_af = extract(d.event_sequence.values, f"{R}/data_model/repr_model_phase2a_seed{SEED}.pt", 64)
    z_gl = extract(d.event_sequence.values, f"{R}/data_model/repr_model_seed{SEED}.pt", 128)
    B = d[bcols].to_numpy(dtype=float)
    persist = np.nan_to_num(d["violence_lag1"].to_numpy()) if "violence_lag1" in d else np.zeros(len(d))

    blocks = {
        "africa_emb":   z_af,
        "global_emb":   z_gl,
        "baseline":     B,
        "africa+base":  np.hstack([z_af, B]),
        "global+base":  np.hstack([z_gl, B]),
    }
    for nm, X in blocks.items():
        roc, pr = fit_eval(X[tr_m], ytr, X[te_m], yte)
        rows.append({"region": name, "model": nm, "roc_auc": roc, "pr_auc": pr})
        print(f"  {nm:12s} ROC {roc:.4f}  PR-AUC {pr:.4f}", flush=True)
    # persistence (violence this month -> violence next month)
    proc = roc_auc_score(yte, persist[te_m]); ppr = average_precision_score(yte, persist[te_m])
    rows.append({"region": name, "model": "persistence", "roc_auc": proc, "pr_auc": ppr})
    print(f"  {'persistence':12s} ROC {proc:.4f}  PR-AUC {ppr:.4f}  ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(f"{R}/outputs/latam_me_transfer_v1.csv", index=False)
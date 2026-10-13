"""transfer_eval_region_ft.py -- LORO low-data fine-tune curve (replaces Table 2's June design).
For held-out region RG: load repr_model_L_contrastive_e_loro_{RG}_s123.pt (pretrained on the
other five regions), fine-tune encoder+head on frac in {1,5,10,25,100}% of RG's TRAIN-period
labels vs an identical from-scratch model; score PR-AUC (and ROC) on RG's TEST period (Mode B).
Mirrors transfer_eval.py Mechanism B exactly (stratified frac subsample, FT_MAX_ROWS cap,
6 epochs, lr 1e-4 pretrained / 5e-4 scratch, BCE pos_weight, grad clip 1.0) but with the
Gen-4 Encoder (featset e, PRIO static covariates) and the LORO split from transfer_eval_region.py.
Env: EVAL_REGION (required), EVAL_TAG (default L_contrastive_e_loro_{RG}_s123), EVAL_FRACS,
EVAL_FT_EPOCHS, EVAL_FT_MAX_ROWS, EVAL_SEED.
Out: outputs/ft_region_{TAG}.csv
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
RG = os.environ["EVAL_REGION"]
SEED = int(os.environ.get("EVAL_SEED", "123"))
TAG = os.environ.get("EVAL_TAG", f"L_contrastive_e_loro_{RG}_s{SEED}")
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
SUFFIX = os.environ.get("EVAL_SUFFIX", "")
assert SUFFIX, "EVAL_SUFFIX required so the 2026-10-06 CSVs are never overwritten"
OUT = f"{R}/outputs/ft_region_{TAG}{SUFFIX}.csv"
ARMS = os.environ.get("EVAL_ARMS", "ft_pretrained,ft_scratch").split(",")
assert set(ARMS) <= {"ft_pretrained", "ft_scratch"}, ARMS
SCR_LR = float(os.environ.get("EVAL_SCR_LR", "5e-4"))
PERM_TIME = int(os.environ.get("EVAL_PERM_TIME", "0"))
print(f"[cfg2] arms={ARMS} scr_lr={SCR_LR} perm_time={PERM_TIME} out={OUT}", flush=True)
FRACS = [float(x) for x in os.environ.get("EVAL_FRACS", "0.01,0.05,0.10,0.25,1.0").split(",")]
FT_EPOCHS = int(os.environ.get("EVAL_FT_EPOCHS", "6"))
FT_MAX_ROWS = int(os.environ.get("EVAL_FT_MAX_ROWS", "150000"))
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[TAG.split("_")[0]]
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
TRAIN_END, TEST_START = "2020-12", "2022-01"
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL); USE_PRIO = True
TIME_COL = FIELDS.index("month_offset")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF, REGIONS
assert RG in REGIONS, f"bad EVAL_REGION {RG}"
assert os.path.exists(CKPT), f"missing checkpoint {CKPT}"
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
print(f"[cfg] RG={RG} TAG={TAG} fracs={FRACS} ep={FT_EPOCHS} cap={FT_MAX_ROWS} dev={DEVICE}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}

class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, use_prio, static_dim, dropout=0.1):
        super().__init__()
        self.fields, self.use_prio = fields, use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio: self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio: cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

class Clf(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc; self.head = nn.Linear(enc.ln.normalized_shape[0], 1)
    def forward(self, seq, mask, static): return self.head(self.enc.cls_vec(seq, mask, static)).squeeze(-1)

class SeqDS(Dataset):
    def __init__(self, seq_strings, gids, labels=None):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seq_strings]
        self.gids = gids; self.labels = None if labels is None else np.asarray(labels, np.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_SEQ_LEN:]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS): t[j, f] = ev[SEL[f]]
            m[j] = True
        if PERM_TIME and len(seq) > 1:
            n = len(seq); perm = torch.from_numpy(np.random.RandomState((PERM_TIME * 1000003 + i) % (2**31)).permutation(n))
            t[:n, TIME_COL] = t[perm, TIME_COL]
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        y = 0.0 if self.labels is None else self.labels[i]
        return t, m, st, torch.tensor(y)

def finetune_eval(seqs_tr, gids_tr, y_tr, seqs_te, gids_te, y_te, ckpt, lr):
    enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        miss, unexp = enc.load_state_dict(sd, strict=False)
        assert not [k for k in miss if k.startswith("transformer.")], f"transformer weights missing: {miss[:5]}"
    clf = Clf(enc).to(DEVICE)
    pos = float(max(y_tr.sum(), 1.0)); neg = float(len(y_tr)) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(neg / pos, 1.0)], device=DEVICE))
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    ld = DataLoader(SeqDS(seqs_tr, gids_tr, y_tr), batch_size=FT_BATCH, shuffle=True, num_workers=0)
    clf.train()
    for ep in range(FT_EPOCHS):
        for t, m, st, y in ld:
            loss = crit(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt.step()
    clf.eval(); preds = []
    with torch.no_grad():
        for t, m, st, _ in DataLoader(SeqDS(seqs_te, gids_te), batch_size=BATCH, shuffle=False, num_workers=0):
            preds.append(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    p = np.concatenate(preds)
    roc, pr = roc_auc_score(y_te, p), average_precision_score(y_te, p)
    del clf, enc
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return roc, pr

panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "event_sequence", "violence_next_month", "violence_lag1"])
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
d = panel[panel["country"].isin(REGION_COUNTRIES) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
assert len(d) >= 200, f"region {RG} too small: {len(d)}"
ym = d["year_month"].to_numpy(); trm = ym <= TRAIN_END; tem = ym >= TEST_START
y = d["violence_next_month"].to_numpy().astype(int); ytr, yte = y[trm], y[tem]
seqs = d["event_sequence"].values; gids = d["grid_id"].values
_chk = [json.loads(s) for s in seqs[np.random.RandomState(0).choice(len(seqs), size=min(5000, len(seqs)), replace=False)] if isinstance(s, str)]
_var = float(np.mean([len({ev[SEL[TIME_COL]] for ev in q}) > 1 for q in _chk if len(q) > 1] or [0.0]))
print(f"[permcheck] share of multi-event seqs with >1 distinct month_offset: {_var:.3f}", flush=True)
assert not PERM_TIME or _var > 0.05, "month_offset barely varies within sequences; time permutation would be a no-op"
assert trm.sum() >= 100 and tem.sum() >= 100 and ytr.sum() >= 10 and yte.sum() >= 10, "too few rows/positives"
print(f"[region {RG}] rows={len(d):,} train={trm.sum():,} test={tem.sum():,} test_pos={yte.mean():.4f}", flush=True)

rows = []
base = dict(tag=TAG, region=RG, seed=SEED, n_train=int(trm.sum()), n_test=int(tem.sum()), test_pos=float(yte.mean()))
persist = np.nan_to_num(d["violence_lag1"].to_numpy())[tem]
rows.append(dict(base, model="persistence", frac=1.0, roc=roc_auc_score(yte, persist), pr=average_precision_score(yte, persist)))
tr_idx = np.where(trm)[0]; pos_idx = tr_idx[y[tr_idx] == 1]; neg_idx = tr_idx[y[tr_idx] == 0]
seqs_te, gids_te = seqs[tem], gids[tem]
for frac in FRACS:
    npos = int(round(len(pos_idx) * frac)); nneg = int(round(len(neg_idx) * frac))
    if npos < 5:
        rows.append(dict(base, model="too_few_pos", frac=frac, roc=np.nan, pr=np.nan)); continue
    idx = np.concatenate([rng.choice(pos_idx, size=npos, replace=False), rng.choice(neg_idx, size=nneg, replace=False)])
    if len(idx) > FT_MAX_ROWS: idx = rng.choice(idx, size=FT_MAX_ROWS, replace=False)
    rng.shuffle(idx)
    for mdl, ck, lr in [a for a in [("ft_pretrained", CKPT, 1e-4), ("ft_scratch", None, SCR_LR)] if a[0] in ARMS]:
        t0 = time.time()
        roc, pr = finetune_eval(seqs[idx], gids[idx], y[idx], seqs_te, gids_te, yte, ck, lr)
        rows.append(dict(base, model=mdl, frac=frac, n_ft=int(len(idx)), roc=roc, pr=pr, lr=lr, perm_time=PERM_TIME))
        print(f"  {mdl} frac={frac} n={len(idx)} ROC {roc:.4f} PR {pr:.4f} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)

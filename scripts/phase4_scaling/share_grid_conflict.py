"""share_grid_conflict.py -- final limited-data design for the conflict LORO tables (Tables 3 and 7).

One job = one held-out region RG x one encoder seed SEED x one ARM:
  ARM=nested    frozen embedding + features (emb_base) vs features alone (base), XGBoost, all shares x all draws
  ARM=finetune  pretrained encoder fine-tuned vs identical network from scratch, ONE share (GRID_SHARES must
                hold one value), all draws; learning-rate x weight-decay sweep; early stopping on 2021
Design (agreed 2026-10-10):
  shares   0.25, 0.5, 0.75, 1, 2.5, 5, 10 percent of RG's training period (1990-01..2020-12), no row cap
  draws    5 stratified random draws per share; the draw depends only on (region, share, draw), never on the
           encoder seed or arm, so every arm and seed sees identical rows and adding a share later changes nothing
  val      2021 rows of RG drawn at the SAME share (stratified), floor of 10 violent cell-months
  sweep    lr {3e-5,1e-4,3e-4,1e-3} x weight decay {0,0.01}, identical for both fine-tune arms; config chosen by
           val PR-AUC; GRID_SWEEP=draw0 sweeps on draw 0 and reuses the chosen config for draws 1-4,
           GRID_SWEEP=all sweeps every draw
  stopping max 50 epochs, patience 5 on val PR-AUC, best-val weights restored before scoring the test period
  XGBoost  fixed settings (depth 6, eta 0.1, subsample 0.8, colsample 0.8, 200 rounds), seed fixed at 123 so the
           base arm is identical across encoder seeds
  test     RG, Jan 2022 .. Apr 2025 forecast months (Mode B), PR-AUC, ROC-AUC, Brier
Outputs (outputs/share_grid/, never overwritten):
  metrics_{ARM}_{RG}_s{SEED}[_f{share}].csv   one row per model x share x draw
  sweep_finetune_{RG}_s{SEED}_f{share}.csv    every (arm, draw, lr, wd) tried: best val PR, best epoch
  preds_{ARM}_{RG}_s{SEED}[_f{share}].npz      test-period forecasts (float32) for every reported row, plus the
                                               test-row identity (__grid_id, __year_month, __y) for cross-job checks
  avail_{ARM}_{RG}_s{SEED}[_f{share}].csv      per share x draw: "ok" or the floor that made it not estimable
                                               (train_positives_below_floor / val_pool_below_floor), with counts
  Files are written as .part and renamed only when the whole job finishes; a .part file means an interrupted run.
Targets (Oct 15): GRID_TARGET=incidence (main) or civilian (any violence-against-civilians event next month, built as in
  transfer_eval_region_fullpreds.py; outputs carry "_civ"). GRID_PREFLIGHT=1 writes preflight_{target}_{region}.csv
  (which share x draw cells pass the positive floors) and exits without training.
Divergence (Oct 15, Codex 9): a fine-tuning configuration whose batch loss, gradient norm or validation forecasts become
  non-finite is recorded as diverged in the sweep file and skipped. If the configuration chosen on draw 0 diverges on a
  later draw, that arm sweeps the full grid on that draw (sweep rows carry fallback=True); the other arm keeps its
  draw-0 choice. This is an adaptive rule and is reported as such. If every configuration diverges the job stops.
Vocabulary (main specification, decided 2026-10-10): tokens the encoder never saw in pretraining are mapped to <UNK>
  for every arm and row (vocab_remap.py). VOCAB_REMAP=0 runs the original vocabulary (appendix sensitivity check);
  its outputs carry "_vorig" in the job name.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = os.environ.get("GRID_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
RG = E("EVAL_REGION"); SEED = int(E("EVAL_SEED", "123")); ARM = E("GRID_ARM", "nested")
SMOKE = E("SMOKE", "0") == "1"
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
assert RG in REGION_ORDER, f"bad EVAL_REGION {RG}"
assert ARM in ("nested", "finetune"), ARM
RIDX = REGION_ORDER.index(RG)
TAG = f"L_contrastive_e_loro_{RG}_s{SEED}"
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
SHARES = [float(x) for x in E("GRID_SHARES", "0.0025,0.005,0.0075,0.01,0.025,0.05,0.10").split(",")]
DRAWS = [int(x) for x in E("GRID_DRAWS", "0,1,2,3,4").split(",")]
LRS = [float(x) for x in E("GRID_LRS", "3e-5,1e-4,3e-4,1e-3").split(",")]
WDS = [float(x) for x in E("GRID_WDS", "0,0.01").split(",")]
SWEEP = E("GRID_SWEEP", "draw0")
TARGET = E("GRID_TARGET", "incidence")       # incidence (main) | civilian (any violence-against-civilians event next month)
assert TARGET in ("incidence", "civilian"), TARGET
MAX_EP = int(E("GRID_MAX_EP", "50")); PATIENCE = int(E("GRID_PATIENCE", "5")); MIN_DELTA = 1e-4
VAL_MIN_POS = 10; MIN_TRAIN_POS = 5; DRAW_BASE = 123; XGB_SEED = 123
assert SWEEP in ("draw0", "all"), SWEEP
PREFLIGHT = E("GRID_PREFLIGHT", "0") == "1"   # count-only: write which share x draw cells are estimable, then exit
if ARM == "finetune" and not PREFLIGHT: assert len(SHARES) == 1, "finetune jobs take exactly one share"
if SMOKE:
    SHARES = SHARES[:1] if ARM == "finetune" else [0.01, 0.05]
    DRAWS = [0, 1]; LRS = [1e-4, 1e-3]; WDS = [0.0]; MAX_EP = 3; PATIENCE = 2
SUF = "_smoke" if SMOKE else ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_remap import remap_enabled, remap_tables
VOCAB_REMAP = remap_enabled(); VSUF = "" if VOCAB_REMAP else "_vorig"
OD = f"{R}/outputs/share_grid"; os.makedirs(OD, exist_ok=True)
TSUF = "" if TARGET == "incidence" else "_civ"
JOB = f"{ARM}_{RG}_s{SEED}" + (f"_f{SHARES[0]:g}" if ARM == "finetune" else "") + VSUF + TSUF + SUF
OUT_M = f"{OD}/metrics_{JOB}.csv"; OUT_P = f"{OD}/preds_{JOB}.npz"; OUT_S = f"{OD}/sweep_{JOB}.csv"
OUT_A = f"{OD}/avail_{JOB}.csv"                # per share x draw: estimable or skipped, with reason and positive counts
OUT_PF = f"{OD}/preflight_{TARGET}_{RG}.csv"
if PREFLIGHT: assert not os.path.exists(OUT_PF), f"refusing to overwrite {OUT_PF}"
else: assert not os.path.exists(OUT_M), f"refusing to overwrite completed job {OUT_M}"   # metrics file = completion marker
for f in (OUT_P, OUT_S):                       # orphans from an interrupted publish (metrics never renamed) are replaced
    if os.path.exists(f): print(f"[recover] {f} exists without {OUT_M}; it will be overwritten", flush=True)
TMP_M, TMP_P, TMP_S = OUT_M + ".part", OUT_P + ".part.npz", OUT_S + ".part"   # interrupted runs leave only these
assert os.path.exists(CKPT), f"missing encoder {CKPT}"

CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[TAG.split("_")[0]]
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
TRAIN_START, TRAIN_END, VAL_END, TEST_START = "1990-01", "2020-12", "2021-12", "2022-01"
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL); USE_PRIO = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF, REGIONS
assert set(REGION_ORDER) == set(REGIONS), "REGION_ORDER out of sync with regions.py"
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"[cfg] VOCAB_REMAP={int(VOCAB_REMAP)} JOB={JOB} shares={SHARES} draws={DRAWS} lrs={LRS} wds={WDS} sweep={SWEEP} "
      f"max_ep={MAX_EP} patience={PATIENCE} dev={DEVICE}", flush=True)

# ---- vocab, PRIO lookup, Encoder: copied from transfer_eval_region_ft2.py / nested_region_frac.py ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
LUT = remap_tables(TAG, FIELDS, vsm, VOCAB_REMAP)   # identity lists when VOCAB_REMAP=0
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

def parse_seq(s):
    """JSON string or decoded list/array -> list of events, truncated to the last MAX_SEQ_LEN at parse time."""
    if isinstance(s, str): seq = json.loads(s)
    elif isinstance(s, (list, tuple, np.ndarray)): seq = [list(ev) for ev in s]
    elif s is None or (isinstance(s, float) and np.isnan(s)): seq = []
    else: raise TypeError(f"unsupported event_sequence type {type(s)}")
    return seq[-MAX_SEQ_LEN:]

class SeqDS(Dataset):
    def __init__(self, seq_strings, gids, labels=None):
        self.seqs = [parse_seq(s) for s in seq_strings]
        self.gids = gids; self.labels = None if labels is None else np.asarray(labels, np.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS): t[j, f] = LUT[f][ev[SEL[f]]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        y = 0.0 if self.labels is None else self.labels[i]
        return t, m, st, torch.tensor(y)

def new_encoder(pretrained):
    enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
    if pretrained:
        sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        miss, unexp = enc.load_state_dict(sd, strict=False)
        assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
    return enc

# ---- data: same panel, merge and split as nested_region_frac.py ----
base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "event_sequence", "violence_next_month",
                                 "etype_violence_against_civilians"] + base_cols)
# civilian target, same construction as transfer_eval_region_fullpreds.py: count of violence-against-civilians events in
# month t (civ_now) and in month t+1 (civ_next = next row of the same cell). Built on the full panel before the region
# row filter so that a missing country label never shifts a cell's months. The panel has no month gaps within cells.
# The panel's own row order is NOT changed: the share draws index rows by position, so reordering would change which
# rows every incidence draw selects. The label is built on a sorted copy and written back by index.
_srt = panel[["grid_id", "year_month", "etype_violence_against_civilians"]].sort_values(["grid_id", "year_month"], kind="mergesort")
_g = _srt["grid_id"].to_numpy(); _per = pd.PeriodIndex(_srt["year_month"].astype(str), freq="M")
_mi = (_per.year * 12 + _per.month).to_numpy()
assert ((_g[1:] != _g[:-1]) | (_mi[1:] - _mi[:-1] == 1)).all(), "month gap within a cell; civ_next would misalign"
_srt["civ_now"] = _srt["etype_violence_against_civilians"].fillna(0.0)
_nx = _srt.groupby("grid_id")["civ_now"].shift(-1)
_srt["civ_has_next"] = _nx.notna()              # Codex 9, F01: an absent outcome month is never turned into a negative
_srt["civ_next"] = _nx.fillna(0.0)
panel["civ_now"] = _srt["civ_now"]; panel["civ_next"] = _srt["civ_next"]       # index-aligned, original order kept
panel["civ_has_next"] = _srt["civ_has_next"]
assert panel["civ_next"].notna().all()
del _srt, _g, _per, _mi, _nx
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
d = panel[panel["country"].isin(REGION_COUNTRIES) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
del panel, cm
n0 = len(d)
assert ps["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "prio merge changed row count"
FEATS = base_cols + static_cols
d[FEATS] = d[FEATS].fillna(0.0)
X = d[FEATS].to_numpy(np.float32)
ym = d["year_month"].astype(str).to_numpy()
trm = (ym >= TRAIN_START) & (ym <= TRAIN_END); vam = (ym > TRAIN_END) & (ym <= VAL_END); tem = ym >= TEST_START
y = (d["violence_next_month"].to_numpy().astype(int) if TARGET == "incidence"
     else (d["civ_next"].to_numpy() > 0).astype(int))      # rows still require violence_next_month (same rows as the main grid)
if TARGET == "civilian":
    assert d["civ_has_next"].all(), f"{int((~d['civ_has_next']).sum())} retained rows have no observed next month"
print(f"[target] {TARGET}: positive rate {y.mean():.4f}", flush=True)
seqs = d["event_sequence"].values; gids = d["grid_id"].values
te_idx = np.where(tem)[0]
if SMOKE: te_idx = np.sort(np.random.default_rng(0).choice(te_idx, size=min(20000, len(te_idx)), replace=False))
yte = y[te_idx]
assert trm.sum() > 1000 and vam.sum() > 100 and len(te_idx) > 1000 and 0 < yte.sum() < len(yte)
assert not (trm & vam).any() and not (vam & tem).any()
print(f"[region {RG}] rows={len(d):,} train={trm.sum():,} val={vam.sum():,} test={len(te_idx):,} "
      f"test_pos={yte.mean():.4f}", flush=True)
TEST_ID = dict(__grid_id=np.asarray(gids[te_idx]).astype(np.int64), __year_month=ym[te_idx].astype("U7"), __y=yte.astype(np.int8))

def draw_rows(mask, share, k, purpose):
    """Stratified draw from rows in `mask`. Depends only on (region, share, draw k, purpose)."""
    rng = np.random.default_rng([DRAW_BASE, RIDX, int(round(share * 1e5)), k, purpose])
    pool = np.where(mask)[0]; pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
    npos = int(round(len(pos) * share)); nneg = int(round(len(neg) * share))
    if purpose == 1:                                 # val floor of VAL_MIN_POS positives, base rate kept
        if len(pos) < VAL_MIN_POS: return None
        if npos < VAL_MIN_POS: npos = VAL_MIN_POS; nneg = min(len(neg), int(round(npos * len(neg) / len(pos))))
    elif npos < MIN_TRAIN_POS: return None
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]))

avail_rows = []
def draw_status(s, k, need_val):
    """Training (and, for fine-tuning, validation) draw for share s, draw k, plus an availability record (Codex 9, F03)."""
    tr = draw_rows(trm, s, k, 0); va = draw_rows(vam, s, k, 1) if need_val else None
    tp = int((y[trm] == 1).sum()); vp = int((y[vam] == 1).sum())
    reason = ("ok" if tr is not None and (va is not None or not need_val)
              else "train_positives_below_floor" if tr is None else "val_pool_below_floor")
    avail_rows.append(dict(region=RG, enc_seed=SEED, arm=ARM, target=TARGET, share=s, draw=k, status=reason,
                           train_pool_pos=tp, train_sel_pos=int(round(tp * s)), min_train_pos=MIN_TRAIN_POS,
                           val_pool_pos=vp if need_val else np.nan, min_val_pos=VAL_MIN_POS if need_val else np.nan))
    return tr, va, reason

if PREFLIGHT:
    for s_ in SHARES:
        for k_ in DRAWS: draw_status(s_, k_, ARM == "finetune")
    A = pd.DataFrame(avail_rows); A.to_csv(OUT_PF + ".part", index=False); os.replace(OUT_PF + ".part", OUT_PF)
    print(A.groupby(["share", "status"]).size().to_string()); print(f"[preflight done] {OUT_PF}", flush=True)
    sys.exit(0)

def scores(yt, p):
    return dict(pr=float(average_precision_score(yt, p)), roc=float(roc_auc_score(yt, p)),
                brier=float(np.mean((np.clip(p, 0, 1) - yt) ** 2)))

rows, preds, sweep_rows = [], {}, []
def emit(model, share, k, idx, p, **extra):
    assert len(p) == len(yte) and np.isfinite(p).all(), (model, share, k)
    r = dict(region=RG, enc_seed=SEED, arm=ARM, model=model, share=share, draw=k,
             n_train=0 if idx is None else int(len(idx)), pos_train=np.nan if idx is None else int(y[idx].sum()),
             n_test=int(len(yte)), test_pos=float(yte.mean()), **scores(yte, p), **extra)
    rows.append(r); preds[f"{model}|{share:g}|{k}"] = p.astype(np.float32)
    pd.DataFrame(rows).to_csv(TMP_M, index=False)
    print(f"  [{model}] share={share:g} draw={k} n={r['n_train']} PR={r['pr']:.4f} ROC={r['roc']:.4f}", flush=True)

_pers = (np.nan_to_num(d["violence_lag1"].to_numpy()) if TARGET == "incidence"
         else d["civ_now"].to_numpy())                     # civilian persistence = month-t count, as in the main evaluator
emit("persistence", 0.0, -1, None, _pers[te_idx].astype(float))

if ARM == "nested":
    draws = {(s, k): draw_status(s, k, False)[0] for s in SHARES for k in DRAWS}
    need = np.unique(np.concatenate([v for v in draws.values() if v is not None] + [te_idx]))
    enc = new_encoder(True); enc.eval()
    Z = []; t0 = time.time()
    with torch.no_grad():
        for t, m, st, _ in DataLoader(SeqDS(seqs[need], gids[need]), batch_size=BATCH, shuffle=False, num_workers=0):
            Z.append(enc.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(Z); assert Z.shape == (len(need), EMBED_DIM) and np.isfinite(Z).all()
    print(f"[emb] {Z.shape} in {time.time()-t0:.0f}s", flush=True)
    posn = -np.ones(len(d), np.int64); posn[need] = np.arange(len(need))
    def Zof(ix): assert (posn[ix] >= 0).all(); return Z[posn[ix]]
    def xgb_p(Xtr, ytr, Xte):
        b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                       "colsample_bytree": 0.8, "seed": XGB_SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
        return b.predict(xgb.DMatrix(Xte))
    Zte, Bte = Zof(te_idx), X[te_idx]
    for (s, k), idx in draws.items():
        if idx is None: print(f"  [skip] share={s:g} draw={k}: too few positives", flush=True); continue
        emit("base", s, k, idx, xgb_p(X[idx], y[idx], Bte))
        emit("emb_base", s, k, idx, xgb_p(np.concatenate([Zof(idx), X[idx]], 1), y[idx], np.concatenate([Zte, Bte], 1)))

else:
    s = SHARES[0]
    ds_te = SeqDS(seqs[te_idx], gids[te_idx])
    def predict(clf, ds):
        clf.eval(); out = []
        with torch.no_grad():
            for t, m, st, _ in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0):
                out.append(torch.sigmoid(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE))).float().cpu().numpy())
        return np.concatenate(out)
    def train_one(pretrained, lr, wd, tr, va, k):
        torch.manual_seed(SEED * 1000 + k)
        clf = Clf(new_encoder(pretrained)).to(DEVICE)
        pos = float(max(y[tr].sum(), 1.0)); neg = float(len(tr)) - pos
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(neg / pos, 1.0)], device=DEVICE))
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
        ld = DataLoader(SeqDS(seqs[tr], gids[tr], y[tr]), batch_size=FT_BATCH, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED * 1000 + k))
        ds_va = SeqDS(seqs[va], gids[va]); yva = y[va]
        best, best_ep, best_sd, bad = -1.0, 0, None, 0
        for ep in range(1, MAX_EP + 1):
            clf.train()
            for t, m, st, yy in ld:
                loss = crit(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)), yy.to(DEVICE))
                if not torch.isfinite(loss):              # diverged (Codex 9, F02): checked every batch, before the step
                    return None, float("-inf"), 0, ep
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                if not torch.isfinite(gn):                # non-finite gradient = diverged; never applied
                    return None, float("-inf"), 0, ep
                opt.step()
            pva = predict(clf, ds_va)
            if not np.isfinite(pva).all():                # non-finite validation forecasts = diverged
                return None, float("-inf"), 0, ep
            v = float(average_precision_score(yva, pva))
            improved = v > best + MIN_DELTA                    # MIN_DELTA governs patience only
            if v > best: best, best_ep, best_sd = v, ep, {kk: t_.detach().cpu().clone() for kk, t_ in clf.state_dict().items()}
            bad = 0 if improved else bad + 1
            if bad >= PATIENCE: break
        clf.load_state_dict(best_sd)
        return clf, best, best_ep, ep
    chosen = {}
    for k in DRAWS:
        tr, va, why = draw_status(s, k, True)
        if why != "ok": print(f"  [skip] share={s:g} draw={k}: {why}", flush=True); continue
        for arm_name, pretrained in (("ft_pretrained", True), ("ft_scratch", False)):
            full_grid = [(lr, wd) for lr in LRS for wd in WDS]
            grid = full_grid if (SWEEP == "all" or arm_name not in chosen) else [chosen[arm_name]]
            best = None
            for lr, wd in grid:
                t0 = time.time()
                clf, vpr, bep, lep = train_one(pretrained, lr, wd, tr, va, k)
                sweep_rows.append(dict(region=RG, enc_seed=SEED, model=arm_name, share=s, draw=k, lr=lr, wd=wd,
                                       val_pr=vpr if clf is not None else np.nan, diverged=clf is None,
                                       best_epoch=bep, last_epoch=lep, n_train=len(tr), n_val=len(va),
                                       val_pos=int(y[va].sum()), secs=round(time.time() - t0), fallback=False))
                pd.DataFrame(sweep_rows).to_csv(TMP_S, index=False)
                print(f"    {arm_name} draw={k} lr={lr:g} wd={wd:g} valPR={vpr:.4f} best_ep={bep} stop_ep={lep} "
                      f"{time.time()-t0:.0f}s", flush=True)
                if clf is None:
                    print(f"    {arm_name} draw={k} lr={lr:g} wd={wd:g} DIVERGED at epoch {lep}; skipped", flush=True)
                    continue
                if best is None or vpr > best[0]: best = (vpr, lr, wd, bep, clf)
                del clf                                         # winner (if any) is held only by `best`
            if best is None and grid != full_grid:              # the draw-0 choice diverged on this draw: sweep this draw
                print(f"    {arm_name} draw={k}: chosen config diverged; sweeping the full grid for this draw", flush=True)
                for lr, wd in full_grid:
                    if (lr, wd) in grid: continue
                    t0 = time.time()
                    clf, vpr, bep, lep = train_one(pretrained, lr, wd, tr, va, k)
                    sweep_rows.append(dict(region=RG, enc_seed=SEED, model=arm_name, share=s, draw=k, lr=lr, wd=wd,
                                           val_pr=vpr if clf is not None else np.nan, diverged=clf is None,
                                           best_epoch=bep, last_epoch=lep, n_train=len(tr), n_val=len(va),
                                           val_pos=int(y[va].sum()), secs=round(time.time() - t0), fallback=True))
                    pd.DataFrame(sweep_rows).to_csv(TMP_S, index=False)
                    if clf is None: continue
                    if best is None or vpr > best[0]: best = (vpr, lr, wd, bep, clf)
                    del clf
            assert best is not None, f"{arm_name} draw={k}: every learning-rate setting diverged"
            if arm_name not in chosen: chosen[arm_name] = (best[1], best[2])
            emit(arm_name, s, k, tr, predict(best[4], ds_te), lr=best[1], wd=best[2], best_epoch=best[3],
                 val_pr=best[0], n_val=len(va))
            del best
            if torch.cuda.is_available(): torch.cuda.empty_cache()

# ---- publish: forecasts first, then metrics (tables read metrics_*.csv only, so a missing csv = not finished) ----
np.savez_compressed(TMP_P, **TEST_ID, **preds)
chk = np.load(TMP_P)                                     # default allow_pickle=False must work
assert all(k in chk.files for k in preds) and chk["__year_month"].dtype.kind == "U"
A = pd.DataFrame(avail_rows)
n_ok = int((A.status == "ok").sum())
exp_models = {"base", "emb_base"} if ARM == "nested" else {"ft_pretrained", "ft_scratch"}
got = {k_ for k_ in preds if not k_.startswith("persistence")}
assert len(got) == n_ok * len(exp_models), f"{len(got)} forecasts for {n_ok} estimable draws"   # every estimable draw has both arms
A.to_csv(OUT_A + ".part", index=False); os.replace(OUT_A + ".part", OUT_A)      # availability published before the marker
if os.path.exists(TMP_S): os.replace(TMP_S, OUT_S)
os.replace(TMP_P, OUT_P); os.replace(TMP_M, OUT_M)
print(f"[done] {OUT_M} ({len(rows)} rows), {OUT_P}", flush=True)

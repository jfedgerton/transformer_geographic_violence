"""ens_finetune.py -- forecasting ensemble, stage 1: fine-tune one pretrained encoder for one forecasting target.
(Forecasting work for the GAPR memo / appendix. Not part of the LORO transfer design.)

One job = one ALL-REGION pretrained encoder (ENS_ENC_SEED, i.e. repr_model_L_contrastive_e_s{seed}.pt; seeds 123, 456,
789, 101112, 131415) x one target ENS_TARGET. This is forecasting with a pretrained encoding, not the LORO transfer
design: the encoders were pretrained on every region (1990-2020) and are fine-tuned on every region. Original
vocabulary (no LORO token remap); tokens first seen after 2020 keep their untrained rows, as in pretraining.

Targets (forecast row = cell-month t, outcome in month t+1):
  incidence  any violent event in t+1                                   (violence_next_month)
  civilian   any violence-against-civilians event in t+1                (as in share_grid_conflict.py)
  onset      any violent event in t+1, rows restricted to cells with no violent event in month t (has_violence == 0)
  fatal25    25 or more fatalities in t+1                               (count_fatalities of the next row of the cell)
Time split (forecast months):
  train  1990-01 .. 2020-12   balanced sample: all positives up to POS_CAP, NEG_PER_POS negatives per positive
  stop   2021-01 .. 2021-06   early stopping and learning-rate choice (stratified sample, >= STOP_MIN_POS positives)
  stack  2021-07 .. 2021-12   all rows; forecasts saved for the stage-3 combiner (never used for fitting here)
  test   2022-01 ..           all rows with an observed outcome; forecasts saved, never used for any choice
The train / stop samples depend only on the target (seed 123), never on the encoder, so all five encoders see the
same rows. Training uses the balanced sample without a positive weight, so forecasts rank well but are not
calibrated probabilities; stage 3 calibrates on the stack window.
ENS_PREFLIGHT=1 builds the samples and compact sequences, prints counts and peak memory, and exits (CPU, no outputs).
Guards: refuses to run unless ENS_PRETRAIN_CUTOFF_OK=1 (set only after checking that the encoder's pretraining data end
no later than 2020-12; otherwise the 2021 and test windows are contaminated). Never overwrites; writes .part files and
renames at the end.
Outputs:
  data_model/ens/ft_{target}_{TAG}.pt              fine-tuned encoder + head (best stop-window PR-AUC)
  outputs/ens/preds_ft_{target}_{TAG}.npz          grid_id, year_month, region, window (stack|test), y, p
  outputs/ens/log_ft_{target}_{TAG}.csv            one row per (lr, epoch): train loss, stop PR-AUC
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
ENC_SEED = int(E("ENS_ENC_SEED", "123")); TARGET = E("ENS_TARGET", "incidence")
SMOKE = E("SMOKE", "0") == "1"
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
assert ENC_SEED in (123, 456, 789, 101112, 131415), f"bad ENS_ENC_SEED {ENC_SEED}"
assert TARGET in TARGETS, f"bad ENS_TARGET {TARGET}"
assert E("ENS_PRETRAIN_CUTOFF_OK", "0") == "1", "set ENS_PRETRAIN_CUTOFF_OK=1 only after verifying pretraining data end <= 2020-12"
TAG = f"L_contrastive_e_s{ENC_SEED}"
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
assert os.path.exists(CKPT), f"missing encoder {CKPT}"

TRAIN_START, TRAIN_END = "1990-01", "2020-12"
STOP_START, STOP_END = "2021-01", "2021-06"
STACK_START, STACK_END = "2021-07", "2021-12"
TEST_START = "2022-01"
POS_CAP = int(E("ENS_POS_CAP", "100000")); NEG_PER_POS = int(E("ENS_NEG_PER_POS", "4"))
STOP_SHARE = 0.25; STOP_MIN_POS = 200
LRS = [float(x) for x in E("ENS_LRS", "1e-4,3e-4").split(",")]; WD = 0.01
MAX_EP = int(E("ENS_MAX_EP", "15")); PATIENCE = int(E("ENS_PATIENCE", "3")); MIN_DELTA = 1e-4
SAMPLE_SEED = 123
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
WORKERS = int(E("ENS_WORKERS", "4"))
if SMOKE:
    POS_CAP = int(E("ENS_POS_CAP", "200")); LRS = LRS if "ENS_LRS" in os.environ else [1e-4]
    MAX_EP = int(E("ENS_MAX_EP", "2")); PATIENCE, STOP_MIN_POS = 1, 5
SUF = "_smoke" if SMOKE else ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_remap import remap_tables
from regions import REGION_OF, REGIONS
assert set(REGION_ORDER) == set(REGIONS), "REGION_ORDER out of sync with regions.py"

JOB = f"{TARGET}_{TAG}{SUF}"
MD, OD = f"{R}/data_model/ens", f"{R}/outputs/ens"
os.makedirs(MD, exist_ok=True); os.makedirs(OD, exist_ok=True)
OUT_W, OUT_P, OUT_L = f"{MD}/ft_{JOB}.pt", f"{OD}/preds_ft_{JOB}.npz", f"{OD}/log_ft_{JOB}.csv"
if E("ENS_PREFLIGHT", "0") != "1":
    assert not os.path.exists(OUT_P), f"refusing to overwrite completed job {OUT_P}"   # preds file = completion marker
TMP_W, TMP_P, TMP_L = OUT_W + ".part", OUT_P + ".part.npz", OUT_L + ".part"

CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[TAG.split("_")[0]]
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(ENC_SEED); np.random.seed(ENC_SEED)
print(f"[cfg] JOB={JOB} lrs={LRS} wd={WD} max_ep={MAX_EP} patience={PATIENCE} pos_cap={POS_CAP} "
      f"neg_per_pos={NEG_PER_POS} dev={DEVICE}", flush=True)

# ---- vocab, PRIO lookup, Encoder: copied from share_grid_conflict.py ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
LUT = remap_tables(TAG, FIELDS, vsm, False)   # identity tables: original vocabulary
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
assert ps["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}

class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, static_dim, dropout=0.1):
        super().__init__()
        self.fields = fields
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1) + self.prio_proj(static).unsqueeze(1)
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
    if isinstance(s, str): seq = json.loads(s)
    elif isinstance(s, (list, tuple, np.ndarray)): seq = [list(ev) for ev in s]
    elif s is None or (isinstance(s, float) and np.isnan(s)): seq = []
    else: raise TypeError(f"unsupported event_sequence type {type(s)}")
    return seq[-MAX_SEQ_LEN:]

LUT_NP = [np.asarray(l, np.int64) for l in LUT]
assert max(int(l.max()) for l in LUT_NP) < 32767, "token ids do not fit int16"
def compact_seq(s):
    """raw sequence -> (L, NUM_FIELDS) int16 array of encoder token ids (LUT applied), L <= MAX_SEQ_LEN (Codex 12, memory)."""
    seq = parse_seq(s)
    if not seq: return np.zeros((0, NUM_FIELDS), np.int16)
    a = np.asarray(seq, np.int64)[:, SEL]
    return np.stack([LUT_NP[f][a[:, f]] for f in range(NUM_FIELDS)], 1).astype(np.int16)

class SeqDS(Dataset):
    def __init__(self, compact, gids, labels=None):
        self.seqs = compact                      # object array of int16 arrays, shared with the caller (no copy)
        self.gids = gids; self.labels = None if labels is None else np.asarray(labels, np.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        a = self.seqs[i]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        if len(a): t[:len(a)] = torch.from_numpy(a.astype(np.int64)); m[:len(a)] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        y = 0.0 if self.labels is None else self.labels[i]
        return t, m, st, torch.tensor(y)

# ---- panel and targets (labels built on a grid-sorted copy, written back by index) ----
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "violence_next_month", "has_violence",
                                 "count_fatalities", "etype_violence_against_civilians"])
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months"
srt = panel[["grid_id", "year_month", "has_violence", "count_fatalities", "etype_violence_against_civilians"]].sort_values(
    ["grid_id", "year_month"], kind="mergesort")
g_ = srt["grid_id"].to_numpy(); per = pd.PeriodIndex(srt["year_month"].astype(str), freq="M"); mi = (per.year * 12 + per.month).to_numpy()
assert ((g_[1:] != g_[:-1]) | (mi[1:] - mi[:-1] == 1)).all(), "month gap within a cell; next-month labels would misalign"
nxt = srt.groupby("grid_id")[["has_violence", "count_fatalities", "etype_violence_against_civilians"]].shift(-1)
srt["has_next"] = nxt["has_violence"].notna()
srt["fat_next"] = nxt["count_fatalities"]; srt["civ_next"] = nxt["etype_violence_against_civilians"].fillna(0.0)
for c in ("has_next", "fat_next", "civ_next"): panel[c] = srt[c]
del srt, nxt, g_, per, mi
# has_violence(t+1) must equal violence_next_month(t): checked on job 56509258 (cell_month_decompose.py); re-checked here
ok = panel["has_next"] & panel["violence_next_month"].notna()
cmp = pd.Series(np.nan, index=panel.index)
_s = panel.sort_values(["grid_id", "year_month"], kind="mergesort")
cmp.loc[_s.index] = _s.groupby("grid_id")["has_violence"].shift(-1).to_numpy()
assert (cmp[ok].astype(int) == panel.loc[ok, "violence_next_month"].astype(int)).all(), "has_violence(t+1) != violence_next_month(t)"
del cmp, _s
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left"); del cm
panel["region"] = panel["country"].map(REGION_OF).fillna("other")
d = panel[panel["region"].isin(REGION_ORDER) & panel["violence_next_month"].notna() & panel["has_next"]].reset_index(drop=True)
del panel
if TARGET == "onset": d = d[d["has_violence"].fillna(0).astype(int) == 0].reset_index(drop=True)
if TARGET == "incidence": y = d["violence_next_month"].to_numpy().astype(int)
elif TARGET == "civilian": y = (d["civ_next"].to_numpy() > 0).astype(int)
elif TARGET == "onset": y = d["violence_next_month"].to_numpy().astype(int)
else:
    assert d["fat_next"].notna().all(), "missing next-month fatalities"
    y = (d["fat_next"].to_numpy() >= 25).astype(int)
ym = d["year_month"].astype(str).to_numpy()
w_tr = (ym >= TRAIN_START) & (ym <= TRAIN_END); w_stop = (ym >= STOP_START) & (ym <= STOP_END)
w_stack = (ym >= STACK_START) & (ym <= STACK_END); w_te = ym >= TEST_START
assert not (w_tr & w_stop).any() and not (w_stop & w_stack).any() and not (w_stack & w_te).any()
for nm, w in (("train", w_tr), ("stop", w_stop), ("stack", w_stack), ("test", w_te)):
    print(f"[split] {nm}: rows={w.sum():,} positives={int(y[w].sum()):,} rate={y[w].mean():.4f}", flush=True)
    assert w.sum() > 0 and y[w].sum() > 0, f"{nm} window has no positives"
gids = d["grid_id"].to_numpy().astype(np.int64); reg = d["region"].to_numpy()

# ---- samples (depend on the target only) ----
rng = np.random.default_rng([SAMPLE_SEED, TARGETS.index(TARGET)])
pos = np.where(w_tr & (y == 1))[0]; neg = np.where(w_tr & (y == 0))[0]
if len(pos) > POS_CAP: pos = rng.choice(pos, POS_CAP, replace=False)
neg = rng.choice(neg, min(len(neg), NEG_PER_POS * len(pos)), replace=False)
tr_idx = np.sort(np.concatenate([pos, neg]))
sp = np.where(w_stop & (y == 1))[0]; sn = np.where(w_stop & (y == 0))[0]
st_idx = np.sort(np.concatenate([rng.choice(sp, max(STOP_MIN_POS, int(round(STOP_SHARE * len(sp)))) if len(sp) > STOP_MIN_POS else len(sp), replace=False),
                                 rng.choice(sn, int(round(STOP_SHARE * len(sn))), replace=False)]))
assert y[st_idx].sum() >= min(STOP_MIN_POS, len(sp)), "stop sample below positive floor"
assert len(np.intersect1d(tr_idx, st_idx)) == 0
out_idx = np.where(w_stack | w_te)[0]
if SMOKE: out_idx = np.sort(np.random.default_rng(0).choice(out_idx, min(3000, len(out_idx)), replace=False))
# ---- event sequences, streamed for the needed rows only (the full column does not fit comfortably in memory) ----
import pyarrow.parquet as pq
def ym_int(a): return np.char.replace(np.asarray(a).astype("U7"), "-", "").astype(np.int64)
need = np.unique(np.concatenate([tr_idx, st_idx, out_idx]))
need_key = gids[need] * 1000000 + ym_int(ym[need])
assert len(np.unique(need_key)) == len(need), "duplicate (grid, month) keys among needed rows"
pos_of = pd.Series(need, index=need_key)
seqs = np.empty(len(d), dtype=object); filled = np.zeros(len(d), bool)
for bt in pq.ParquetFile(f"{R}/data_intermediate/baseline_features_super.parquet").iter_batches(
        batch_size=200000, columns=["grid_id", "year_month", "event_sequence"]):
    bk = bt.column("grid_id").to_numpy().astype(np.int64) * 1000000 + ym_int(bt.column("year_month").to_numpy(zero_copy_only=False))
    hit = np.isin(bk, need_key)
    if not hit.any(): continue
    rows = pos_of.loc[bk[hit]].to_numpy(); es = bt.column("event_sequence").to_numpy(zero_copy_only=False)[hit]
    for r_, e_ in zip(rows, es): seqs[r_] = compact_seq(e_)
    filled[rows] = True
assert filled[need].all(), f"{int((~filled[need]).sum())} needed rows have no event sequence"
import hashlib
def data_version():
    out = []
    for f in ("baseline_features_super.parquet", "baseline_features.parquet"):
        st_ = os.stat(f"{R}/data_intermediate/{f}"); out.append(f"{f}:{st_.st_size}:{int(st_.st_mtime)}")
    return "|".join(out)
def sample_hash(keys_g, keys_ym):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(keys_g.astype(np.int64)).tobytes()); h.update("|".join(keys_ym).encode())
    h.update(f"{TARGET}|{POS_CAP}|{NEG_PER_POS}|{SAMPLE_SEED}|{TRAIN_START}|{TRAIN_END}|{data_version()}".encode())
    return h.hexdigest()
SAMPLE_HASH = sample_hash(gids[tr_idx], ym[tr_idx].astype(str))
print(f"[provenance] train-sample hash {SAMPLE_HASH[:16]} ({data_version()})", flush=True)
if E("ENS_PREFLIGHT", "0") == "1":                      # memory / data check without a GPU; writes nothing
    import resource
    lens = np.array([len(seqs[i]) for i in need])
    print(f"[preflight] rows needed={len(need):,} mean events={lens.mean():.1f} max={lens.max()} "
          f"peak RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f} GB", flush=True)
    sys.exit(0)
print(f"[sample] train={len(tr_idx):,} (pos {int(y[tr_idx].sum()):,}) stop={len(st_idx):,} (pos {int(y[st_idx].sum()):,}) "
      f"scored={len(out_idx):,}", flush=True)

def new_clf():
    enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, STATIC_DIM).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    miss, _ = enc.load_state_dict(sd, strict=False)
    assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
    return Clf(enc).to(DEVICE)

def predict(clf, ds):
    clf.eval(); out = []
    with torch.no_grad():
        for t, m, st, _ in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS):
            out.append(torch.sigmoid(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE))).float().cpu().numpy())
    return np.concatenate(out)

ds_tr = SeqDS(seqs[tr_idx], gids[tr_idx], y[tr_idx]); ds_st = SeqDS(seqs[st_idx], gids[st_idx]); y_st = y[st_idx]
log_rows, best = [], None          # best = (stop PR, lr, epoch, state_dict)
for lr in LRS:
    torch.manual_seed(ENC_SEED * 1000 + LRS.index(lr))
    clf = new_clf(); crit = nn.BCEWithLogitsLoss(); opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=WD)
    ld = DataLoader(ds_tr, batch_size=FT_BATCH, shuffle=True, num_workers=WORKERS,
                    generator=torch.Generator().manual_seed(ENC_SEED * 1000 + LRS.index(lr)))
    run_best, bad = -1.0, 0
    for ep in range(1, MAX_EP + 1):
        clf.train(); t0 = time.time(); tot, n = 0.0, 0; diverged = False
        for t, m, st, yy in ld:
            loss = crit(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)), yy.to(DEVICE))
            if not torch.isfinite(loss): diverged = True; break
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            if not torch.isfinite(gn): diverged = True; break
            opt.step(); tot += float(loss) * len(yy); n += len(yy)
        if diverged:
            log_rows.append(dict(job=JOB, lr=lr, epoch=ep, train_loss=np.nan, stop_pr=np.nan, diverged=True))
            pd.DataFrame(log_rows).to_csv(TMP_L, index=False); print(f"  lr={lr:g} DIVERGED at epoch {ep}", flush=True); break
        p_st = predict(clf, ds_st); assert np.isfinite(p_st).all(), "non-finite stop forecasts"
        v = float(average_precision_score(y_st, p_st))
        log_rows.append(dict(job=JOB, lr=lr, epoch=ep, train_loss=tot / n, stop_pr=v, diverged=False, secs=round(time.time() - t0)))
        pd.DataFrame(log_rows).to_csv(TMP_L, index=False)
        print(f"  lr={lr:g} ep={ep} loss={tot / n:.4f} stopPR={v:.4f} {time.time() - t0:.0f}s", flush=True)
        if best is None or v > best[0]:
            best = (v, lr, ep, {k: t_.detach().cpu().clone() for k, t_ in clf.state_dict().items()})
        bad = 0 if v > run_best + MIN_DELTA else bad + 1
        run_best = max(run_best, v)
        if bad >= PATIENCE: break
    del clf, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
assert best is not None, "every learning rate diverged before its first epoch finished"
print(f"[best] lr={best[1]:g} epoch={best[2]} stopPR={best[0]:.4f}", flush=True)

clf = new_clf(); clf.load_state_dict(best[3])
p_out = predict(clf, SeqDS(seqs[out_idx], gids[out_idx]))
assert len(p_out) == len(out_idx) and np.isfinite(p_out).all()
win = np.where(w_te[out_idx], "test", "stack")
for w in ("stack", "test"):
    s = win == w
    print(f"[score] {w}: PR-AUC={average_precision_score(y[out_idx][s], p_out[s]):.4f} n={s.sum():,} pos={int(y[out_idx][s].sum()):,}", flush=True)

# ---- publish: weights, log, then forecasts (the forecasts file is the completion marker) ----
pd.DataFrame(log_rows).to_csv(TMP_L, index=False)        # complete log, including any failed trials
torch.save(dict(state_dict=best[3], lr=best[1], epoch=best[2], stop_pr=best[0], tag=TAG, target=TARGET), TMP_W)
np.savez_compressed(TMP_P, grid_id=gids[out_idx], year_month=ym[out_idx].astype("U7"), region=reg[out_idx].astype("U12"),
                    window=win.astype("U5"), y=y[out_idx].astype(np.int8), p=p_out.astype(np.float32),
                    train_rows=tr_idx.size, stop_rows=st_idx.size, sample_hash=np.array(SAMPLE_HASH),
                    data_version=np.array(data_version()), pos_cap=POS_CAP, neg_per_pos=NEG_PER_POS)
chk = np.load(TMP_P); assert chk["p"].shape == chk["y"].shape == chk["grid_id"].shape
os.replace(TMP_W, OUT_W); os.replace(TMP_L, OUT_L); os.replace(TMP_P, OUT_P)
print(f"[done] {OUT_P}", flush=True)

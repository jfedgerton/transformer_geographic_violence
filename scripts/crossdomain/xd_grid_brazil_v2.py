"""xd_grid_brazil.py (2026-10-23) -- cross-domain transfer test C3: conflict encoder -> Brazil homicide (SIM),
PRIO-GRID cell-months built by build_brazil_panel.py. Design frozen in crossdomain_plan_2026-10-23.md and
settled_decisions.md (fine-tuning protocol + scope, 2026-10-23).

One job = one encoder (XD_ENC = americas | all) x one seed x one ARM x one MAPPING (rich | min):
  ARM=nested    frozen CLS embedding + base features vs base features (XGBoost count:poisson), plus the same with a
                random-weight encoder (control); shares 0.25/1/5/10% (5 draws) and 100% (1 draw)
  ARM=finetune  ONE share (GRID_SHARES), 5 draws, four arms with a Poisson head on the CLS vector:
                  top2       pretrained, embeddings + field_proj + cls/prio + transformer layers 0..3 FROZEN,
                             layers 4..5 + final LayerNorm + head trained            (MAIN fine-tune arm)
                  full       pretrained, everything trained                           (appendix)
                  scratch    same architecture, random init, everything trained       (comparison)
                  top2_rand  random init, same freezing as top2                       (control: frozen random features)
                lr {3e-5,1e-4,3e-4,1e-3} x wd {0,0.01} sweep on draw 0 (chosen config reused for draws 1-4), early
                stopping on 2021 at the same share (patience 5, max 50), best-val weights restored.
Target: deaths in month t+1 (count). Scores on test (2022-01..2024-11): mean Poisson deviance (primary, lower is
better), CRPS from 100 Poisson draws (seed 123). Nested also scores the surge target (PR-AUC, surge-eligible rows).
Splits: train 2001-01..2020-12, val 2021, test 2022-01..2024-11. Draws: stratified on any death next month; depend
only on (share, draw), never on encoder/seed/arm.
Vocabulary: tokens the encoder never saw in pretraining -> <UNK> (americas: seen_tokens_fullsrc_loro_americas.json;
all-region: actor ids that occur in acled_with_grid events dated <= 2020; other fields as in the americas file).
Outputs (outputs/crossdomain/, never overwritten): metrics_{JOB}.csv, preds_{JOB}.npz, sweep_{JOB}.csv.
Env XD_TEST=1 runs on synthetic inputs (XD_PANEL, XD_CKPT, XD_SEEN, XD_VOCAB, XD_STATIC, XD_OUTDIR, XD_CAP).
"""
import os, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
E = os.environ.get
TEST = E("XD_TEST", "0") == "1"
ENC = E("XD_ENC", "americas"); SEED = int(E("EVAL_SEED", "123")); ARM = E("GRID_ARM", "nested")
MAPPING = E("XD_MAPPING", "rich")
# 2026-10-24 favorable-construction ladder (crossdomain_run_2026-10-23.md; declared after the nested results):
#   XD_BASE  rich  = 48 lags + cal_month + year + statics (C0, pre-registered)
#            paper = the conflict paper's nested base rebuilt from Brazil deaths + statics (C1)
#   XD_PCA   0 = raw 256-d embedding; k > 0 = PCA(k) fit on each draw's TRAINING rows only (C2)
#   XD_MAPPING rich_proxy / min_proxy = Brazil actor ids replaced by seen non-Americas proxies (C3, americas only)
BASE = E("XD_BASE", "rich"); PCA_K = int(E("XD_PCA", "0"))
assert ENC in ("americas", "all") and ARM in ("nested", "finetune") and MAPPING in ("rich", "min", "rich_proxy", "min_proxy")
assert BASE in ("rich", "paper") and PCA_K >= 0
assert not MAPPING.endswith("_proxy") or ENC == "americas", "proxy mapping is only defined for the Americas-held-out encoder"
if ARM == "finetune": assert BASE == "rich" and PCA_K == 0, "XD_BASE / XD_PCA only apply to the nested arm"
TAG = f"L_contrastive_e_fullsrc_loro_americas_s{SEED}" if ENC == "americas" else f"L_contrastive_e_fullsrc_all_s{SEED}"
CKPT = E("XD_CKPT", f"{R}/data_model/repr_model_{TAG}.pt")
PANEL = E("XD_PANEL", f"{R}/data_intermediate/xd_brazil_panel.parquet")
VOCAB = E("XD_VOCAB", f"{R}/data_intermediate/vocab_super.json")
SEEN = E("XD_SEEN", f"{R}/data_intermediate/seen_tokens_fullsrc_loro_americas.json")
STATIC = E("XD_STATIC", f"{R}/data_intermediate/priogrid_static.parquet")
OD = E("XD_OUTDIR", f"{R}/outputs/crossdomain")
default_shares = "0.0025,0.01,0.05,0.10,1.0" if ARM == "nested" else "0.01"
SHARES = [float(x) for x in E("GRID_SHARES", default_shares).split(",")]
if ARM == "finetune": assert len(SHARES) == 1 and SHARES[0] < 1.0, "finetune jobs take exactly one share < 1"
DRAWS = [int(x) for x in E("GRID_DRAWS", "0,1,2,3,4").split(",")]
LRS = [float(x) for x in E("GRID_LRS", "3e-5,1e-4,3e-4,1e-3").split(",")]
WDS = [float(x) for x in E("GRID_WDS", "0,0.01").split(",")]
MAX_EP = int(E("GRID_MAX_EP", "50")); PATIENCE = int(E("GRID_PATIENCE", "5")); MIN_DELTA = 1e-4
MIN_TRAIN_POS, VAL_MIN_POS, DRAW_BASE, XGB_SEED, CRPS_SEED, N_DRAWS_CRPS = 5, 10, 777, 123, 123, 100
CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[E("XD_CAP", "L")]
N_TRAIN_LAYERS = 2                                     # top2: last two transformer layers trained
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = "2001-01", "2020-12", "2021-01", "2021-12", "2022-01", "2024-11"
FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type", "actor2_type", "month_offset"]
SEQ_COL = "seq_rich" if MAPPING.startswith("rich") else "seq_min"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(OD, exist_ok=True)
JOB = (f"brazil_{ARM}_{ENC}_s{SEED}_{MAPPING}" + (f"_f{SHARES[0]:g}" if ARM == "finetune" else "")
       + ("_bpaper" if BASE == "paper" else "") + (f"_pca{PCA_K}" if PCA_K else ""))
OUT_M, OUT_P, OUT_S = f"{OD}/metrics_{JOB}.csv", f"{OD}/preds_{JOB}.npz", f"{OD}/sweep_{JOB}.csv"
assert not os.path.exists(OUT_M), f"refusing to overwrite completed job {OUT_M}"
TMP_M, TMP_P, TMP_S = OUT_M + ".part", OUT_P + ".part.npz", OUT_S + ".part"
assert not any(os.path.exists(p_) for p_ in (TMP_M, TMP_P, TMP_S)), f"in-flight .part files exist for {JOB} (another task running?)"
assert os.path.exists(CKPT), f"missing encoder {CKPT}"
print(f"[cfg] JOB={JOB} TAG={TAG} shares={SHARES} draws={DRAWS} lrs={LRS} wds={WDS} dev={DEVICE}", flush=True)

# ---- vocabulary remap (unseen in pretraining -> <UNK>) ----
V = json.load(open(VOCAB))
vsm = {k: max(v.values()) + 1 + 1 for k, v in V.items()}                     # +1 mask slot, as in the trainer
seen = json.load(open(SEEN))
for f in FIELDS:
    if f != "month_offset": assert V[f].get("<UNK>") == 1 and V[f].get("<PAD>") == 0, f"{f}: <UNK>/<PAD> ids not 1/0"
assert seen["region"] == "americas", f"{SEEN} is for {seen['region']}"
LUT = []
for f in FIELDS:
    if f == "month_offset":
        LUT.append(np.arange(vsm[f])); continue
    ok = np.zeros(vsm[f], bool); ok[np.asarray(seen["tokens"][f], np.int64)] = True
    if ENC == "all" and f in ("actor1", "actor2"):
        if TEST:
            ok[:] = True
        else:
            import pyarrow.parquet as pq
            ev_path = f"{R}/data_intermediate/acled_with_grid.parquet"
            cols = pq.read_schema(ev_path).names
            dc = next(c for c in ["event_date", "date", "year_month", "year"] if c in cols)
            ev = pd.read_parquet(ev_path, columns=[f, dc])
            yr = ev["year"] if dc == "year" else pd.to_datetime(ev[dc].astype(str), errors="coerce").dt.year
            names = set(ev.loc[yr <= 2020, f].astype(str).unique())
            ok = np.zeros(vsm[f], bool)
            for n, j in V[f].items():
                if n in names: ok[j] = True
    ok[0] = True; ok[vsm[f] - 1] = True
    LUT.append(np.where(ok, np.arange(vsm[f]), 1))
for name, f in [("A1 UAG Brazil", ("actor1", "Unidentified Armed Group (Brazil)")), ("A1 Gang Brazil", ("actor1", "Unidentified Gang (Brazil)")),
                ("A1 MP 2016", ("actor1", "Military Forces of Brazil (2016-2018) Military Police")),
                ("A1 MP 2019", ("actor1", "Military Forces of Brazil (2019-2022) Military Police")),
                ("A1 MP 2023", ("actor1", "Military Forces of Brazil (2023-) Military Police")),
                ("A2 Civ Brazil", ("actor2", "Civilians (Brazil)")), ("A2 Gang Brazil", ("actor2", "Unidentified Gang (Brazil)"))]:
    j = V[f[0]][f[1]]; print(f"[vocab] {ENC} {name} id {j} -> {int(LUT[FIELDS.index(f[0])][j])}", flush=True)

# ---- C3 proxy mapping (Americas encoder): Brazil actor -> most frequent seen token of the same ACLED role ----
# Rule fixed before any C3 result: lowest vocab id (vocab is frequency-ordered) among seen tokens of that role;
# actor1 gang-pattern token gets the 2nd most frequent armed-group token so rich != min (actor2 gang gets the 1st). Civilians (Brazil) is seen -> kept.
PROXY = {"actor1": {"Unidentified Armed Group (Brazil)": "Unidentified Armed Group (Syria)",
                    "Unidentified Gang (Brazil)": "Unidentified Armed Group (Nigeria)",
                    "Military Forces of Brazil (2016-2018) Military Police": "Police Forces of the Philippines (2016-2022)",
                    "Military Forces of Brazil (2019-2022) Military Police": "Police Forces of the Philippines (2016-2022)",
                    "Military Forces of Brazil (2023-) Military Police": "Police Forces of the Philippines (2016-2022)"},
         "actor2": {"Unidentified Gang (Brazil)": "Unidentified Armed Group (Somalia)"}}
if MAPPING.endswith("_proxy"):
    def _rank(f, prefix, k):  # k-th most frequent (lowest id) seen token of a role; vocab ids are frequency-ordered
        ss = set(int(x) for x in seen["tokens"][f])
        c = sorted(j for n, j in V[f].items() if j in ss and n.startswith(prefix))
        assert len(c) > k, (f, prefix); return c[k]
    assert V["actor1"][PROXY["actor1"]["Unidentified Armed Group (Brazil)"]] == _rank("actor1", "Unidentified Armed Group (", 0)
    assert V["actor1"][PROXY["actor1"]["Unidentified Gang (Brazil)"]] == _rank("actor1", "Unidentified Armed Group (", 1)
    assert V["actor1"][PROXY["actor1"]["Military Forces of Brazil (2019-2022) Military Police"]] == _rank("actor1", "Police Forces of", 0)
    assert V["actor2"][PROXY["actor2"]["Unidentified Gang (Brazil)"]] == _rank("actor2", "Unidentified Armed Group (", 0)
    if not TEST:  # frequency ordering of the vocab: ids of the main event types must rise with ACLED frequency
        assert V["actor2"]["Civilians (Syria)"] < V["actor2"]["Civilians (Palestine)"], "vocab not frequency-ordered?"
    for f, mp in PROXY.items():
        fi = FIELDS.index(f); seen_ids = set(int(x) for x in seen["tokens"][f])
        P = np.arange(vsm[f])
        for src, dst in mp.items():
            s_, d_ = V[f][src], V[f][dst]
            assert d_ in seen_ids, f"proxy {dst} not seen by the Americas encoder"
            assert s_ not in seen_ids, f"{src} is already seen; no proxy needed"
            P[s_] = d_
        LUT[fi] = LUT[fi][P]
        for src, dst in mp.items():
            assert LUT[fi][V[f][src]] == V[f][dst], (f, src)
            print(f"[proxy] {f} {src} (id {V[f][src]}) -> {dst} (id {V[f][dst]})", flush=True)
    assert LUT[FIELDS.index("actor2")][V["actor2"]["Civilians (Brazil)"]] == V["actor2"]["Civilians (Brazil)"], "Civilians (Brazil) should be seen"

# ---- data ----
ps = pd.read_parquet(STATIC)
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
assert ps["grid_id"].is_unique
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}
d = pd.read_parquet(PANEL)
# ---- C1 paper-matched base, built on the full 2000+ panel before the date filter (all inputs at t-1 or earlier) ----
# Conflict base (transfer_eval_region_fullpreds.py base_cols) -> Brazil analogue (death = event):
#   violence_lag1/3/6 -> any death at t-1/t-3/t-6; spatial_lag_violence -> share of queen neighbours (cells in the
#   panel) with any death at t-1; violent_count_roll1 / all_count_roll1 -> deaths t-1; violent_count_roll3 -> deaths
#   t-1..t-3; any_event_lag1 -> any death t-1; log_event_density_lag1 -> log1p(deaths t-1);
#   months_since_last_event -> months since the last month <= t-1 with a death (no death since 2000-01 -> months
#   since 1999-12). roll1/all_count and any_event duplicate other columns because Brazil has one event type.
d = d.sort_values(["grid_id", "t"]).reset_index(drop=True)
assert (d.groupby("grid_id")["t"].diff().dropna() == 1).all(), "panel not contiguous per cell"
assert np.allclose(d.groupby("grid_id")["deaths_t"].shift(1).dropna(), d.loc[d.groupby("grid_id")["deaths_t"].shift(1).notna(), "lag1_deaths"]), "lag1_deaths != deaths at t-1"
any1 = (d["lag1_deaths"] > 0).astype(float)
d["pb_violence_lag1"] = any1
d["pb_violence_lag3"] = (d["lag3_deaths"] > 0).astype(float)
d["pb_violence_lag6"] = (d["lag6_deaths"] > 0).astype(float)
d["pb_violent_count_roll1"] = d["lag1_deaths"].fillna(0).astype(float)
d["pb_violent_count_roll3"] = (d["lag1_deaths"].fillna(0) + d["lag2_deaths"].fillna(0) + d["lag3_deaths"].fillna(0)).astype(float)
d["pb_all_count_roll1"] = d["lag1_deaths"].fillna(0).astype(float)
d["pb_any_event_lag1"] = any1
d["pb_log_event_density_lag1"] = np.log1p(d["lag1_deaths"].fillna(0).astype(float))
_last = d["t"].where(d["deaths_t"] > 0).groupby(d["grid_id"]).transform(lambda s: s.ffill().shift(1))   # last death month <= t-1
d["pb_months_since_last_event"] = (d["t"] - _last.fillna(-1)).astype(float)
_k = pd.Series(any1.values, index=pd.MultiIndex.from_arrays([d["grid_id"].values, d["t"].values]))
_nb = np.zeros(len(d)); _nn = np.zeros(len(d))
for off in (-721, -720, -719, -1, 1, 719, 720, 721):
    v = _k.reindex(pd.MultiIndex.from_arrays([d["grid_id"].values + off, d["t"].values])).to_numpy()
    _nb += np.nan_to_num(v); _nn += np.isfinite(v)
d["pb_spatial_lag_violence"] = np.where(_nn > 0, _nb / np.maximum(_nn, 1), 0.0)
PB_COLS = ["pb_violence_lag1", "pb_violence_lag3", "pb_violence_lag6", "pb_spatial_lag_violence", "pb_violent_count_roll1",
           "pb_violent_count_roll3", "pb_all_count_roll1", "pb_any_event_lag1", "pb_log_event_density_lag1",
           "pb_months_since_last_event"]
assert d[PB_COLS].notna().all().all() and (d["pb_months_since_last_event"] >= 1).all()
d = d[(d["year_month"] >= TRAIN_START) & (d["year_month"] <= TEST_END)].sort_values(["year_month", "grid_id"]).reset_index(drop=True)
n0 = len(d); d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0
miss_static = float(d[static_cols[0]].isna().mean()); print(f"[data] rows={len(d):,} cells without statics={miss_static:.4f}", flush=True)
lag_cols = [c for c in d.columns if c.startswith("lag")]
assert len(lag_cols) == 48, f"expected 48 lag columns, got {len(lag_cols)}"
FEATS = lag_cols + ["cal_month", "year"] + static_cols        # cell_train_mean excluded: full-sample, includes train targets
if BASE == "paper": FEATS = PB_COLS + static_cols            # C1: the conflict paper's nested base (+ PRIO statics)
d[FEATS] = d[FEATS].fillna(0.0)
print(f"[base] {BASE}: {len(FEATS)} features (statics {len(static_cols)}); PCA_K={PCA_K}", flush=True)
X = d[FEATS].to_numpy(np.float32)
ym = d["year_month"].to_numpy().astype(str)
y = d["deaths_next"].to_numpy().astype(np.float64)
ybin = (y > 0).astype(int)
trm = (ym >= TRAIN_START) & (ym <= TRAIN_END); vam = (ym >= VAL_START) & (ym <= VAL_END); tem = (ym >= TEST_START) & (ym <= TEST_END)
MINR = (20, 10, 20) if TEST else (1000, 100, 1000)
assert not (trm & vam).any() and not (vam & tem).any() and trm.sum() > MINR[0] and vam.sum() > MINR[1] and tem.sum() > MINR[2]
seqs = d[SEQ_COL].values; gids = d["grid_id"].values
te_idx = np.where(tem)[0]; yte = y[te_idx]
print(f"[data] train={trm.sum():,} val={vam.sum():,} test={len(te_idx):,} test mean deaths={yte.mean():.3f} "
      f"test any-death={ybin[te_idx].mean():.3f}", flush=True)
TEST_ID = dict(__grid_id=gids[te_idx].astype(np.int64), __year_month=ym[te_idx].astype("U7"), __y=yte.astype(np.int32))


def draw_rows(mask, share, k, purpose):
    """Stratified (any death next month) draw; depends only on (share, draw k, purpose)."""
    pool = np.where(mask)[0]
    if share >= 1.0: return pool
    rng = np.random.default_rng([DRAW_BASE, int(round(share * 1e5)), k, purpose])
    pos = pool[ybin[pool] == 1]; neg = pool[ybin[pool] == 0]
    npos = int(round(len(pos) * share)); nneg = int(round(len(neg) * share))
    if purpose == 1 and npos < VAL_MIN_POS:
        npos = min(VAL_MIN_POS, len(pos)); nneg = min(len(neg), int(round(npos * len(neg) / max(len(pos), 1))))
    if purpose == 0 and npos < MIN_TRAIN_POS: return None
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]))


def poisson_dev(yt, mu):
    mu = np.clip(mu, 1e-8, None)
    t = np.where(yt > 0, yt * np.log(np.where(yt > 0, yt, 1.0) / mu), 0.0)
    return float(np.mean(2.0 * (t - (yt - mu))))


def crps_poisson(yt, mu):
    rng = np.random.default_rng(CRPS_SEED)
    dr = rng.poisson(np.clip(mu, 0, None)[:, None], size=(len(mu), N_DRAWS_CRPS)).astype(float)
    a = np.abs(dr - yt[:, None]).mean(1)
    srt = np.sort(dr, 1); n = N_DRAWS_CRPS; w = (2 * np.arange(1, n + 1) - n - 1) / (n * n)
    b = (srt * w).sum(1)                                   # = 0.5 * E|X - X'| for the empirical draws
    return float(np.mean(a - b))


rows, preds, sweep_rows = [], {}, []
def emit(model, share, k, idx, mu, **extra):
    mu = np.asarray(mu, np.float64)
    assert len(mu) == len(yte) and np.isfinite(mu).all() and (mu >= 0).all(), (model, share, k)
    r = dict(domain="brazil", encoder=ENC, enc_seed=SEED, mapping=MAPPING, base=BASE, pca=PCA_K, arm=ARM, model=model, share=share, draw=k,
             n_train=0 if idx is None else int(len(idx)), n_test=int(len(yte)), test_mean=float(yte.mean()),
             deviance=poisson_dev(yte, mu), crps=crps_poisson(yte, mu), **extra)
    rows.append(r); preds[f"{model}|{share:g}|{k}"] = mu.astype(np.float32)
    pd.DataFrame(rows).to_csv(TMP_M, index=False)
    print(f"  [{model}] share={share:g} draw={k} n={r['n_train']} dev={r['deviance']:.4f} crps={r['crps']:.4f}", flush=True)


emit("persistence", 0.0, -1, None, d["deaths_t"].to_numpy()[te_idx].astype(float))
emit("cell_train_mean", 0.0, -1, None, d["cell_train_mean"].to_numpy()[te_idx].astype(float))


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, use_prio, static_dim, dropout=0.1):
        super().__init__()
        self.fields, self.use_prio = fields, use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio: self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4, dropout=dropout, batch_first=True)
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


class Reg(nn.Module):
    def __init__(self, enc, bias0):
        super().__init__(); self.enc = enc; self.head = nn.Linear(enc.ln.normalized_shape[0], 1)
        nn.init.zeros_(self.head.weight); nn.init.constant_(self.head.bias, bias0)
    def forward(self, seq, mask, static): return self.head(self.enc.cls_vec(seq, mask, static)).squeeze(-1)  # log-rate


def tokenize(idx):
    """JSON sequences -> (n, MAX_SEQ_LEN, 7) int16 token ids after the remap, and (n, MAX_SEQ_LEN) masks. Done once."""
    T = np.zeros((len(idx), MAX_SEQ_LEN, len(FIELDS)), np.int16); M = np.zeros((len(idx), MAX_SEQ_LEN), bool)
    for r, i in enumerate(idx):
        sq = seqs[i]
        if not isinstance(sq, str) or sq == "[]": continue
        a = np.asarray(json.loads(sq), np.int64)[-MAX_SEQ_LEN:, :len(FIELDS)]
        n = len(a); M[r, :n] = True
        for f in range(len(FIELDS)): T[r, :n, f] = LUT[f][a[:, f]]
    return T, M


STAT = np.stack([np.nan_to_num(prio_lookup.get(int(g), np.zeros(STATIC_DIM, np.float32))) for g in gids]).astype(np.float32)


class SeqDS(Dataset):
    def __init__(self, idx, labels=None):
        self.T, self.M = tokenize(idx); self.S = STAT[idx]
        self.labels = None if labels is None else np.asarray(labels, np.float32)
    def __len__(self): return len(self.T)
    def __getitem__(self, i):
        return (torch.from_numpy(self.T[i].astype(np.int64)), torch.from_numpy(self.M[i]), torch.from_numpy(self.S[i]),
                torch.tensor(0.0 if self.labels is None else self.labels[i]))


def new_encoder(pretrained, seed_offset=0):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(SEED + seed_offset)
        enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, True, STATIC_DIM).to(DEVICE)
    if pretrained:
        sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        miss, unexp = enc.load_state_dict(sd, strict=False)
        assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
    return enc


def embed(enc, idx):
    enc.eval(); Z = []
    with torch.no_grad():
        for t, m, st, _ in DataLoader(SeqDS(idx), batch_size=BATCH, shuffle=False, num_workers=0):
            Z.append(enc.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(Z); assert Z.shape == (len(idx), EMBED_DIM) and np.isfinite(Z).all(); return Z


def xgb_fit(Xtr, ytr, Xte, objective):
    b = xgb.train({"objective": objective, "max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                   "seed": XGB_SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
    return b.predict(xgb.DMatrix(Xte))


if ARM == "nested":
    draws = {}
    for s in SHARES:
        for k in ([0] if s >= 1.0 else DRAWS):
            draws[(s, k)] = draw_rows(trm, s, k, 0)
    need = np.unique(np.concatenate([v for v in draws.values() if v is not None] + [te_idx]))
    t0 = time.time()
    Zp = embed(new_encoder(True), need); Zr = embed(new_encoder(False, seed_offset=100000), need)
    print(f"[emb] {Zp.shape} pretrained + random in {time.time()-t0:.0f}s", flush=True)
    posn = -np.ones(len(d), np.int64); posn[need] = np.arange(len(need))
    Zof = lambda Z, ix: Z[posn[ix]]
    surge = d["surge_next"].to_numpy(); elig_te = np.isfinite(surge[te_idx])
    from sklearn.decomposition import PCA
    def zpair(Z, idx):
        """Embedding for the draw's training rows and the test rows; C2: PCA fit on the training rows only."""
        Ztr, Zte = Zof(Z, idx), Zof(Z, te_idx)
        if not PCA_K: return Ztr, Zte
        pca = PCA(n_components=PCA_K, random_state=XGB_SEED).fit(Ztr)
        return pca.transform(Ztr).astype(np.float32), pca.transform(Zte).astype(np.float32)
    for (s, k), idx in draws.items():
        if idx is None: print(f"  [skip] share={s:g} draw={k}", flush=True); continue
        (Pp_tr, Pp_te), (Pr_tr, Pr_te) = zpair(Zp, idx), zpair(Zr, idx)
        for name, Xtr, Xte in [("base", X[idx], X[te_idx]),
                               ("emb_base", np.c_[Pp_tr, X[idx]], np.c_[Pp_te, X[te_idx]]),
                               ("rand_emb_base", np.c_[Pr_tr, X[idx]], np.c_[Pr_te, X[te_idx]])]:
            mu = xgb_fit(Xtr, y[idx], Xte, "count:poisson")
            el = np.isfinite(surge[idx]); pr = np.nan
            if el.sum() > 50 and np.nansum(surge[idx][el]) >= MIN_TRAIN_POS and elig_te.sum() > 50:
                ps_ = xgb_fit(Xtr[el], surge[idx][el], Xte[elig_te], "binary:logistic")
                pr = float(average_precision_score(surge[te_idx][elig_te], ps_))
            emit(name, s, k, idx, mu, surge_pr=pr)
    st = d["surge_t"].to_numpy()[te_idx][elig_te]
    rows.append(dict(domain="brazil", encoder=ENC, enc_seed=SEED, mapping=MAPPING, base=BASE, pca=PCA_K, arm=ARM, model="surge_persistence",
                     share=0.0, draw=-1, surge_pr=float(average_precision_score(surge[te_idx][elig_te], np.nan_to_num(st)))))
    pd.DataFrame(rows).to_csv(TMP_M, index=False)

else:
    s = SHARES[0]
    ds_te = SeqDS(te_idx)
    def predict(m, ds):
        m.eval(); out = []
        with torch.no_grad():
            for t, mk, st, _ in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0):
                out.append(torch.exp(m(t.to(DEVICE), mk.to(DEVICE), st.to(DEVICE)).clamp(-20.0, 12.0)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    def set_trainable(reg, mode):
        for p in reg.parameters(): p.requires_grad = True
        if mode in ("top2", "top2_rand"):
            for p in reg.enc.parameters(): p.requires_grad = False
            for layer in reg.enc.transformer.layers[N_LAYERS - N_TRAIN_LAYERS:]:
                for p in layer.parameters(): p.requires_grad = True
            for p in reg.enc.ln.parameters(): p.requires_grad = True
        n_tr = sum(p.numel() for p in reg.parameters() if p.requires_grad); n_all = sum(p.numel() for p in reg.parameters())
        return n_tr, n_all
    def train_one(mode, lr, wd, tr, va, k):
        torch.manual_seed(SEED * 1000 + k)
        pretrained = mode in ("top2", "full")
        reg = Reg(new_encoder(pretrained, seed_offset=0 if pretrained else 100000), float(np.log(max(y[tr].mean(), 1e-3)))).to(DEVICE)
        n_tr, n_all = set_trainable(reg, mode)
        frozen0 = reg.enc.embs['actor1'].weight.detach().clone() if mode in ('top2', 'top2_rand') else None
        opt = torch.optim.AdamW([p for p in reg.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        crit = nn.PoissonNLLLoss(log_input=True, full=False)
        ld = DataLoader(SeqDS(tr, y[tr]), batch_size=FT_BATCH, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED * 1000 + k))
        ds_va = SeqDS(va); yva = y[va]
        best, best_ep, bad = np.inf, 0, 0
        best_sd = {kk: t_.detach().cpu().clone() for kk, t_ in reg.state_dict().items()}
        diverged = False
        for ep in range(1, MAX_EP + 1):
            reg.train()
            if mode in ("top2", "top2_rand"):                # frozen parts stay in eval mode (no dropout noise)
                reg.enc.eval()
                for layer in reg.enc.transformer.layers[N_LAYERS - N_TRAIN_LAYERS:]: layer.train()
            for t, mk, st, yy in ld:
                logit = reg(t.to(DEVICE), mk.to(DEVICE), st.to(DEVICE)).clamp(-20.0, 12.0)
                loss = crit(logit, yy.to(DEVICE))
                if not torch.isfinite(loss): diverged = True; break
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in reg.parameters() if p.requires_grad], 1.0); opt.step()
            if diverged: break
            pv = predict(reg, ds_va)
            if not np.isfinite(pv).all(): diverged = True; break
            v = poisson_dev(yva, pv)
            improved = v < best - MIN_DELTA
            if v < best: best, best_ep, best_sd = v, ep, {kk: t_.detach().cpu().clone() for kk, t_ in reg.state_dict().items()}
            bad = 0 if improved else bad + 1
            if bad >= PATIENCE: break
        reg.load_state_dict(best_sd)
        if diverged and best_ep == 0: best = np.inf
        if frozen0 is not None:
            assert torch.equal(frozen0, reg.enc.embs['actor1'].weight.detach()), 'frozen weights changed'
        return reg, best, best_ep, ep, n_tr, n_all
    chosen = {}
    for k in DRAWS:
        tr = draw_rows(trm, s, k, 0); va = draw_rows(vam, s, k, 1)
        if tr is None or va is None: print(f"  [skip] share={s:g} draw={k}", flush=True); continue
        for mode in ("top2", "full", "scratch", "top2_rand"):
            grid = [(lr, wd) for lr in LRS for wd in WDS] if mode not in chosen else [chosen[mode]]
            best = None
            for lr, wd in grid:
                t0 = time.time()
                reg, vdev, bep, lep, n_tr, n_all = train_one(mode, lr, wd, tr, va, k)
                sweep_rows.append(dict(encoder=ENC, enc_seed=SEED, mapping=MAPPING, model=mode, share=s, draw=k, lr=lr, wd=wd,
                                       val_dev=vdev, best_epoch=bep, last_epoch=lep, n_train=len(tr), n_val=len(va),
                                       n_trainable=n_tr, n_params=n_all, secs=round(time.time() - t0)))
                pd.DataFrame(sweep_rows).to_csv(TMP_S, index=False)
                print(f"    {mode} draw={k} lr={lr:g} wd={wd:g} valDev={vdev:.4f} best_ep={bep} stop_ep={lep} "
                      f"trainable={n_tr}/{n_all} {time.time()-t0:.0f}s", flush=True)
                if best is None or vdev < best[0]: best = (vdev, lr, wd, bep, reg)
                del reg
            assert np.isfinite(best[0]), f'{mode}: every sweep config diverged'
            if mode not in chosen: chosen[mode] = (best[1], best[2])
            emit(mode, s, k, tr, predict(best[4], ds_te), lr=best[1], wd=best[2], best_epoch=best[3], val_dev=best[0], n_val=len(va))
            del best
            if torch.cuda.is_available(): torch.cuda.empty_cache()

np.savez_compressed(TMP_P, **TEST_ID, **preds)
chk = np.load(TMP_P); assert all(k in chk.files for k in preds)
if os.path.exists(TMP_S): os.replace(TMP_S, OUT_S)
os.replace(TMP_P, OUT_P); os.replace(TMP_M, OUT_M)
print(f"[done] {OUT_M} ({len(rows)} rows)", flush=True)

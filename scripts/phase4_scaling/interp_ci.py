"""interp_ci.py — LORO defense analyses. INTERP_MODE=interp | ci.

interp (gaps #1/#3/#4):
  #1 region-invariance probe: how recoverable is REGION from the frozen embedding
     vs from engineered base features (chance=1/6). Lower emb accuracy => more invariant.
  #3 event-volume confound: onset PR-AUC for volume-only vs base vs emb vs emb+base
     (base already includes log_event_density) => emb must add beyond volume.
  #4 null encoders: incidence PR-AUC for real vs RANDOM-init vs SHUFFLED-sequence encoder
     => the learned event-order structure, not the architecture, is what transfers.
  Output: outputs/interp_defense_summary.csv

ci (gap #2): bootstrap CIs on the LORO region result. For each region, load its LORO
  checkpoint, Mode-B temporal split, per-cell preds, resample cells 2000x (seed 123)
  => CI on emb AP, base AP, and deltas. Output: outputs/loro_region_ci.csv
Full A100; region evals need big mem (americas ~360G) -> ci mode caps sample at 250k/region.
"""
import os, json, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF, REGIONS

MODE = os.environ.get("INTERP_MODE", "interp")
SEED = 123; MAXLEN = 200; BATCH = 512
D, NL, NH = 256, 6, 8
FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset"]
SEL = [0,1,2,3,4,5,6]; NUM_FIELDS = 7; USE_PRIO = True
TRAIN_END, TEST_START = "2020-12", "2022-01"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
print(f"[cfg] MODE={MODE} dev={DEV}", flush=True)

vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 2 for k, v in vocabs.items()}
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32)
               for r in ps.itertuples(index=False)}

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], D, padding_idx=0) for f in FIELDS})
        self.field_proj = nn.Linear(NUM_FIELDS * D, D)
        self.proj_dropout = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        if USE_PRIO: self.prio_proj = nn.Linear(STATIC_DIM, D)
        layer = nn.TransformerEncoderLayer(D, NH, D*4, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(FIELDS)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if USE_PRIO: cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

class SeqDS(Dataset):
    def __init__(self, seqs, gids, shuffle=False):
        self.seqs = [json.loads(s) if isinstance(s, str) else [] for s in seqs]
        self.gids = gids; self.shuffle = shuffle
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAXLEN:]
        if self.shuffle and len(seq) > 1:
            seq = [seq[j] for j in rng.permutation(len(seq))]
        t = torch.zeros(MAXLEN, NUM_FIELDS, dtype=torch.long)
        m = torch.zeros(MAXLEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS): t[j, f] = ev[SEL[f]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        return t, m, st

def build_encoder(ckpt=None):
    enc = Encoder().to(DEV)
    if ckpt: enc.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True), strict=False)
    enc.eval(); return enc

def extract(enc, seqs, gids, shuffle=False):
    ld = DataLoader(SeqDS(seqs, gids, shuffle), batch_size=BATCH, shuffle=False, num_workers=0)
    Z = []
    with torch.no_grad():
        for t, m, st in ld:
            Z.append(enc.cls_vec(t.to(DEV), m.to(DEV), st.to(DEV)).cpu().numpy())
    return np.concatenate(Z, 0)

BASE = ["violence_lag1","violence_lag3","violence_lag6","spatial_lag_violence","violent_count_roll1",
        "violent_count_roll3","all_count_roll1","any_event_lag1","log_event_density_lag1","months_since_last_event"]
need = ["grid_id","year_month","event_sequence","violence_next_month","count_all_events"] + BASE
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=[c for c in need if c])
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id","year_month","country"])
panel = panel.merge(cm, on=["grid_id","year_month"], how="left")
panel["region"] = panel["country"].map(lambda c: REGION_OF.get(c, "other"))
panel = panel[(panel.region != "other") & panel["violence_next_month"].notna()].reset_index(drop=True)
panel[BASE] = panel[BASE].fillna(0.0)
vol = np.nan_to_num(panel["count_all_events"].to_numpy().astype(float)) if "count_all_events" in panel.columns else np.nan_to_num(panel["all_count_roll1"].to_numpy().astype(float))

def ap(y, s): return average_precision_score(y, s) if len(np.unique(y)) > 1 else np.nan
def xgb_ap(Xtr, ytr, Xte, yte):
    if ytr.sum() < 3: return np.nan
    c = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="aucpr", n_jobs=8, random_state=SEED)
    c.fit(Xtr, ytr); return ap(yte, c.predict_proba(Xte)[:, 1])

rows = []

if MODE == "interp":
    # stratified sample per region
    idx = []
    for rg in REGIONS:
        ri = panel.index[panel.region == rg].to_numpy()
        idx.append(rng.choice(ri, size=min(30000, len(ri)), replace=False))
    idx = np.concatenate(idx); d = panel.loc[idx].reset_index(drop=True)
    y = d["violence_next_month"].to_numpy().astype(int)
    lag1 = d["violence_lag1"].to_numpy(); B = d[BASE].to_numpy()
    reg_y = d["region"].to_numpy()
    enc = build_encoder(f"{R}/data_model/repr_model_L_contrastive_e_s123.pt")
    Zr = extract(enc, d["event_sequence"].values, d["grid_id"].values)
    Zshuf = extract(enc, d["event_sequence"].values, d["grid_id"].values, shuffle=True)
    Zrand = extract(build_encoder(None), d["event_sequence"].values, d["grid_id"].values)
    # #1 region probe
    for name, X in [("emb", Zr), ("base", B)]:
        acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                              X, reg_y, cv=3, scoring="accuracy").mean()
        rows.append(dict(analysis="region_probe", variant=name, metric="cv_accuracy", value=round(float(acc), 4)))
    rows.append(dict(analysis="region_probe", variant="chance", metric="cv_accuracy", value=round(1/6, 4)))
    # 3-fold temporal-agnostic split for #3/#4
    ym = d["year_month"].to_numpy(); tr = ym <= TRAIN_END; te = ym >= TEST_START
    onset = lag1 == 0
    vd = vol[idx] if len(vol) == len(panel) else np.nan_to_num(d["all_count_roll1"].to_numpy().astype(float))
    # #3 volume confound (onset subset)
    trm, tem = tr & onset, te & onset
    if tem.sum() > 30 and y[tem].sum() > 5:
        rows.append(dict(analysis="volume", variant="volume_only", metric="onset_ap",
                         value=round(float(xgb_ap(vd[trm, None], y[trm], vd[tem, None], y[tem])), 4)))
        rows.append(dict(analysis="volume", variant="base", metric="onset_ap",
                         value=round(float(xgb_ap(B[trm], y[trm], B[tem], y[tem])), 4)))
        rows.append(dict(analysis="volume", variant="emb", metric="onset_ap",
                         value=round(float(xgb_ap(Zr[trm], y[trm], Zr[tem], y[tem])), 4)))
        rows.append(dict(analysis="volume", variant="emb_base", metric="onset_ap",
                         value=round(float(xgb_ap(np.c_[Zr, B][trm], y[trm], np.c_[Zr, B][tem], y[tem])), 4)))
    # #4 null encoders (incidence)
    for name, Z in [("emb_real", Zr), ("emb_shuffled", Zshuf), ("emb_random", Zrand), ("base", B)]:
        rows.append(dict(analysis="null_encoder", variant=name, metric="incidence_ap",
                         value=round(float(xgb_ap(Z[tr], y[tr], Z[te], y[te])), 4)))
    out = pd.DataFrame(rows); out.to_csv(f"{R}/outputs/interp_defense_summary.csv", index=False)
    print(out.to_string(index=False), flush=True)
    print("[done] interp", flush=True)

elif MODE == "ci":
    NB = 2000
    for rg in REGIONS:
        ck = f"{R}/data_model/repr_model_L_contrastive_e_loro_{rg}_s123.pt"
        if not os.path.exists(ck):
            print(f"[skip] no ckpt {rg}", flush=True); continue
        d = panel[panel.region == rg].sort_values("year_month").reset_index(drop=True)
        if len(d) > 250000:
            d = d.sample(n=250000, random_state=SEED).sort_values("year_month").reset_index(drop=True)
        ym = d["year_month"].to_numpy(); tr = ym <= TRAIN_END; te = ym >= TEST_START
        if tr.sum() < 100 or te.sum() < 50: 
            print(f"[skip] {rg} thin split", flush=True); continue
        y = d["violence_next_month"].to_numpy().astype(int)
        lag1 = np.nan_to_num(d["violence_lag1"].to_numpy()); B = d[BASE].to_numpy()
        enc = build_encoder(ck); Z = extract(enc, d["event_sequence"].values, d["grid_id"].values)
        Zrnd = extract(build_encoder(None), d["event_sequence"].values, d["grid_id"].values)
        EB = np.c_[Z, B]
        # per-cell test predictions
        def preds(X): 
            c = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="aucpr", n_jobs=8, random_state=SEED)
            c.fit(X[tr], y[tr]); return c.predict_proba(X[te])[:, 1]
        yte = y[te]; p_persist = lag1[te]; p_base = preds(B); p_emb = preds(Z); p_eb = preds(EB)
        p_rnd = preds(Zrnd)
        # bootstrap over test cells
        n = len(yte); boot = {k: [] for k in ["emb","base","d_emb_persist","d_emb_base","d_eb_base","rnd","d_rnd_persist","d_emb_rnd"]}
        for _ in range(NB):
            bi = rng.integers(0, n, n)
            if yte[bi].sum() < 2: continue
            ae = ap(yte[bi], p_emb[bi]); ab = ap(yte[bi], p_base[bi]); aeb = ap(yte[bi], p_eb[bi]); apr = ap(yte[bi], p_persist[bi])
            boot["emb"].append(ae); boot["base"].append(ab)
            boot["d_emb_persist"].append(ae - apr); boot["d_emb_base"].append(ae - ab); boot["d_eb_base"].append(aeb - ab)
            arn = ap(yte[bi], p_rnd[bi]); boot["rnd"].append(arn); boot["d_rnd_persist"].append(arn - apr); boot["d_emb_rnd"].append(ae - arn)
        def ci(v): a = np.array(v); return round(float(np.mean(a)), 4), round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)
        for k in boot:
            m, lo, hi = ci(boot[k])
            rows.append(dict(region=rg, n_test=int(n), quantity=k, mean=m, lo=lo, hi=hi))
        print(f"[{rg}] n_test={n} d_emb_persist={ci(boot['d_emb_persist'])} d_emb_base={ci(boot['d_emb_base'])}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(f"{R}/outputs/loro_region_ci.csv", index=False)
    print(out.to_string(index=False), flush=True)
    print("[done] ci", flush=True)

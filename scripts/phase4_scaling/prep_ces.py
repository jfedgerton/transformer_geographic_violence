"""prep_ces.py -- builds the CES appendix panel (respondent-level tokens) from the CES cumulative file.

Source: data_raw/ces/cumulative_2006-2025.feather (Kuriwaki, CES cumulative common content, doi:10.7910/DVN/II2DB6
v12.0, CC0). One row per respondent.
Outcome  y = validated general-election turnout (vv_turnout_gvm): 1 = "Voted", 0 = "No Record of Voting".
         Defined only in even years 2006-2024 (validation is not done in odd years); NaN otherwise.
Unit     9 Census divisions (state -> division, hard-coded below). The division is the LORO unit.
Tokens   one token per answered PRE-ELECTION item: (token_id for item=value, item_id). Missing answers are absent
         tokens, so years that did not ask an item simply lack that token. A year-type token (presidential /
         midterm / odd-year) is added. No post-election items, no vote-validation variables, no geography.
Naive    intent_turnout_self mapped to an ordinal score (the respondent's own pre-election statement): the CES
         analogue of persistence in the conflict tables.
Outputs  data_intermediate/ces_tokens.parquet, data_intermediate/ces_vocab.json
"""
import json, os
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SRC = f"{R}/data_raw/ces/cumulative_2006-2025.feather"
OUT = f"{R}/data_intermediate/ces_tokens.parquet"; VOC = f"{R}/data_intermediate/ces_vocab.json"
for f in (OUT, VOC): assert not os.path.exists(f), f"refusing to overwrite {f}"

ITEMS = ["pid7", "ideo5", "gender", "age", "race", "hispanic", "educ", "marstat", "faminc", "union", "union_hh",
         "employ", "no_healthins", "investor", "has_child", "ownhome", "religion", "relig_imp", "relig_bornagain",
         "relig_church", "citizen", "newsint", "economy_retro", "approval_pres", "approval_rep", "approval_sen1",
         "approval_sen2", "approval_gov", "reg_self", "intent_turnout_self", "intent_pres_party"]
EXCLUDED = ["voted_*", "*_post", "tookpost", "vv_*", "weights", "state/district/zip/county", "intent_rep_party (35 levels)"]

DIVISION = {   # US Census Bureau divisions
    "New England": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
    "Middle Atlantic": ["New Jersey", "New York", "Pennsylvania"],
    "East North Central": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"],
    "West North Central": ["Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"],
    "South Atlantic": ["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", "North Carolina",
                       "South Carolina", "Virginia", "West Virginia"],
    "East South Central": ["Alabama", "Kentucky", "Mississippi", "Tennessee"],
    "West South Central": ["Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "Mountain": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming"],
    "Pacific": ["Alaska", "California", "Hawaii", "Oregon", "Washington"],
}
DIV_OF = {st: dv for dv, sts in DIVISION.items() for st in sts}
assert len(DIV_OF) == 51

d = pd.read_feather(SRC, columns=["year", "case_id", "state", "vv_turnout_gvm"] + ITEMS)
print(f"[load] {d.shape}", flush=True)
unknown = sorted(set(d["state"].dropna().astype(str)) - set(DIV_OF))
assert not unknown, f"states not in division map: {unknown}"
d = d[d["state"].notna()].copy()
d["division"] = d["state"].astype(str).map(DIV_OF)
assert d["division"].notna().all()

vv = d["vv_turnout_gvm"].astype("object")
d["y"] = np.where(vv == "Voted", 1.0, np.where(vv == "No Record of Voting", 0.0, np.nan))
even = d["year"] % 2 == 0
assert d.loc[even & d["year"].between(2006, 2024), "y"].notna().mean() > 0.99, "validated turnout missing in even years"
d.loc[~even, "y"] = np.nan

# age -> 5-year bins (17-19, 20-24, ..., 85+); all other items are already categorical codes or labels
age = pd.to_numeric(d["age"], errors="coerce")
d["age"] = pd.cut(age, [0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 200]).astype(str).where(age.notna())

# string-ify every item (codes and labels alike); NaN stays NaN
for c in ITEMS:
    v = d[c].astype("object")
    d[c] = v.where(v.isna(), v.astype(str))

d["year_type"] = np.where(d["year"] % 4 == 0, "pres", np.where(even, "mid", "odd"))
TOKEN_ITEMS = ITEMS + ["year_type"]
item_id = {c: i + 1 for i, c in enumerate(TOKEN_ITEMS)}          # 0 = padding
tok_id = {}
for c in TOKEN_ITEMS:
    for v in sorted(d[c].dropna().unique()):
        tok_id[f"{c}={v}"] = len(tok_id) + 1                    # 0 = padding
print(f"[vocab] {len(item_id)} items, {len(tok_id)} item=value tokens", flush=True)

vals = d[TOKEN_ITEMS].to_numpy(dtype=object)
seqs, lens = [], []
for row in vals:
    toks = [[tok_id[f"{c}={v}"], item_id[c]] for c, v in zip(TOKEN_ITEMS, row) if v is not None and v == v]
    seqs.append(json.dumps(toks)); lens.append(len(toks))
d["event_sequence"] = seqs
nt = pd.Series(lens)
print(f"[tokens] per respondent: median {nt.median():.0f}, min {nt.min()}, max {nt.max()}", flush=True)
assert nt.min() >= 2

NAIVE = {"No": 0.0, "Undecided": 0.25, "Probably": 0.5, "Yes, definitely": 0.9,
         "I plan to vote before Election Day": 0.9, "I already voted (early or absentee)": 1.0}
seen = set(d["intent_turnout_self"].dropna()) - set(NAIVE)
assert not seen, f"unmapped intent_turnout_self values: {seen}"
d["naive"] = d["intent_turnout_self"].map(NAIVE).astype(float)   # NaN when not asked

keep = ["case_id", "year", "state", "division", "y", "naive", "event_sequence"] + ITEMS
d[keep].reset_index(drop=True).to_parquet(OUT, index=False)
json.dump(dict(item_id=item_id, tok_id=tok_id, items=ITEMS, excluded=EXCLUDED, division=DIVISION,
               naive_map=NAIVE, source="doi:10.7910/DVN/II2DB6 v12.0"), open(VOC, "w"), indent=1)
summ = d[even & d["year"].between(2006, 2024)].groupby(["year"])["y"].agg(["size", "mean"]).round(3)
print(summ.to_string(), flush=True)
print(d[d["year"] == 2024].groupby("division")["y"].agg(["size", "mean"]).round(3).to_string(), flush=True)
print(f"[done] {OUT} ({len(d):,} rows)", flush=True)

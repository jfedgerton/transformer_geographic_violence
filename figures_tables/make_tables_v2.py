# make_tables_v2.py -- manuscript tables (Oct 14, 2026): 3 encoder seeds, vocab remap, persistence in month t.
# Inputs (this folder): ladder_with_persistence_t.csv (persistence_month_t.py joined to eval files),
#   random_control_summary.csv (cmp_random_v2.py), grid/metrics_nested_{region}_s{seed}.csv (share_grid_conflict.py)
# Outputs: tables/*.tex
import os, glob
import numpy as np, pandas as pd
os.makedirs("tables", exist_ok=True)
REG = {"africa": "Africa", "americas": "Americas", "asia_pacific": "Asia-Pacific", "europe": "Europe",
       "mena": "MENA", "south_asia": "South Asia"}
ORDER = list(REG)
f3 = lambda x: f"${x:+.3f}$"
p3 = lambda x: f"{x:.3f}"
cnt = lambda x: f"{int((x > 0).sum())}/{len(x)}"

L = pd.read_csv("ladder_with_persistence_t.csv")
assert len(L) == 36 and set(L["mode"]) == {"Temporal", "Pooled"} and not L.isna().any().any()
L["gain"] = L.emb_base - L.base
L["vs_pers"] = L.emb_base - L.persistence_t

def ladder_table(mode, label, caption, note):
    d = L[L["mode"] == mode]; assert len(d) == 18
    rows = []
    for g in ORDER + ["ALL"]:
        x = d if g == "ALL" else d[d.region == g]
        name = "All regions" if g == "ALL" else REG[g]
        rows.append(f"{name} & {p3(x.persistence_t.mean())} & {p3(x.base.mean())} & {p3(x.emb_base.mean())} & "
                    f"{f3(x.vs_pers.mean())} & {f3(x.gain.mean())} & {cnt(x.gain)} \\\\")
        if g == ORDER[-1]: rows.append("\\hline")
    return f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcccccc}}
\\hline
 & \\multicolumn{{3}}{{c}}{{PR-AUC}} & Base + emb. & Base + emb. & Runs \\\\
Held-out region & Persistence & Base & Base + emb. & $-$ persistence & $-$ base & gain $>0$ \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\begin{{minipage}}{{0.98\\textwidth}}\\footnotesize\\vspace{{0.5ex}}
{note}
\\end{{minipage}}
\\end{{table}}
"""
old_t = L[L["mode"] == "Temporal"].persistence_tm1_reported.mean()
old_p = L[L["mode"] == "Pooled"].persistence_tm1_reported.mean()
NOTE = ("Notes. Means over three encoder seeds (123, 456, 789). Target is any violent event in the 0.5$^\\circ$ cell in month $t+1$; "
        "all models use information through month $t$. Persistence is the presence of violence in month $t$. "
        "Base is gradient boosting on hand-built and PRIO-GRID features; Base + emb. adds the frozen encoder embedding. "
        "Runs gain $>0$ counts region-seed runs in which Base + emb. beats Base. ")
open("tables/table1_loro_general.tex", "w").write(ladder_table("Temporal", "tab:loro_general",
    "Held-out regions, temporal test: persistence, base model and base model with the embedding (PR-AUC)",
    NOTE + f"Temporal test trains through December 2020 and tests from January 2022. Persistence measured in month $t-1$ averages {old_t:.3f}."))
open("tables/tab_nested_embbase_pooled.tex", "w").write(ladder_table("Pooled", "tab:nested_embbase",
    "Held-out regions, pooled five-fold test: persistence, base model and base model with the embedding (PR-AUC)",
    NOTE + f"Pooled scores average five folds drawn over cell-months. Persistence measured in month $t-1$ averages {old_p:.3f}."))

# ---- random-encoder control (incidence) ----
R = pd.read_csv("random_control_summary.csv")
def rand_table(mode, label, caption, note):
    d = R[(R.target == "incidence") & (R.m == mode)]; assert len(d) == 18, len(d)
    lm = L[L["mode"] == mode.capitalize()].set_index(["region", "seed"])
    chk = d.set_index(["region", "seed"]).join(lm[["gain"]])
    assert (chk.pre_gain - chk.gain).abs().max() < 1e-9, "random summary and ladder disagree on the pretrained gain"
    rows = []
    for g in ORDER + ["ALL"]:
        x = d if g == "ALL" else d[d.region == g]
        name = "All regions" if g == "ALL" else REG[g]
        rows.append(f"{name} & {p3(x.base.mean())} & {f3(x.pre_gain.mean())} & {f3(x.rand_gain.mean())} & "
                    f"{f3(x.transfer.mean())} & {cnt(x.transfer)} \\\\")
        if g == ORDER[-1]: rows.append("\\hline")
    return f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{5pt}}
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccccc}}
\\hline
 & PR-AUC & \\multicolumn{{2}}{{c}}{{Gain over base}} & Pretrained & Runs \\\\
Held-out region & Base & Pretrained encoder & Random encoder & $-$ random & difference $>0$ \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\begin{{minipage}}{{0.98\\textwidth}}\\footnotesize\\vspace{{0.5ex}}
{note}
\\end{{minipage}}
\\end{{table}}
"""
RN = ("Notes. Means over three encoder seeds. Gain over base is the PR-AUC of base + embedding minus base. "
      "The random encoder has the same architecture and random weights; the base model and test rows are identical across the two arms. ")
open("tables/tab_random_control_temporal.tex", "w").write(rand_table("temporal", "tab:random_temporal",
    "Political violence, temporal test: gain from a pretrained and a random encoder", RN + "Temporal test as in Table~\\ref{tab:loro_general}."))
open("tables/tab_random_control_pooled.tex", "w").write(rand_table("pooled", "tab:random_pooled",
    "Political violence, pooled five-fold test: gain from a pretrained and a random encoder", RN + "Pooled scores average five folds."))

# ---- share grid, nested panel (fine-tune panel added when the fine-tune runs finish) ----
G = pd.concat([pd.read_csv(f) for f in glob.glob("grid/metrics_nested_*_s*.csv")])
assert set(G.region) == set(ORDER) and set(G.enc_seed) == {123, 456, 789}
n = G[G.model.isin(["base", "emb_base"])].pivot_table(index=["region", "enc_seed", "share", "draw"], columns="model", values="pr").reset_index()
assert not n[["base", "emb_base"]].isna().any().any()
n["gain"] = n.emb_base - n.base
per_rs = n.groupby(["region", "enc_seed", "share"]).gain.mean().reset_index()
assert (per_rs.groupby(["region", "share"]).size() == 3).all()
per_r = per_rs.groupby(["region", "share"]).gain.mean().unstack("share")
SH = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1]
assert list(per_r.columns) == SH, list(per_r.columns)
ntr = G[(G.model == "base")].groupby(["region", "share"]).pos_train.mean().unstack("share")
rows = [f"{REG[g]} & " + " & ".join(f3(per_r.loc[g, s]) for s in SH) + " \\\\" for g in ORDER]
rows.append("\\hline")
rows.append("Mean & " + " & ".join(f3(per_r[s].mean()) for s in SH) + " \\\\")
rows.append("Regions $>0$ & " + " & ".join(f"{int((per_r[s] > 0).sum())}/6" for s in SH) + " \\\\")
eu_l = [f"{int(round(ntr.loc['europe', s]))}" for s in SH]; eu = ", ".join(eu_l[:-1]) + " and " + eu_l[-1]
t2 = f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Held-out regions with limited local data, temporal test: gain from adding the embedding to the base model (PR-AUC)}}
\\label{{tab:pretrained_1pct}}
\\begin{{tabular}}{{lccccccc}}
\\hline
 & \\multicolumn{{7}}{{c}}{{Share of the held-out region's training data}} \\\\
Held-out region & 0.25\\% & 0.5\\% & 0.75\\% & 1\\% & 2.5\\% & 5\\% & 10\\% \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\begin{{minipage}}{{0.98\\textwidth}}\\footnotesize\\vspace{{0.5ex}}
Notes. Base + embedding minus base, both trained on the stated share of the held-out region's training months and scored on the full temporal test.
Means over three encoder seeds and five random draws of the training share. The encoder itself is never retrained.
Europe has very few violent training cell-months at small shares ({eu} across the seven shares).
Fine-tuned pretrained versus scratch encoders: Appendix Table XX (runs in progress).
\\end{{minipage}}
\\end{{table}}
"""
open("tables/table2_pretrained_1pct.tex", "w").write(t2)
print(per_r.round(4)); print("Europe positive training cell-months by share:", ntr.loc["europe"].round(0).tolist())
for f in sorted(glob.glob("tables/*.tex")): print(f)

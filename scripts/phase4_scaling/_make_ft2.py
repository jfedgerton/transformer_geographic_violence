src = open("scripts/phase4_scaling/transfer_eval_region_ft.py").read().split("\n")
out = []; hits = dict(out=0, sel=0, arms=0, rows=0, perm=0, chk=0)
for ln in src:
    ind = ln[:len(ln) - len(ln.lstrip())]
    if ln.startswith("OUT = "):
        out += ['SUFFIX = os.environ.get("EVAL_SUFFIX", "")',
                'assert SUFFIX, "EVAL_SUFFIX required so the 2026-10-06 CSVs are never overwritten"',
                'OUT = f"{R}/outputs/ft_region_{TAG}{SUFFIX}.csv"',
                'ARMS = os.environ.get("EVAL_ARMS", "ft_pretrained,ft_scratch").split(",")',
                'assert set(ARMS) <= {"ft_pretrained", "ft_scratch"}, ARMS',
                'SCR_LR = float(os.environ.get("EVAL_SCR_LR", "5e-4"))',
                'PERM_TIME = int(os.environ.get("EVAL_PERM_TIME", "0"))',
                'print(f"[cfg2] arms={ARMS} scr_lr={SCR_LR} perm_time={PERM_TIME} out={OUT}", flush=True)']
        hits["out"] += 1; continue
    if ln.startswith("SEL = "):
        out.append(ln); out.append('TIME_COL = FIELDS.index("month_offset")'); hits["sel"] += 1; continue
    if '("ft_pretrained", CKPT, 1e-4)' in ln:
        out.append(ind + 'for mdl, ck, lr in [a for a in [("ft_pretrained", CKPT, 1e-4), ("ft_scratch", None, SCR_LR)] if a[0] in ARMS]:')
        hits["arms"] += 1; continue
    if "rows.append(dict(base, model=mdl" in ln:
        new = ln.replace("roc=roc, pr=pr))", "roc=roc, pr=pr, lr=lr, perm_time=PERM_TIME))")
        assert new != ln; out.append(new); hits["rows"] += 1; continue
    if ln.strip().startswith("st = torch.from_numpy(prio_lookup"):
        out += [ind + "if PERM_TIME and len(seq) > 1:",
                ind + "    n = len(seq); perm = torch.from_numpy(np.random.RandomState((PERM_TIME * 1000003 + i) % (2**31)).permutation(n))",
                ind + "    t[:n, TIME_COL] = t[perm, TIME_COL]"]
        hits["perm"] += 1
    out.append(ln)
    if ln.startswith('seqs = d["event_sequence"]'):
        out += ['_chk = [json.loads(s) for s in seqs[np.random.RandomState(0).choice(len(seqs), size=min(5000, len(seqs)), replace=False)] if isinstance(s, str)]',
                '_var = float(np.mean([len({ev[SEL[TIME_COL]] for ev in q}) > 1 for q in _chk if len(q) > 1] or [0.0]))',
                'print(f"[permcheck] share of multi-event seqs with >1 distinct month_offset: {_var:.3f}", flush=True)',
                'assert not PERM_TIME or _var > 0.05, "month_offset barely varies within sequences; time permutation would be a no-op"']
        hits["chk"] += 1
assert all(v == 1 for v in hits.values()), hits
open("scripts/phase4_scaling/transfer_eval_region_ft2.py", "w").write("\n".join(out))
print("OK", hits)

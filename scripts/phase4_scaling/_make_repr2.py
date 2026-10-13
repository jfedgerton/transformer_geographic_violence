src = open("scripts/phase4_scaling/train_repr_obj.py").read().split("\n")
out = []; hits = dict(cfg=0, mask=0)
for ln in src:
    ind = ln[:len(ln) - len(ln.lstrip())]
    if ln.strip() == "panel = panel[~drop].reset_index(drop=True)":
        out += [ind + "if ONLY:",
                ind + '    assert "country" in panel.columns, "country column not loaded"',
                ind + '    drop |= ~panel["country"].isin(ONLY_COUNTRIES).to_numpy()']
        hits["mask"] += 1
    out.append(ln)
    if ln.startswith("LORO_COUNTRIES = []"):
        out += ['ONLY = E("REPR_ONLY","")  # single SOURCE region: pretrain on this region alone (transfer matrix)',
                'ONLY_COUNTRIES = []',
                'if ONLY:',
                '    import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))',
                '    from regions import REGION_OF, REGIONS',
                '    assert ONLY in REGIONS and not LORO, f"bad REPR_ONLY {ONLY} (REPR_LORO must be empty)"',
                '    ONLY_COUNTRIES = [c for c,r in REGION_OF.items() if r == ONLY]',
                '    print(f"[only] single source region {ONLY}: {len(ONLY_COUNTRIES)} countries", flush=True)']
        hits["cfg"] += 1
assert all(v == 1 for v in hits.values()), hits
open("scripts/phase4_scaling/train_repr_obj2.py", "w").write("\n".join(out))
print("OK", hits)

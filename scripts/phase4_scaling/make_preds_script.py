# make_preds_script.py -- builds transfer_eval_region_preds.py from transfer_eval_region.py.
# Only three changes, each asserted to match exactly once:
#   1. output CSV renamed (never overwrites outputs/eval_region_{TAG}.csv)
#   2. after the Mode B incidence scores, save cell-level forecasts to outputs/preds_region_{TAG}.npz
#   3. Mode A (pooled 5-fold) skipped -- Table 1 uses Mode B only
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/scripts/phase4_scaling/"
src = open(R + "transfer_eval_region.py").read()

def rep(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for: {old[:60]!r} (found {s.count(old)})"
    return s.replace(old, new)

src = rep(src, 'OUT = f"{R}/outputs/eval_region_{TAG}.csv"',
               'OUT = f"{R}/outputs/eval_region_preds_{TAG}.csv"   # preds run: never overwrite the original')
old2 = '            add("rare_event_f1", "emb_base", best_f1(y_inc[tem], peb), tem.sum())\n'
new2 = old2 + ('            if mode == "B_temporal":\n'
               '                np.savez_compressed(f"{R}/outputs/preds_region_{TAG}.npz",\n'
               '                    grid_id=d["grid_id"].to_numpy()[tem], year_month=d["year_month"].astype(str).to_numpy()[tem],\n'
               '                    y=y_inc[tem], persistence=lag1[tem], base=pb, emb=pe, emb_base=peb)\n'
               '                print(f"[preds] saved {tem.sum()} rows", flush=True)\n')
src = rep(src, old2, new2)
src = rep(src, '        score_block(uname.replace(" ", "_"), f"A_pooled_f{k}", d, trm, tem, Z)',
               '        continue   # preds run: Mode A skipped')
open(R + "transfer_eval_region_preds.py", "w").write(src)
print("wrote transfer_eval_region_preds.py")

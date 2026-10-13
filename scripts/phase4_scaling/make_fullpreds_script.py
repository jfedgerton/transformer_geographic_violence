# make_fullpreds_script.py -- builds transfer_eval_region_fullpreds.py from transfer_eval_region.py.
# ONE change (asserted to match exactly once): after Mode B incidence scores, save cell-level forecasts
# to outputs/preds_region_{TAG}.npz. Both modes still run; OUT name unchanged (used for new seed tags only).
P = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/scripts/phase4_scaling/"
src = open(P + "transfer_eval_region.py").read()
old2 = '            add("rare_event_f1", "emb_base", best_f1(y_inc[tem], peb), tem.sum())\n'
assert src.count(old2) == 1, f"anchor found {src.count(old2)} times"
new2 = old2 + ('            if mode == "B_temporal":\n'
               '                np.savez_compressed(f"{R}/outputs/preds_region_{TAG}.npz",\n'
               '                    grid_id=d["grid_id"].to_numpy()[tem], year_month=d["year_month"].astype(str).to_numpy()[tem],\n'
               '                    y=y_inc[tem], persistence=lag1[tem], base=pb, emb=pe, emb_base=peb)\n'
               '                print(f"[preds] saved {tem.sum()} rows", flush=True)\n')
out = src.replace(old2, new2)
assert out.count("np.savez_compressed") == src.count("np.savez_compressed") + 1
open(P + "transfer_eval_region_fullpreds.py", "w").write(out)
print("wrote transfer_eval_region_fullpreds.py")

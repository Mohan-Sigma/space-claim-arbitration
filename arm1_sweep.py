import sys, json, time, numpy as np
import arm1_waterfall as W, evaluators as ev
part = int(sys.argv[1])
runtimes = np.linspace(42, 62, 10)
v_batts = [rt * 0.038 / 600 * 1e6 for rt in runtimes]       # invert runtime model
v_ants = [900, 1200, 1500, 1800, 2100]
jobs = [(vb, va) for vb in v_batts for va in v_ants][part::2]
out = []
for vb, va in jobs:
    W.EVALS["acoustic"] = 0; t0 = time.time()
    r = W.run(vb, va)
    r.update(acoustic_evals=W.EVALS["acoustic"], system_evals=r["loops"] + 1, wall_s=time.time() - t0)
    out.append(r)
    print(f"spec {vb/0.038*600/1e6:.1f} h / ant {va}: ok={r['ok']} loops={r['loops']} evals={W.EVALS['acoustic']}"
          + (f" -> runtime {r['f'][0]:.1f} siren {r['f'][1]:.2f} voice {r['f'][2]:.1f} L5 {r['f'][3]:.3f}" if r["ok"] else ""), flush=True)
json.dump(out, open(f"arm1_part{part}.json", "w"))

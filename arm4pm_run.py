import sys, json, time
import arm4v2_negotiation as N, arm4pm_negotiation as M
from arm4_run import DIRECTIONS
import os
KEY = os.environ.get("ANTHROPIC_KEY_FILE", "anthropic_key.txt")   # path to a file holding the API key (not included)
be = N.CachedAnthropicBackend("claude-sonnet-5-5", KEY, max_tokens=4000)
for nm in sys.argv[1].split(","):
    t0 = time.time()
    r = M.negotiate(be, priority_note=DIRECTIONS[nm], log=lambda m: print(f"[{nm}]{m}", flush=True), ckpt=f"arm4pm_{nm}.ckpt.json")
    r.update(direction=nm, wall_s=time.time() - t0, model="claude-sonnet-5-5", sampling="model default")
    json.dump(r, open(f"arm4pm_{nm}.json", "w"), indent=1)
    print(f"[{nm}] DONE feasible={r['feasible']} confirmed={r['confirmed']} rounds={r['rounds']} cost~${r['est_cost_usd']} "
          f"violated={r['violated']} f={[round(v,3) for v in r['f']]} {r['wall_s']:.0f}s", flush=True)

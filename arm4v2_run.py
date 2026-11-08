import sys, json, time
import arm4v2_negotiation as N
from arm4_run import DIRECTIONS
import os
KEY = os.environ.get("ANTHROPIC_KEY_FILE", "anthropic_key.txt")   # path to a file holding the API key (not included)
be = N.CachedAnthropicBackend("claude-sonnet-5-5", KEY)
for nm in sys.argv[1].split(","):
    t0 = time.time()
    r = N.negotiate(be, priority_note=DIRECTIONS[nm], log=lambda m: print(f"[{nm}]{m}", flush=True))
    r.update(direction=nm, wall_s=time.time() - t0, model="claude-sonnet-5-5", sampling="model default")
    json.dump(r, open(f"arm4v2_{nm}.json", "w"), indent=1)
    print(f"[{nm}] DONE feasible={r['feasible']} confirmed={r['confirmed']} rounds={r['rounds']} "
          f"cost~${r['est_cost_usd']} usage={r['usage']} violated={r['violated']} f={[round(v,3) for v in r['f']]} {r['wall_s']:.0f}s", flush=True)

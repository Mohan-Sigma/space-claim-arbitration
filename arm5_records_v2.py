"""Regenerate Arm 5 decision records with a 4000-token output cap and the program-direction text.
Designs, specs and solver outputs are reused unchanged from the saved runs.
Usage: python3 arm5_records_v2.py <model> <in_json> <out_json>"""
import sys, json, time
import arm5_llm as L
from arm4_run import DIRECTIONS
model, fin, fout = sys.argv[1:4]
L.MODEL = model
llm = L.LLM()
stops = []
_orig = llm.c.messages.create
def create(**kw):
    kw["max_tokens"] = 4000
    r = _orig(**kw); stops.append(r.stop_reason); return r
llm.c.messages.create = create
d = json.load(open(fin)); t0 = time.time(); out = []
for r in d["directions"]:
    note = DIRECTIONS[r["direction"]]
    trace, data = L.rationale(llm, r["spec"], r, note)
    g = L.grounding(trace, data)
    out.append(dict(direction=r["direction"], trace=trace, grounding=g, stop_reason=stops[-1], words=len(trace.split())))
    print(f"[{model} {fin}] {r['direction']:<14} stop={stops[-1]} words={len(trace.split())} grounded={g['grounded']}/{g['numbers']}", flush=True)
json.dump(dict(model=model, source=fin, records=out, usage=llm.usage, wall_s=time.time() - t0), open(fout, "w"), indent=1)
print(f"[{model} {fin}] DONE usage={llm.usage}", flush=True)

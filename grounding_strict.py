"""Stricter grounding re-score: ignores bare integers 0-10 (list labels, counts) and section numbering."""
import json, glob, re, sys, numpy as np
import arm5_llm as L, arm2v6_problem as P
from arm4_run import DIRECTIONS
E = L.E
def build_data(spec, result, note):
    claims = {n: float(P.TARGET[P.NAMES.index(n)]) for n in P.NAMES}
    alloc = {n: round(float(v), 3) for n, v in zip(P.NAMES, result["x"])}
    obj = dict(zip(E.OBJ, [round(v, 3) for v in result["f"]]))
    binding = [n for n, g in zip(P.CONSTRAINTS, result["g"]) if g > -0.02]
    prices = {k: round(v, 2) for k, v in sorted(result["prices"].items(), key=lambda kv: -abs(kv[1]))}
    info = {k: round(v, 3) for k, v in result["info"].items() if k in ("p_siren", "q_ss", "f_res", "vfc", "vb", "mass", "pmarg")}
    return dict(program_direction=note or "none", formulation={k: spec[k] for k in ("scalarisation", "objective_floors", "weights")},
                objectives_achieved=obj, allocation=alloc, each_domains_original_claim=claims,
                binding_or_near_binding_constraints=binding, acoustic_details=info, coupling_prices=prices)
def strip(text):
    t = re.sub(r"(?m)^\s*(#+\s*)?(\*\*)?\(?\d\)?[.)]?\s", " ", text)        # section labels "1." "(2)" "**3."
    t = re.sub(r"[A-Za-z_]+\d+[A-Za-z_]*", " ", t)                              # identifiers like C1_volume, O6
    return t
pairs = {"sonnet": "arm5e_r{}.json", "haiku": "arm5x_claude-haiku-5-5_r{}.json", "opus": "arm5x_claude-opus-5-5_r{}.json"}
out = {}
for m, pat in pairs.items():
    G = N = 0; bad = []
    for k in (1, 2, 3):
        src = {r["direction"]: r for r in json.load(open(pat.format(k)))["directions"]}
        for rec in json.load(open(f"arm5rec_{m}_r{k}.json"))["records"]:
            r = src[rec["direction"]]; data = build_data(r["spec"], r, DIRECTIONS[rec["direction"]])
            t = strip(rec["trace"])
            nums = [s for s in re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*", t)]
            for s in nums:
                v = float(s.replace(",", "").rstrip("."))
                if "." not in s.rstrip(".") and abs(v) <= 10: continue
                g = L.grounding(s, data)
                N += g["numbers"]; G += g["grounded"]
                if g["numbers"] and not g["grounded"]: bad.append(s)
    out[m] = (G, N); print(m, f"strict {G}/{N} = {100*G/N:.1f}%", "examples of ungrounded:", bad[:12])
json.dump(out, open("grounding_strict.json", "w"))

"""
Arm 5 - LLM-formulated ATC.

Layer 1 (formulation): the LLM reads the SAME requirements text the Arm 4 agents received, plus a
catalog of what the arbiter can control and evaluate, and writes the formulation spec that drives
arm5_engine (normalisation, scalarisation, aspiration, objective floors, coordination, weights).
Layer 2 (search): arm5_engine - ATC with augmented-Lagrangian coordination. No LLM in the loop.
Layer 3 (rationale): the LLM receives the solved design, each domain's allocation vs its own claim,
active constraints and the coupling prices (multipliers), and writes a decision trace. Every number
in the trace is checked against the data it was given.
"""
import json, re, time
import numpy as np
import arm4_negotiation as A4
import arm5_engine as E

MODEL = "claude-sonnet-5-5"
import os
KEY = os.environ.get("ANTHROPIC_KEY_FILE", "anthropic_key.txt")   # path to a file holding the API key (not included)

REQUIREMENTS = A4.BRIEF + "\n\nDOMAIN REQUIREMENTS:\n" + "\n".join(f"- {a}: {d['req']}" for a, d in A4.AGENTS.items())

CATALOG = """ARBITER FORMULATION CATALOG (Analytical Target Cascading).
Fixed by the domains (you do not change these): the arbiter sets 14 targets - every domain's allocation
(battery volume/footprint, board area, structure ribs and wall thickness, antenna keepout, acoustic volume,
port area, haptics, thermal spreader, NFC loop and ferrite) plus two acoustic RESPONSES (siren tone power,
voice SII). Each domain holds its own constraints and matches the targets as closely as it can; an
augmented-Lagrangian loop makes arbiter and domains consistent. Interior volume, caseback, perimeter and
port pools are always enforced at the arbiter.

System objectives the arbiter can evaluate (all maximised):
  runtime : battery runtime, hours (physically ~41 to ~62 h in this envelope)
  siren   : emergency-beacon endurance in hours at 86 dB @ 1 m, 50% duty (depends on battery energy AND siren power)
  voice   : voice intelligibility, 100 x ANSI SII (roughly 70 to 90 for feasible designs)
  l5      : GNSS L5 antenna efficiency, 0..1 (roughly 0.05 to 0.30 for feasible designs)

YOU DECIDE (return as JSON):
  "normalisation": {obj: [worst_acceptable, best_plausible]} for all four objectives. Scales each objective to
                   0..1 for the trade; choose from the requirements, not to flatter any one objective.
  "scalarisation": "weighted_sum" or "achievement". weighted_sum maximises sum(w*n); it cannot reach
                   non-convex parts of the trade-off. achievement minimises the largest weighted shortfall
                   from an aspiration point (reference-point / Tchebycheff method) and can reach any Pareto point.
  "aspiration":    {obj: value} required if scalarisation is achievement - what the program is aiming for.
  "objective_floors": {obj: minimum} - program statements that are REQUIREMENTS rather than goals become hard
                   constraints at the arbiter. Use {} if none. Floors that cannot be met make every run infeasible.
  "coordination":  {"beta": 1.5-3, "gamma": 0.25-0.5, "w0": 0.5-5, "tol": 1e-4-1e-2, "max_outer": 20-60}
  "weights":       {obj: w} default priorities (they will be normalised).
  "justification": {field: one sentence citing the requirement that drove it}
"""

FORMAT = 'Return ONLY one fenced ```json block with keys normalisation, scalarisation, aspiration, objective_floors, coordination, weights, justification.'


class LLM:
    def __init__(self):
        import anthropic
        self.c = anthropic.Anthropic(api_key=open(KEY).read().strip(), base_url="https://api.anthropic.com")
        self.usage = dict(input=0, output=0, calls=0)

    def ask(self, system, user, max_tokens=2500):
        for attempt in range(4):
            try:
                r = self.c.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                           messages=[{"role": "user", "content": user}])
                self.usage["input"] += r.usage.input_tokens; self.usage["output"] += r.usage.output_tokens
                self.usage["calls"] += 1
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:
                if "credit balance" in str(e):
                    raise RuntimeError("API credit exhausted")
                time.sleep(2 ** attempt)
        raise RuntimeError("API failed")

    def cost(self):
        return (self.usage["input"] * 2.0 + self.usage["output"] * 10.0) / 1e6


def _json(text):
    m = re.findall(r"```json\s*(\{.*\})\s*```", text, re.S)
    return json.loads(m[-1]) if m else None


def validate(spec):
    errs = []
    for k in E.OBJ:
        rng = (spec.get("normalisation") or {}).get(k)
        if not (isinstance(rng, list) and len(rng) == 2 and float(rng[1]) > float(rng[0])):
            errs.append(f"normalisation.{k} must be [worst, best] with best > worst")
    if spec.get("scalarisation") not in ("weighted_sum", "achievement"):
        errs.append("scalarisation must be weighted_sum or achievement")
    if spec.get("scalarisation") == "achievement" and not all(k in (spec.get("aspiration") or {}) for k in E.OBJ):
        errs.append("aspiration needs all four objectives")
    if not all(k in (spec.get("weights") or {}) for k in E.OBJ):
        errs.append("weights needs all four objectives")
    for k in (spec.get("objective_floors") or {}):
        if k not in E.OBJ:
            errs.append(f"objective_floors key {k} unknown")
    return errs


def formulate(llm, direction_note="", base_spec=None):
    system = "You are a senior systems engineer setting up the arbiter of a multidisciplinary design optimisation."
    user = f"REQUIREMENTS (as given to the engineering leads):\n{REQUIREMENTS}\n\n"
    if direction_note:
        user += f"PROGRAM DIRECTION FOR THIS STUDY: {direction_note}\n\n"
    if base_spec:
        user += ("BASELINE FORMULATION (already reviewed; change only what the program direction requires, "
                 f"and say why in justification):\n```json\n{json.dumps(base_spec)}\n```\n\n")
    user += CATALOG + "\n" + FORMAT
    text = llm.ask(system, user)
    spec = _json(text) or {}
    errs = validate(spec)
    if errs:                                                     # one repair attempt
        text = llm.ask(system, user + "\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION: " + "; ".join(errs) +
                       "\nPrevious answer:\n" + text[:3000])
        spec = _json(text) or {}
        errs = validate(spec)
    spec["_errors"] = errs
    for k in E.OBJ:
        spec["normalisation"][k] = [float(v) for v in spec["normalisation"][k]]
    return spec


RATIONALE_SYS = ("You are the program's systems engineer writing the decision record for an architecture review. "
                 "Explain the allocation to engineers who will challenge it. Use ONLY the numbers provided; do not "
                 "invent values. Cite numbers exactly as given.")


def rationale(llm, spec, result, direction_note):
    import arm2v6_problem as P
    claims = {n: float(P.TARGET[P.NAMES.index(n)]) for n in P.NAMES}
    alloc = {n: round(float(v), 3) for n, v in zip(P.NAMES, result["x"])}
    obj = dict(zip(E.OBJ, [round(v, 3) for v in result["f"]]))
    binding = [n for n, g in zip(P.CONSTRAINTS, result["g"]) if g > -0.02]
    prices = {k: round(v, 2) for k, v in sorted(result["prices"].items(), key=lambda kv: -abs(kv[1]))}
    info = {k: round(v, 3) for k, v in result["info"].items() if k in ("p_siren", "q_ss", "f_res", "vfc", "vb", "mass", "pmarg")}
    data = dict(program_direction=direction_note or "none", formulation={k: spec[k] for k in ("scalarisation", "objective_floors", "weights")},
                objectives_achieved=obj, allocation=alloc, each_domains_original_claim=claims,
                binding_or_near_binding_constraints=binding, acoustic_details=info,
                coupling_prices=prices,
                price_meaning="Lagrange multiplier on each linking variable copy (domain:variable): larger magnitude = the arbiter and that domain disagreed more strongly and that coupling was more costly to settle; sign shows which side pushed.")
    user = ("DATA:\n```json\n" + json.dumps(data) + "\n```\n\nWrite the decision trace in at most 250 words: "
            "(1) which domains conceded what versus their original claims, (2) which couplings drove those concessions, "
            "citing the coupling prices, (3) what the design gives up, (4) one risk a reviewer should probe.")
    text = llm.ask(RATIONALE_SYS, user, max_tokens=1200)
    return text, data


def grounding(text, data):
    """Share of numbers in the trace that appear (within rounding) in the data it was given."""
    pool = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            pool.append(float(o))
    walk(data)
    pool = np.array(pool)
    nums = [float(s.replace(",", "")) for s in re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*", text)]
    nums = [n for n in nums if n not in (1, 2, 3, 4) or True]
    checked, ok = 0, 0
    for n in nums:
        if abs(n) < 1e-12:
            continue
        checked += 1
        derived_ok = np.any(np.isclose(pool, n, rtol=0.01, atol=0.006)) or np.any(np.isclose(pool * 100, n, rtol=0.01)) \
            or np.any(np.isclose(np.abs(pool), abs(n), rtol=0.01, atol=0.006))
        # differences and percentages of claims vs allocations count as grounded
        if not derived_ok:
            al, cl = data["allocation"], data["each_domains_original_claim"]
            for k in al:
                d = cl[k] - al[k]
                if cl[k] and (np.isclose(abs(d), abs(n), rtol=0.02, atol=0.6) or np.isclose(100 * al[k] / cl[k], n, atol=1.0)
                              or np.isclose(100 * abs(d) / cl[k], n, atol=1.0)):
                    derived_ok = True; break
        ok += bool(derived_ok)
    return dict(numbers=checked, grounded=ok, rate=(ok / checked if checked else 1.0))

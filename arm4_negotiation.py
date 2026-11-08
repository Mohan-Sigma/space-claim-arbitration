"""
Arm 4 - naive LLM negotiation. Main benchmark case, schema v0.7.

Eight domain agents negotiate the space claim in natural language. There is NO arbiter
mathematics: the moderator only does what a program manager's spreadsheet does (sum the
claims, show which pools are over-committed, relay messages, check that shared values
agree). Every allocation decision is made by the agents.

Each agent
  - knows the program requirements (shared brief) and its OWN requirements
  - sees only its OWN performance numbers, computed by its private evaluator
  - may request numbers from others in prose, and may "probe" alternative claims of its own
  - replies with a short message + a JSON block of its updated claims, and ACCEPT / OBJECT

Protocol per round: moderator posts the ledger -> agents speak in a fixed order, each seeing
the messages already posted this round -> claims applied -> stop when every agent ACCEPTs and
the ledger is consistent, or after MAX_ROUNDS.

Backends: AnthropicBackend (needs ANTHROPIC_API_KEY), OpenAICompatBackend (LM Studio / Ollama),
MockBackend (scripted, for testing the harness only).
"""
import json as _json, re, time, copy


class json:  # numpy-safe JSON helper
    loads = staticmethod(_json.loads)
    JSONDecodeError = _json.JSONDecodeError
    @staticmethod
    def dumps(o, **k):
        return _json.dumps(o, default=lambda v: v.item() if hasattr(v, 'item') else str(v), **k)
import numpy as np
import evaluators as ev
import arm2v6_problem as P
base = P.base

MAX_ROUNDS = 12

# ------------------------------------------------------------------ agent definitions
AGENTS = {
    "battery":   dict(vars=["V_batt", "A_batt"],
                      req="Runtime target ~60 h typical use (marketing). Cell thickness 2.6-4.8 mm; battery + 2.24 mm board stack <= 7.1 mm. "
                          "Battery footprint and the thermal spreader share the 1143 mm^2 caseback plane."),
    "board":     dict(vars=["A_board"],
                      req="10-layer board, [SoC withheld]. Routability index must stay >= 0.95 (about 1197 mm^2 minimum). Board volume = area x 2.24 mm."),
    "antenna":   dict(vars=["V_ant"],
                      req="GNSS L1+L5, LTE, Wi-Fi, BT. Non-conductive perimeter keepout 900-2400 mm^3; shares the 2400 mm^3 perimeter with the NFC loop. "
                          "Thicker titanium walls detune you. Maximise L5 efficiency."),
    "thermal":   dict(vars=["A_spr"],
                      req="Skin temperature <= 43 C at 0.9 W sustained. Spreader area 180-520 mm^2 on the caseback (volume = area x 0.78 mm). Depends on wall thickness."),
    "structure": dict(vars=["V_struct", "t_wall"],
                      req="Titanium case, 10 ATM and MIL-STD-810H. Pressure margin >= 1.5 including the speaker port opening. Ribs 1400-2400 mm^3, wall 1.0-1.8 mm. "
                          "Each extra 0.1 mm of wall removes ~150 mm^3 of interior volume from everyone."),
    "acoustics": dict(vars=["V_acou", "port", "split", "h_out", "r_mesh"],
                      req="Single 18x13 mm driver (540 mm^3 envelope), open water-tolerant cavity, hydrophobic mesh >= 20 rayl. Emergency siren 86 dB @ 1 m with <= 1.0 W driver power, "
                          "gapped 50% duty; voice intelligibility (SII). Volume 900-2400 mm^3; port 3-11 mm^2 (through the structure's wall); split = back-cavity share of free volume."),
    "haptics":   dict(vars=["V_hap"],
                      req="LRA force >= 0.9 N (about 295 mm^3 minimum)."),
    "nfc":       dict(vars=["V_loop", "V_ferr"],
                      req="NFC read range >= 10 mm. Loop 180-520 mm^3 in the perimeter keepout, ferrite 110-300 mm^3 inside."),
}
DEPS = {"structure": ["port"], "thermal": ["t_wall"], "antenna": ["t_wall"], "acoustics": ["V_batt"],
        "battery": [], "board": [], "haptics": [], "nfc": []}
ORDER = ["board", "battery", "antenna", "thermal", "structure", "acoustics", "haptics", "nfc"]

BRIEF = ("You are one engineering lead in the product-architecture review for a rugged 47 mm titanium smartwatch "
         "(reference platform: [withheld]). The case envelope is frozen by ID. All internal components must share "
         "11,200 mm^3 of interior volume at a 1.2 mm wall (each +0.1 mm of wall costs everyone 150 mm^3). Other shared pools: "
         "caseback plane 1143 mm^2 (battery footprint + thermal spreader), perimeter keepout 2400 mm^3 (antenna + NFC loop; the keepout lies INSIDE the 11,200 mm^3 interior, so antenna and "
         "NFC volumes are counted once, in the interior total, and the keepout is an extra limit on those two only - nothing is "
         "double-counted), "
         "through-wall openings 11 mm^2 for the speaker port. Program goals, all important: ~60 h battery runtime, an emergency "
         "siren of 86 dB at 1 m that keeps sounding for >= 3 hours (endurance depends on battery energy and siren power), good "
         "voice intelligibility, and strong GNSS L5 reception. The current claims over-commit the interior. You must reach "
         "agreement with the other leads. Be concise and concrete; argue from your numbers.")

REPLY_FMT = ("Probe dicts may contain your own variables and, as read-only what-ifs, the shared inputs your numbers depend on "
             "(listed under SHARED INPUTS YOU MAY PROBE). Values outside bounds are clipped and the probe result says so.\n"
             "Reply with at most 120 words of message to the other leads, then a fenced JSON block exactly like:\n"
             "```json\n{\"claims\": {<your variables and values>}, \"probe\": [<optional: up to 3 alternative claim dicts to evaluate for you>], "
             "\"status\": \"ACCEPT\" or \"OBJECT\"}\n```\n"
             "ACCEPT means you can live with the current ledger including your new claims.")


def own_metrics(agent, d):
    """Private evaluator: only the agent's own numbers."""
    t = d["t_wall"]
    if agent == "battery":
        th = d["V_batt"] / d["A_batt"]
        return dict(runtime_h=round(ev.battery_runtime_h(d["V_batt"]), 2), thickness_mm=round(th, 2),
                    z_stack_ok=th + base.T_BOARD <= base.Z_AVAIL + 1e-9, thickness_ok=2.6 <= th <= 4.8)
    if agent == "board":
        r = ev.board_routability(d["A_board"]); return dict(routability=round(r, 3), ok=r >= 0.95)
    if agent == "antenna":
        cond = d["V_ant"] / (167.0 * 6.0)
        l5 = ev.antenna_efficiency(d["V_ant"], cond, base.L5_F, base.L5_BW) * (0.8 / t) ** 0.5
        return dict(l5_efficiency=round(l5, 3))
    if agent == "thermal":
        s = ev.skin_temp_c(d["A_spr"], 0.9, t, r_case=base.R_CASE_CAL); return dict(skin_temp_c=round(s, 2), ok=s <= 43.0)
    if agent == "structure":
        m = ev.pressure_margin(t, d["port"], d["V_struct"]); return dict(pressure_margin=round(m, 3), ok=m >= 1.5)
    if agent == "acoustics":
        free = d["V_acou"] - base.V_DRIVER
        if free < 300:
            return dict(error="less than 300 mm^3 free cavity volume")
        s, q, fres, vs, _, _ = base._acoustics(free * d["split"], free * (1 - d["split"]), d["port"], d["h_out"], d["r_mesh"])
        endur = ev.battery_energy_wh(d["V_batt"]) / (P.SIREN_DUTY * s["P"] / base.AMP_EFF + 0.15)
        return dict(siren_power_w=round(s["P"], 3), siren_ok=s["P"] <= 1.0, voice_sii_score=round(vs, 1),
                    resonance_hz=round(fres), q=round(q, 2), siren_endurance_h_with_current_battery=round(endur, 2))
    if agent == "haptics":
        f = ev.haptic_force_n(d["V_hap"], base.HAPTIC_STROKE); return dict(force_n=round(f, 3), ok=f >= 0.9)
    if agent == "nfc":
        r = ev.nfc_read_range_mm(400 * d["V_loop"] / 320, d["V_ferr"] / 900, 1.0); return dict(read_range_mm=round(r, 1), ok=r >= 10)


def ledger(d):
    t = d["t_wall"]; pool = base.POOL_R1_NOM - base.DV_DT * (t - 1.2)
    vol = {"battery": d["V_batt"], "board": d["A_board"] * base.T_BOARD, "antenna": d["V_ant"],
           "thermal": d["A_spr"] * base.T_SPREADER, "structure": d["V_struct"], "acoustics": d["V_acou"],
           "haptics": d["V_hap"], "nfc": d["V_loop"] + d["V_ferr"]}
    tot = sum(vol.values())
    return dict(interior_pool_mm3=round(pool), interior_claimed_mm3=round(tot), interior_over_by_mm3=round(tot - pool),
                volume_by_domain=({k: round(v) for k, v in vol.items()}),
                caseback_claimed_mm2=round(d["A_batt"] + d["A_spr"]), caseback_pool_mm2=1143,
                perimeter_claimed_mm3=round(d["V_ant"] + d["V_loop"]), perimeter_pool_mm3=2400,
                port_mm2=round(d["port"], 2), port_pool_mm2=11, wall_mm=round(t, 3))


def ledger_ok(d):
    L = ledger(d)
    return (L["interior_over_by_mm3"] <= 0 and L["caseback_claimed_mm2"] <= 1143
            and L["perimeter_claimed_mm3"] <= 2400 and d["port"] <= 11.0)


def clip_claims(agent, claims):
    out = {}
    for k, v in claims.items():
        if k in AGENTS[agent]["vars"]:
            j = P.NAMES.index(k)
            try:
                out[k] = float(np.clip(float(v), P.XL[j], P.XU[j]))
            except (TypeError, ValueError):
                pass
    return out


def parse_reply(text):
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{\s*\"claims\".*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except json.JSONDecodeError:
        return None


def agent_prompt(agent, d, transcript_round, prev_round, probes_result, round_no):
    mine = {k: round(d[k], 3) for k in AGENTS[agent]["vars"]}
    shared = {"wall_mm (structure)": round(d["t_wall"], 3), "port_mm2 (acoustics)": round(d["port"], 2),
              "battery_volume_mm3 (battery)": round(d["V_batt"])}
    msg = [f"ROUND {round_no} of at most {MAX_ROUNDS}.",
           f"LEDGER (from the program manager's spreadsheet): {json.dumps(ledger(d))}",
           f"SHARED VALUES set by other leads: {json.dumps(shared)}",
           f"YOUR CURRENT CLAIMS: {json.dumps(mine)}",
           f"VARIABLE BOUNDS: {json.dumps({k: [float(P.XL[P.NAMES.index(k)]), float(P.XU[P.NAMES.index(k)])] for k in AGENTS[agent]['vars']})}",
           f"SHARED INPUTS YOU MAY PROBE (what-if only): {DEPS[agent] or 'none'}",
           f"YOUR NUMBERS (private evaluator): {json.dumps(own_metrics(agent, d))}"]
    if probes_result:
        msg.append(f"RESULTS OF YOUR PROBES LAST ROUND: {json.dumps(probes_result)}")
    if prev_round:
        msg.append("LAST ROUND'S MESSAGES:\n" + "\n".join(f"[{a}] {m}" for a, m in prev_round))
    if transcript_round:
        msg.append("THIS ROUND SO FAR:\n" + "\n".join(f"[{a}] {m}" for a, m in transcript_round))
    msg.append(REPLY_FMT)
    system = f"{BRIEF}\n\nYOU ARE THE {agent.upper()} LEAD. Your requirements: {AGENTS[agent]['req']}\nYour variables: {AGENTS[agent]['vars']}."
    return system, "\n\n".join(msg)


def negotiate(backend, start=None, priority_note="", seed=0, log=print):
    d = dict(zip(P.NAMES, P.TARGET if start is None else start))
    d.update(split=0.78, h_out=1.0, r_mesh=40.0) if start is None else None
    global BRIEF
    brief0 = BRIEF
    if priority_note:
        BRIEF = brief0 + " PROGRAM DIRECTION: " + priority_note
    transcript, prev_round, probes = [], [], {}
    calls = tokens = 0
    try:
        for r in range(1, MAX_ROUNDS + 1):
            this_round, statuses = [], {}
            for agent in ORDER:
                system, user = agent_prompt(agent, d, this_round, prev_round, probes.get(agent), r)
                text, ntok = backend.complete(system, user, agent=agent, state=d, round_no=r, seed=seed)
                calls += 1; tokens += ntok
                rep = parse_reply(text)
                message = re.split(r"```", text)[0].strip()[:900]
                this_round.append((agent, message))
                if rep:
                    d.update(clip_claims(agent, rep.get("claims", {})))
                    statuses[agent] = str(rep.get("status", "OBJECT")).upper()
                    pr = []
                    for alt in (rep.get("probe") or [])[:3]:
                        alt = alt if isinstance(alt, dict) else {}
                        dd = dict(d); used, notes = {}, []
                        for k, v in alt.items():
                            if k in AGENTS[agent]["vars"] or k in DEPS[agent]:
                                j = P.NAMES.index(k)
                                try:
                                    fv = float(v)
                                except (TypeError, ValueError):
                                    continue
                                cv = float(np.clip(fv, P.XL[j], P.XU[j]))
                                if abs(cv - fv) > 1e-9:
                                    notes.append(f"{k} {fv:g} clipped to bound {cv:g}")
                                dd[k] = cv; used[k] = cv
                            else:
                                notes.append(f"{k} ignored (not yours to probe)")
                        pr.append(dict(probed=used, notes=notes, numbers=own_metrics(agent, dd)))
                    probes[agent] = pr
                else:
                    statuses[agent] = "OBJECT"; probes[agent] = []
            transcript.append(dict(round=r, messages=this_round, statuses=statuses, ledger=ledger(d),
                                   claims={k: d[k] for k in P.NAMES}))
            log(f"  round {r}: over {ledger(d)['interior_over_by_mm3']} mm3, accept {sum(s == 'ACCEPT' for s in statuses.values())}/8")
            prev_round = this_round
            if all(s == "ACCEPT" for s in statuses.values()) and ledger_ok(d):
                break
    finally:
        BRIEF = brief0
    x = np.array([d[n] for n in P.NAMES])
    f, g, info = P.evaluate(x)
    return dict(x=x.tolist(), f=f.tolist(), g=g.tolist(), feasible=bool(np.all(g <= 1e-6)),
                agreed=all(s == "ACCEPT" for s in transcript[-1]["statuses"].values()),
                ledger_ok=ledger_ok(d), rounds=len(transcript), llm_calls=calls, tokens=tokens,
                violated=[n for n, v in zip(P.CONSTRAINTS, g) if v > 1e-6], transcript=transcript,
                priority_note=priority_note)


# ------------------------------------------------------------------ backends
class MockBackend:
    """Scripted stand-in to test the harness: concede 5% toward the floor while over-committed."""
    FLOORS = dict(V_batt=2600, A_batt=520, A_board=1197, V_ant=900, A_spr=320, V_struct=1400, t_wall=1.0,
                  V_acou=900, port=3.0, split=0.3, h_out=0.3, r_mesh=20, V_hap=295, V_loop=180, V_ferr=110)

    def complete(self, system, user, agent, state, round_no, seed):
        over = ledger(state)["interior_over_by_mm3"]
        claims = {}
        for k in AGENTS[agent]["vars"]:
            v = state[k]
            if over > 0 and k not in ("split", "h_out", "r_mesh", "port", "t_wall"):
                v = v - 0.05 * (v - self.FLOORS[k])
            claims[k] = v
        st = "ACCEPT" if over <= 0 else "OBJECT"
        txt = f"{agent}: conceding 5% while over-committed.\n```json\n{json.dumps(dict(claims=claims, status=st))}\n```"
        return txt, 0


class AnthropicBackend:
    def __init__(self, model, key_path, temperature=None, max_tokens=1200):
        import anthropic
        key = open(key_path).read().strip()
        # explicit base_url: the key must go to Anthropic only, never to a workspace proxy
        self.c = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")
        self.model = model; self.t = temperature; self.mt = max_tokens
        self.errors = 0

    def complete(self, system, user, **kw):
        kw = {} if self.t is None else dict(temperature=self.t)
        last = None
        for attempt in range(4):
            try:
                r = self.c.messages.create(model=self.model, max_tokens=self.mt, system=system,
                                           messages=[{"role": "user", "content": user}], **kw)
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return text, r.usage.input_tokens + r.usage.output_tokens
            except TypeError:
                raise                                   # programming error: fail immediately
            except Exception as e:
                self.errors += 1; last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API failed 4 times: {type(last).__name__}: {str(last)[:200]}")


class OpenAICompatBackend:
    """LM Studio (http://localhost:1234/v1) or Ollama (http://localhost:11434/v1)."""
    def __init__(self, base_url, model, temperature=0.7, max_tokens=500):
        import urllib.request; self.u = base_url.rstrip("/") + "/chat/completions"
        self.m, self.t, self.mt, self.req = model, temperature, max_tokens, urllib.request

    def complete(self, system, user, **kw):
        body = json.dumps(dict(model=self.m, temperature=self.t, max_tokens=self.mt,
                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])).encode()
        rq = self.req.Request(self.u, data=body, headers={"content-type": "application/json"})
        with self.req.urlopen(rq, timeout=600) as resp:
            j = json.loads(resp.read())
        return j["choices"][0]["message"]["content"], j.get("usage", {}).get("total_tokens", 0)

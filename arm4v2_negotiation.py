"""
Arm 4 v2 - naive LLM negotiation with the protocol fixes found in the first sweep.

Changes vs arm4_negotiation.py (agents, evaluators, ledger, probes unchanged):
  P1  ACCEPT means: your OWN requirements are met in your numbers AND you can live with the ledger.
      The moderator's spreadsheet records an ACCEPT as OBJECT when the agent's own pass/fail
      flags fail, and tells the agent so next round (bookkeeping, not arbitration).
  P2  Confirmation round: when all eight ACCEPT and the pools balance, claims are frozen and every
      agent must CONFIRM against the final ledger and its own numbers. Any REJECT reopens the
      negotiation (if rounds remain). Removes "stale acceptance" (a later speaker changing a
      shared value after an earlier speaker accepted).
  P3  Prompt caching: the brief + rules (system) and the round-start ledger + last round's messages
      (shared block) are identical for all eight agents in a round and are sent as a cached prefix.
"""
import re, time
import numpy as np
import arm4_negotiation as A
from arm4_negotiation import json, AGENTS, DEPS, ORDER, ledger, ledger_ok, own_metrics, clip_claims, parse_reply
P = A.P

MAX_ROUNDS = 12
RULES = ("PROTOCOL: Each round every lead speaks once, in a fixed order, seeing what was said earlier that round. "
         "Reply with at most 120 words to the other leads, then one fenced JSON block: "
         "{\"claims\": {<your variables>}, \"probe\": [<up to 3 alternative dicts>], \"status\": \"ACCEPT\" or \"OBJECT\"}. "
         "Probe dicts may contain your own variables and, as read-only what-ifs, the shared inputs listed for you; "
         "out-of-bound values are clipped and reported. ACCEPT means BOTH (a) your own requirements are met in your "
         "numbers and (b) you can live with the whole ledger. If your own requirement fails, you must OBJECT. When "
         "everyone accepts and the pools balance, claims are frozen for a CONFIRMATION round; any REJECT reopens the negotiation.")


def own_ok(agent, d):
    m = own_metrics(agent, d)
    if "error" in m:
        return False, ["error: " + m["error"]]
    fails = [k for k, v in m.items() if k.endswith("ok") and v is False]
    return (len(fails) == 0), fails


def shared_block(d, prev_round, round_no, notices):
    s = [f"ROUND {round_no} of at most {MAX_ROUNDS}.",
         f"LEDGER AT START OF ROUND: {json.dumps(ledger(d))}"]
    if notices:
        s.append("MODERATOR NOTICES: " + " ".join(notices))
    if prev_round:
        s.append("LAST ROUND'S MESSAGES:\n" + "\n".join(f"[{a}] {m}" for a, m in prev_round))
    return "\n\n".join(s)


def agent_block(agent, d, this_round, probes_result):
    mine = {k: round(d[k], 3) for k in AGENTS[agent]["vars"]}
    shared = {"wall_mm (structure)": round(d["t_wall"], 3), "port_mm2 (acoustics)": round(d["port"], 2),
              "battery_volume_mm3 (battery)": round(d["V_batt"])}
    b = [f"YOU ARE THE {agent.upper()} LEAD. Your requirements: {AGENTS[agent]['req']}",
         f"LEDGER NOW (after earlier speakers this round): {json.dumps(ledger(d))}",
         f"SHARED VALUES set by other leads: {json.dumps(shared)}",
         f"YOUR CURRENT CLAIMS: {json.dumps(mine)}",
         f"VARIABLE BOUNDS: {json.dumps({k: [float(P.XL[P.NAMES.index(k)]), float(P.XU[P.NAMES.index(k)])] for k in AGENTS[agent]['vars']})}",
         f"SHARED INPUTS YOU MAY PROBE (what-if only): {DEPS[agent] or 'none'}",
         f"YOUR NUMBERS (private evaluator): {json.dumps(own_metrics(agent, d))}"]
    if probes_result:
        b.append(f"RESULTS OF YOUR PROBES LAST ROUND: {json.dumps(probes_result)}")
    if this_round:
        b.append("THIS ROUND SO FAR:\n" + "\n".join(f"[{a}] {m}" for a, m in this_round))
    return "\n\n".join(b)


def confirm_block(agent, d):
    return (f"YOU ARE THE {agent.upper()} LEAD. Your requirements: {AGENTS[agent]['req']}\n\n"
            f"CONFIRMATION ROUND - claims are FROZEN.\nFINAL LEDGER: {json.dumps(ledger(d))}\n"
            f"YOUR FINAL NUMBERS: {json.dumps(own_metrics(agent, d))}\n\n"
            "Reply with at most 40 words, then a fenced JSON block {\"status\": \"CONFIRM\" or \"REJECT\"}. "
            "CONFIRM only if your own requirements are met and you can live with this exact ledger.")


def negotiate(backend, priority_note="", log=print):
    d = dict(zip(P.NAMES, P.TARGET))
    system = A.BRIEF + (" PROGRAM DIRECTION: " + priority_note if priority_note else "") + "\n\n" + RULES
    transcript, prev_round, probes, notices = [], [], {}, []
    usage = dict(input=0, output=0, cache_write=0, cache_read=0, calls=0)
    confirmations = []
    done = False

    def call(shared, own):
        text, u = backend.complete(system, shared, own)
        for k in ("input", "output", "cache_write", "cache_read"):
            usage[k] += u.get(k, 0)
        usage["calls"] += 1
        return text

    for r in range(1, MAX_ROUNDS + 1):
        shared = shared_block(d, prev_round, r, notices)
        notices = []
        this_round, statuses = [], {}
        for agent in ORDER:
            text = call(shared, agent_block(agent, d, this_round, probes.get(agent)))
            rep = parse_reply(text)
            this_round.append((agent, re.split(r"```", text)[0].strip()[:900]))
            if not rep:
                statuses[agent] = "OBJECT"; probes[agent] = []
                continue
            d.update(clip_claims(agent, rep.get("claims", {})))
            st = str(rep.get("status", "OBJECT")).upper()
            ok, fails = own_ok(agent, d)
            if st == "ACCEPT" and not ok:                                   # P1
                st = "OBJECT"
                notices.append(f"{agent}'s ACCEPT was recorded as OBJECT because its own requirement fails ({', '.join(fails)}).")
            statuses[agent] = st
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
        transcript.append(dict(round=r, messages=this_round, statuses=statuses, ledger=ledger(d),
                               claims={k: d[k] for k in P.NAMES}, notices=list(notices)))
        n_acc = sum(s == "ACCEPT" for s in statuses.values())
        log(f"  round {r}: over {ledger(d)['interior_over_by_mm3']} mm3, accept {n_acc}/8")
        prev_round = this_round
        if n_acc == 8 and ledger_ok(d):                                      # P2
            cshared = shared_block(d, prev_round, r, [])
            conf, cmsgs = {}, []
            for agent in ORDER:
                text = call(cshared, confirm_block(agent, d))
                rep = parse_reply(text) or {}
                st = str(rep.get("status", "REJECT")).upper()
                ok, fails = own_ok(agent, d)
                if st == "CONFIRM" and not ok:
                    st = "REJECT"
                conf[agent] = st; cmsgs.append((agent, re.split(r"```", text)[0].strip()[:400]))
            confirmations.append(dict(after_round=r, statuses=conf, messages=cmsgs))
            log(f"  confirmation after round {r}: {sum(s == 'CONFIRM' for s in conf.values())}/8 confirm")
            if all(s == "CONFIRM" for s in conf.values()):
                done = True
                break
            prev_round = this_round + [(a, f"[CONFIRMATION {conf[a]}] {m}") for a, m in cmsgs]
            notices.append("Confirmation failed: " + ", ".join(a for a, s in conf.items() if s != "CONFIRM") + " rejected; negotiation reopened.")
    x = np.array([d[n] for n in P.NAMES])
    f, g, info = P.evaluate(x)
    cost = (usage["input"] * 2.0 + usage["output"] * 10.0 + usage["cache_write"] * 2.5 + usage["cache_read"] * 0.10) / 1e6
    return dict(x=x.tolist(), f=f.tolist(), g=g.tolist(), feasible=bool(np.all(g <= 1e-6)),
                confirmed=done, rounds=len(transcript), confirmations=confirmations, usage=usage,
                est_cost_usd=round(cost, 3), violated=[n for n, v in zip(P.CONSTRAINTS, g) if v > 1e-6],
                transcript=transcript, priority_note=priority_note)


class CachedAnthropicBackend:
    def __init__(self, model, key_path, max_tokens=1200):
        import anthropic
        key = open(key_path).read().strip()
        self.c = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")   # key goes to Anthropic only
        self.model, self.mt, self.errors = model, max_tokens, 0

    def complete(self, system, shared, own):
        content = []
        if shared:
            content.append({"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": own})
        last = None
        for attempt in range(4):
            try:
                r = self.c.messages.create(model=self.model, max_tokens=self.mt,
                                           system=[{"type": "text", "text": system}],
                                           messages=[{"role": "user", "content": content}])
                u = r.usage
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return text, dict(input=u.input_tokens, output=u.output_tokens,
                                  cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
                                  cache_read=getattr(u, "cache_read_input_tokens", 0) or 0)
            except TypeError:
                raise
            except Exception as e:
                self.errors += 1; last = e
                if "credit balance" in str(e):
                    raise RuntimeError("API credit exhausted")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API failed 4 times: {type(last).__name__}: {str(last)[:200]}")


class MockBackend:
    def complete(self, system, shared, own):
        agent = re.search(r"YOU ARE THE (\w+) LEAD", own).group(1).lower()
        if "CONFIRMATION ROUND" in own:
            return 'ok\n```json\n{"status": "CONFIRM"}\n```', {}
        m = re.search(r"LEDGER NOW \(after earlier speakers this round\): (\{.*?\})\n", own)
        over = json.loads(m.group(1))["interior_over_by_mm3"]
        st = "ACCEPT" if over <= 0 else "OBJECT"
        return f"{agent} test\n```json\n" + json.dumps(dict(claims={}, status=st)) + "\n```", {}

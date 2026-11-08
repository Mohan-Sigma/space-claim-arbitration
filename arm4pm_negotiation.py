"""
A4-PM: the A4 v2 negotiation (P1 + P2 + caching) plus a program-manager agent with decision authority.

The PM speaks last in every round. Unlike the leads, the PM sees the integrated system numbers of the
current claims (all four objectives and every hard-constraint margin), as a PM would in an integration
review, and may request up to three what-if studies of complete allocations per round (results next round).
The PM may issue binding DIRECTIVES that fix any variable; leads cannot change a directed variable until
the PM changes or releases it. Agreement requires all leads to ACCEPT, the pools to balance, and the PM to
APPROVE; the confirmation round then includes the PM. Everything else is as in arm4v2_negotiation.py.
"""
import re, numpy as np
import arm4v2_negotiation as N
from arm4_negotiation import json, AGENTS, ORDER, ledger, ledger_ok, own_metrics, clip_claims, parse_reply
P = N.P
OBJ = ["runtime_h", "siren_endurance_h", "voice_sii100", "l5_efficiency"]

PM_ROLE = ("YOU ARE THE PROGRAM MANAGER. You own the product requirements and have decision authority over every "
           "allocation. You see the integrated system numbers of the current claims, which no lead sees, and you may "
           "request up to 3 what-if studies of complete allocations per round (results arrive next round). You may issue "
           "binding DIRECTIVES that fix any variable; a lead cannot change a directed variable until you change or release it "
           "(set it to null to release). Goal: an agreed design that meets EVERY hard requirement, including siren >= 3 h, "
           "with the program direction as priority. Reply with at most 150 words to the leads, then one fenced JSON block: "
           "{\"directives\": {<variable>: value or null}, \"probe\": [<up to 3 dicts of variable values>], "
           "\"status\": \"APPROVE\" or \"CONTINUE\"}. APPROVE only if the current design meets every requirement.")


def system_numbers(d):
    x = np.array([d[n] for n in P.NAMES]); f, g, info = P.evaluate(x)
    return dict(objectives=dict(zip(OBJ, [round(float(v), 3) for v in f])),
                failing_requirements=[n for n, v in zip(P.CONSTRAINTS, g) if v > 1e-6],
                siren_tone_power_w=round(float(info["p_siren"]), 3), all_requirements_met=bool(np.all(g <= 1e-6)))


def pm_block(d, this_round, directives, studies):
    b = [PM_ROLE,
         f"LEDGER: {json.dumps(ledger(d))}",
         f"CURRENT CLAIMS (all variables): {json.dumps({k: round(v, 3) for k, v in d.items()})}",
         f"VARIABLE BOUNDS: {json.dumps({n: [float(P.XL[i]), float(P.XU[i])] for i, n in enumerate(P.NAMES)})}",
         f"VARIABLE OWNERS: {json.dumps({a: AGENTS[a]['vars'] for a in ORDER})}",
         f"INTEGRATED SYSTEM NUMBERS OF CURRENT CLAIMS: {json.dumps(system_numbers(d))}",
         f"YOUR ACTIVE DIRECTIVES: {json.dumps(directives) if directives else 'none'}"]
    if studies:
        b.append(f"RESULTS OF YOUR WHAT-IF STUDIES LAST ROUND: {json.dumps(studies)}")
    b.append("THIS ROUND'S LEAD MESSAGES:\n" + "\n".join(f"[{a}] {m}" for a, m in this_round))
    return "\n\n".join(b)


def apply_directives(d, directives):
    for k, v in directives.items():
        j = P.NAMES.index(k); d[k] = float(np.clip(v, P.XL[j], P.XU[j]))


CHECKPOINT = {}

def negotiate(backend, priority_note="", log=print, ckpt=None):
    d = dict(zip(P.NAMES, P.TARGET))
    system = N.A.BRIEF + (" PROGRAM DIRECTION: " + priority_note if priority_note else "") + "\n\n" + N.RULES + \
        " A PROGRAM MANAGER has decision authority: PM DIRECTIVES are binding, and a directed variable cannot be changed by its lead."
    transcript, prev_round, probes, notices, studies = [], [], {}, [], []
    directives = {}
    usage = dict(input=0, output=0, cache_write=0, cache_read=0, calls=0)
    confirmations, done = [], False

    def call(shared, own):
        text, u = backend.complete(system, shared, own)
        for k in ("input", "output", "cache_write", "cache_read"):
            usage[k] += u.get(k, 0)
        usage["calls"] += 1
        return text

    for r in range(1, N.MAX_ROUNDS + 1):
        shared = N.shared_block(d, prev_round, r, notices) + \
            (f"\n\nPM DIRECTIVES IN FORCE (binding): {json.dumps(directives)}" if directives else "")
        notices = []
        this_round, statuses = [], {}
        for agent in ORDER:
            text = call(shared, N.agent_block(agent, d, this_round, probes.get(agent)))
            rep = parse_reply(text)
            this_round.append((agent, re.split(r"```", text)[0].strip()[:900]))
            if not rep:
                statuses[agent] = "OBJECT"; probes[agent] = []; continue
            claims = clip_claims(agent, rep.get("claims", {}))
            blocked = [k for k in claims if k in directives]
            for k in blocked:
                claims.pop(k)
            if blocked:
                notices.append(f"{agent} tried to change PM-directed {', '.join(blocked)}; ignored.")
            d.update(claims)
            st = str(rep.get("status", "OBJECT")).upper()
            ok, fails = N.own_ok(agent, d)
            if st == "ACCEPT" and not ok:
                st = "OBJECT"
                notices.append(f"{agent}'s ACCEPT was recorded as OBJECT because its own requirement fails ({', '.join(fails)}).")
            statuses[agent] = st
            pr = []
            for alt in (rep.get("probe") or [])[:3]:
                alt = alt if isinstance(alt, dict) else {}
                dd = dict(d); used = {}
                for k, v in alt.items():
                    if k in AGENTS[agent]["vars"] or k in N.A.DEPS[agent]:
                        try:
                            j = P.NAMES.index(k); dd[k] = float(np.clip(float(v), P.XL[j], P.XU[j])); used[k] = dd[k]
                        except (TypeError, ValueError):
                            pass
                pr.append(dict(probed=used, numbers=own_metrics(agent, dd)))
            probes[agent] = pr
        # ---- PM speaks last
        text = call(shared, pm_block(d, this_round, directives, studies))
        rep = parse_reply(text) or {}
        this_round.append(("PM", re.split(r"```", text)[0].strip()[:1000]))
        for k, v in (rep.get("directives") or {}).items():
            if k not in P.NAMES:
                continue
            if v is None:
                directives.pop(k, None)
            else:
                try:
                    directives[k] = float(v)
                except (TypeError, ValueError):
                    pass
        apply_directives(d, directives)
        studies = []
        for alt in (rep.get("probe") or [])[:3]:
            if not isinstance(alt, dict):
                continue
            dd = dict(d)
            for k, v in alt.items():
                if k in P.NAMES:
                    try:
                        j = P.NAMES.index(k); dd[k] = float(np.clip(float(v), P.XL[j], P.XU[j]))
                    except (TypeError, ValueError):
                        pass
            studies.append(dict(study={k: alt[k] for k in alt if k in P.NAMES}, ledger_over_mm3=ledger(dd)["interior_over_by_mm3"],
                                system=system_numbers(dd)))
        pm_status = str(rep.get("status", "CONTINUE")).upper()
        sysn = system_numbers(d)
        if pm_status == "APPROVE" and not sysn["all_requirements_met"]:
            pm_status = "CONTINUE"
            notices.append("PM's APPROVE was recorded as CONTINUE because the integrated design fails: " + ", ".join(sysn["failing_requirements"]))
        transcript.append(dict(round=r, messages=this_round, statuses=statuses, pm_status=pm_status, directives=dict(directives),
                               ledger=ledger(d), claims={k: d[k] for k in P.NAMES}, system=sysn, notices=list(notices)))
        n_acc = sum(s == "ACCEPT" for s in statuses.values())
        log(f"  round {r}: over {ledger(d)['interior_over_by_mm3']} mm3, accept {n_acc}/8, PM {pm_status}, "
            f"siren {sysn['objectives']['siren_endurance_h']}, failing {sysn['failing_requirements']}, directives {len(directives)}")
        prev_round = this_round
        if ckpt:                                    # per-round checkpoint so an API failure loses at most one round
            open(ckpt, "w").write(json.dumps(dict(transcript=transcript, usage=usage, directives=directives, claims=d)))
        if n_acc == 8 and ledger_ok(d) and pm_status == "APPROVE":
            cshared = N.shared_block(d, prev_round, r, [])
            conf, cmsgs = {}, []
            for agent in ORDER:
                t2 = call(cshared, N.confirm_block(agent, d))
                rr = parse_reply(t2) or {}
                st = str(rr.get("status", "REJECT")).upper()
                ok, _ = N.own_ok(agent, d)
                if st == "CONFIRM" and not ok:
                    st = "REJECT"
                conf[agent] = st; cmsgs.append((agent, re.split(r"```", t2)[0].strip()[:400]))
            conf["PM"] = "CONFIRM" if system_numbers(d)["all_requirements_met"] else "REJECT"
            confirmations.append(dict(after_round=r, statuses=conf, messages=cmsgs))
            log(f"  confirmation after round {r}: {sum(s == 'CONFIRM' for s in conf.values())}/9 confirm")
            if all(s == "CONFIRM" for s in conf.values()):
                done = True; break
            prev_round = this_round + [(a, f"[CONFIRMATION {conf[a]}] {m}") for a, m in cmsgs]
            notices.append("Confirmation failed: " + ", ".join(a for a, s in conf.items() if s != "CONFIRM") + " rejected; negotiation reopened.")
    x = np.array([d[n] for n in P.NAMES]); f, g, info = P.evaluate(x)
    cost = (usage["input"] * 2.0 + usage["output"] * 10.0 + usage["cache_write"] * 2.5 + usage["cache_read"] * 0.10) / 1e6
    return dict(x=x.tolist(), f=f.tolist(), g=g.tolist(), feasible=bool(np.all(g <= 1e-6)), confirmed=done,
                rounds=len(transcript), confirmations=confirmations, usage=usage, est_cost_usd=round(cost, 3),
                violated=[n for n, v in zip(P.CONSTRAINTS, g) if v > 1e-6], transcript=transcript,
                priority_note=priority_note, final_directives=directives)

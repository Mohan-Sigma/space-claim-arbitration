import sys, json, time
import arm4_negotiation as N
import os
KEY = os.environ.get("ANTHROPIC_KEY_FILE", "anthropic_key.txt")   # path to a file holding the API key (not included)
DIRECTIONS = {
    "balanced":      "",
    "battery_first": "Battery runtime is the headline spec; protect ~60 h first.",
    "siren_first":   "The emergency siren is the safety-critical feature; it must meet 86 dB @ 1 m and last >= 3 h.",
    "gnss_first":    "Outdoor athletes are the core customer; GNSS L5 reception is the top priority.",
    "voice_first":   "Call quality reviews were poor last generation; voice intelligibility is the top priority.",
    "mass_light":    "Keep every domain lean; give back any volume you do not need.",
    "schedule":      "Schedule is tight; converge quickly and avoid reopening the port or wall.",
    "battery_siren": "Battery runtime and siren endurance are joint headline specs.",
    "rf_voice":      "Connectivity and calls matter most: GNSS and voice first.",
    "no_direction":  "There is no program priority beyond the stated goals; settle it among yourselves.",
}
if __name__ != "__main__":
    pass
names = sys.argv[1].split(",") if __name__ == "__main__" else []
be = N.AnthropicBackend("claude-sonnet-5-5", KEY) if names else None
for nm in names:
    t0 = time.time()
    r = N.negotiate(be, priority_note=DIRECTIONS[nm], log=lambda m: print(f"[{nm}]{m}", flush=True))
    r.update(direction=nm, wall_s=time.time() - t0, model="claude-sonnet-5-5", temperature="model default", api_errors=be.errors)
    json.dump(r, open(f"arm4_{nm}.json", "w"), indent=1)
    print(f"[{nm}] DONE feasible={r['feasible']} agreed={r['agreed']} rounds={r['rounds']} calls={r['llm_calls']} "
          f"tokens={r['tokens']} violated={r['violated']} f={[round(v,3) for v in r['f']]} {r['wall_s']:.0f}s", flush=True)

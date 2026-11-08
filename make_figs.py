import sys, json, glob, io, contextlib, runpy, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRID = "#d9d9d6"; INK = "#333333"
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path("./arm_xmodel_compare.py")
groups, H, comp, Hu, Huc = G["groups"], G["H"], G["comp"], G["Hu"], G["Huc"]
order = [("A1 waterfall", "A1 Waterfall"), ("A1 + PM escalation", "A1-PM Waterfall + PM"), ("A2 monolithic MDO (per seed)", "A2 Monolithic MDO"),
         ("A3 hand ATC, restored", "A3 Hand ATC"), ("A4 LLM negotiation", "A4 LLM negotiation"), ("A4 + PM agent", "A4-PM Negotiation + PM"),
         ("A5d grounded, 1-sided floor", "A5 one-sided floor"),
         ("A5e Haiku 5.5", "A5 Haiku 5.5"), ("A5e Sonnet 5.5", "A5 Sonnet 5.5"), ("A5e Opus 5.5", "A5 Opus 5.5")]
stats = {}
for k, lab in order:
    reps = groups[k]
    hf = [100 * H(r["F"]) / Hu for r in reps]; hc = [100 * H(comp(r["F"])) / Huc for r in reps]
    stats[lab] = (np.mean(hf), np.std(hf), np.mean(hc), np.std(hc), len(reps))
json.dump(dict(union_n=int(sum(len(r["F"]) for v in groups.values() for r in v)), Hu=Hu, Huc=Huc, stats=stats),
          open("./fig_stats.json", "w"), indent=1)

# Fig: hypervolume shares
fig, ax = plt.subplots(figsize=(3.45, 2.9))
labs = [l for _, l in order][::-1]; y = np.arange(len(labs)); h = 0.36
for j, (idx, name, col) in enumerate([(0, "Full HV", C[0]), (2, "Compliant HV (siren $\\geq$ 3 h)", C[1])]):
    m = [stats[l][idx] for l in labs]; s = [stats[l][idx + 1] for l in labs]
    ax.barh(y + (h / 2 + 0.02) * (1 if j == 0 else -1), m, height=h, color=col, label=name, edgecolor="white", linewidth=0.8,
            xerr=s, error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.5))
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlabel("Share of union hypervolume (%)")
ax.set_xlim(0, 100); ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=6.5, columnspacing=0.8)
fig.tight_layout(); fig.savefig("./fig_hv.pdf"); plt.close(fig)

# Fig: runtime vs siren endurance
fig, ax = plt.subplots(figsize=(3.45, 2.75))
def pts(key): return np.vstack([r["F"] for r in groups[key] if len(r["F"])])
ser = [("A2 monolithic MDO (per seed)", "A2 Monolithic MDO", "#b4b2a9", "o", 7),
       ("A1 waterfall", "A1 Waterfall", C[3], "s", 11),
       ("A1 + PM escalation", "A1-PM", C[2], "P", 13),
       ("A3 hand ATC, restored", "A3 Hand ATC", C[0], "^", 13),
       ("A5e Sonnet 5.5", "A5 Sonnet 5.5", C[1], "D", 11),
       ("A5e Opus 5.5", "A5 Opus 5.5", C[6], "v", 11)]
for k, lab, col, mk, s in ser:
    F = pts(k); ax.scatter(F[:, 0], F[:, 1], s=s, marker=mk, color=col, label=lab, linewidths=0.4, edgecolors="white", zorder=3)
pr = json.load(open("./arm5_probe.json"))["probes"]["runtime"]["objectives"]
ax.axhline(3.0, color=INK, lw=0.6, ls="--"); ax.axvline(59.8, color=INK, lw=0.6, ls=":")
ax.text(41.3, 3.03, "siren requirement 3 h", fontsize=6.5, color=INK)
ax.set_xlabel("Battery runtime (h)"); ax.set_ylabel("Siren endurance (h)")
ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=6.5, handletextpad=0.2, ncol=3, columnspacing=0.6)
fig.tight_layout(); fig.savefig("./fig_scatter.pdf"); plt.close(fig)

# Fig: repair size ECDF
fig, ax = plt.subplots(figsize=(3.45, 2.35))
def moves(pat):
    out = []
    for f in glob.glob("./" + pat):
        d = json.load(open(f)); out += [100 * r["move_max"] for r in d["sweep"] + d["directions"]]
    return np.sort(np.maximum(out, 1e-3))
for pat, lab, col in [("arm5d_r*.json", "Floor in acoustics only", C[3]), ("arm5e_r*.json", "Floor split battery/acoustics", C[1]),
                      ("arm3_restored.json", None, None)]:
    if lab is None: continue
    m = moves(pat); ax.step(m, np.arange(1, len(m) + 1) / len(m), where="post", color=col, lw=1.6, label=lab)
A3r = json.load(open("./arm3_restored.json"))
m3 = np.sort(np.maximum([100 * r.get("move_max", 0) for r in A3r], 1e-3))
ax.step(m3, np.arange(1, len(m3) + 1) / len(m3), where="post", color=C[0], lw=1.6, label="Hand ATC (no floor)")
ax.axvline(2.0, color=INK, lw=0.6, ls="--"); ax.text(2.2, 0.06, "2% limit", fontsize=6.5)
ax.set_xscale("log"); ax.set_xlim(1e-3, 50); ax.set_xlabel("Largest variable move in restoration (% of range)")
ax.set_ylabel("Fraction of runs"); ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5, columnspacing=0.8)
fig.tight_layout(); fig.savefig("./fig_repair.pdf"); plt.close(fig)
print(json.dumps(stats, indent=0)); print("probe runtime", pr)

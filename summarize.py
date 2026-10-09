"""Aggregate sim results + tracking stats -> out/sim/summary.md and out/sim/summary.png."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = []
for f in sorted(glob.glob("out/sim/json/*.json")):
    clip, strat, p = os.path.basename(f)[:-5].split("__")
    for r in json.load(open(f)):
        rows.append(dict(clip=clip, strategy=strat, perturb=int(p[1:]), **r))
if not rows:
    raise SystemExit("no results in out/sim/json")

perts = sorted({r["perturb"] for r in rows})
clips = sorted({r["clip"] for r in rows})
S = ["naive", "anchored"]
L = ["## Sim pick-and-place results", "",
     "| strategy | layout shift | success | lifted | median place error | episodes |",
     "|---|---|---|---|---|---|"]
agg = {}
for s in S:
    for p in perts:
        rr = [r for r in rows if r["strategy"] == s and r["perturb"] == p]
        if not rr:
            continue
        k = sum(r["success"] for r in rr)
        pe = np.median([r["place_err_cm"] for r in rr if r["lifted"]] or [np.nan])
        agg[(s, p)] = (k / len(rr), k, len(rr))
        L.append(f"| {s} | {p} cm | **{k}/{len(rr)}** ({k/len(rr):.0%}) | "
                 f"{np.mean([r['lifted'] for r in rr]):.0%} | {pe:.1f} cm | {len(rr)} |")
for s in S:
    rr = [r for r in rows if r["strategy"] == s]
    L.append(f"| **{s} (all)** | | **{sum(r['success'] for r in rr)}/{len(rr)}** | | | |")

L += ["", "## Per clip", "",
      "| clip | your direction | marker spacing | hand frames | grasp err | release err | naive | anchored |",
      "|---|---|---|---|---|---|---|---|"]
for c in clips:
    t = dict(np.load(f"out/{c}/traj.npz"))
    v = t["valid"]
    # names are from the demonstrator's viewpoint (sitting opposite the camera)
    direction = "R→L" if t["place_cm"][0] > t["pick_cm"][0] else "L→R"
    cells = []
    for s in S:
        rr = [r for r in rows if r["clip"] == c and r["strategy"] == s]
        cells.append(f"{sum(r['success'] for r in rr)}/{len(rr)}")
    L.append(f"| {c} | {direction} | {np.linalg.norm(t['marker_b_cm']):.1f} cm | {int(v.sum())}/{len(v)} | "
             f"{float(t['grasp_err_cm']):.1f} cm | {float(t['release_err_cm']):.1f} cm | {cells[0]} | {cells[1]} |")
L += ["", "Success = cube lifted > 3 cm, then resting within 4 cm of the target. "
      "Grasp/release err = smoothed hand tip to object at the detected contact frame."]
os.makedirs("out/sim", exist_ok=True)
open("out/sim/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- figure -----------------------------------------------------------------------------------
C = {"naive": "#c0504d", "anchored": "#2e7d5b"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
w = 0.38
for i, s in enumerate(S):
    vals = [agg.get((s, p), (np.nan, 0, 0)) for p in perts]
    b = ax[0].bar(np.arange(len(perts)) + (i - .5) * w, [v[0] * 100 for v in vals], w, color=C[s], label=s)
    ax[0].bar_label(b, labels=[f"{v[1]}/{v[2]}" for v in vals], fontsize=8)
ax[0].set_xticks(range(len(perts)), [f"{p} cm" for p in perts]); ax[0].set_ylim(0, 112)
ax[0].set_xlabel("random shift of cube & target vs recorded layout"); ax[0].set_ylabel("success (%)")
ax[0].set_title(f"Sim success vs layout change ({len(clips)} demos)"); ax[0].legend(frameon=False, loc="lower left")

rng = np.random.default_rng(0)
for s, mk in [("naive", "o"), ("anchored", "s")]:
    rr = [r for r in rows if r["strategy"] == s and r["lifted"]]
    ax[1].scatter([perts.index(r["perturb"]) + (.18 if s == "anchored" else -.18) + rng.uniform(-.07, .07) for r in rr],
                  [r["place_err_cm"] for r in rr], c=C[s], marker=mk, label=s, alpha=.75, s=22)
ax[1].axhline(4, ls="--", c="gray", lw=1)
ax[1].set_xticks(range(len(perts)), [f"{p} cm" for p in perts])
ax[1].set_ylabel("final cube-to-target error (cm)"); ax[1].set_title("Placement error (cube lifted)")
ax[1].legend(frameon=False)

x = np.arange(len(clips))
for i, s in enumerate(S):
    sr = [np.mean([r["success"] for r in rows if r["clip"] == c and r["strategy"] == s]) * 100 for c in clips]
    ax[2].bar(x + (i - .5) * w, sr, w, color=C[s], label=s)
ax[2].set_xticks(x, clips, rotation=30, fontsize=8); ax[2].set_ylim(0, 105)
ax[2].set_ylabel("success over all layouts (%)"); ax[2].set_title("Per demo")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("out/sim/summary.png", dpi=120)
print("\nwrote out/sim/summary.md, out/sim/summary.png")

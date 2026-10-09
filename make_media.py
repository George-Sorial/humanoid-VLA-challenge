"""Figures/GIFs: all hand tracks in the table frame + demo|naive|anchored comparison GIFs."""
import argparse
import glob
import os

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retarget import robust_smooth

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
args = ap.parse_args()

clips = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("out/*/traj.npz"))

# ---- tracks figure ----------------------------------------------------------------------
cols = min(5, len(clips)); rws = int(np.ceil(len(clips) / cols))
fig, axs = plt.subplots(rws, cols, figsize=(4.4 * cols, 3.6 * rws), squeeze=False)
for a in axs.flat:
    a.axis("off")
for a, c in zip(axs.flat, clips):
    a.axis("on")
    d = dict(np.load(f"out/{c}/traj.npz")); xy, v = d["xy"], d["valid"]
    g, r = int(d["grasp_idx"]), int(d["release_idx"]); s = robust_smooth(xy, v)
    for cx, cy in [(0, 0), tuple(d["marker_b_cm"])]:
        a.add_patch(plt.Rectangle((cx - 5, cy - 5), 10, 10, fc="k", alpha=.75))
    idx = np.where(v)[0]
    a.scatter(xy[idx, 0], xy[idx, 1], c=idx, cmap="viridis", s=9, alpha=.6)
    a.plot(s[idx[0]:idx[-1] + 1, 0], s[idx[0]:idx[-1] + 1, 1], "-", c="#555", lw=1)
    a.plot(*d["pick_cm"], "b*", ms=13); a.plot(*d["place_cm"], "c*", ms=13)
    a.plot(*s[g], "ro", ms=7); a.plot(*s[r], "ms", ms=7)
    a.set_title(c, fontsize=10); a.set_aspect("equal"); a.set_xlim(-12, 50); a.set_ylim(8, -24)
    a.tick_params(labelsize=7)
fig.suptitle("hand tip (dots: raw, line: robust-smoothed) | blue*: pick, cyan*: place | "
             "red: grasp, magenta: release | table frame, cm", fontsize=9)
plt.tight_layout(); plt.savefig("out/tracks.png", dpi=100); plt.close()
print("wrote out/tracks.png")

# ---- comparison GIFs ---------------------------------------------------------------------
os.makedirs("out/sim/gifs", exist_ok=True)
for c in clips:
    vn, va = f"out/sim/videos/{c}_naive.mp4", f"out/sim/videos/{c}_anchored.mp4"
    if not (os.path.exists(vn) and os.path.exists(va)):
        continue
    cap = cv2.VideoCapture(f"{args.data}/{c}.mp4"); hf = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        hf.append(cv2.resize(f, (320, 180))[:, :, ::-1])
    nf = [np.asarray(f) for f in imageio.get_reader(vn)]
    af = [np.asarray(f) for f in imageio.get_reader(va)]
    n = max(len(nf), len(af)); out = []
    sep = np.full((180, 4, 3), 255, np.uint8)
    for k in range(0, n, 4):
        h = hf[int(k * len(hf) / n)]
        q = cv2.resize(np.ascontiguousarray(nf[min(k, len(nf) - 1)]), (240, 180))
        s = cv2.resize(np.ascontiguousarray(af[min(k, len(af) - 1)]), (240, 180))
        fr = np.hstack([h, sep, q, sep, s]).copy()
        cv2.putText(fr, f"my demo: {c}", (6, 16), 0, 0.45, (255, 255, 255), 1)
        cv2.putText(fr, "naive replay", (330, 16), 0, 0.45, (0, 0, 0), 1)
        cv2.putText(fr, "anchored replay", (574, 16), 0, 0.45, (0, 0, 0), 1)
        out.append(fr)
    imageio.mimsave(f"out/sim/gifs/{c}_compare.gif", out, duration=1 / 12, loop=0)
print(f"wrote out/sim/gifs/*_compare.gif")

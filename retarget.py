"""Retarget an extracted human trajectory to Panda end-effector waypoints in sim.

Two strategies (this comparison is the core experiment):
  naive    : fixed cm->m scale, translated so the grasp lands on the sim pick point.
  anchored : similarity transform (scale + rotation + translation) fitted so the
             human GRASP point -> sim pick point and human RELEASE point -> sim place
             point. Absorbs workspace size / orientation mismatch.

Height is recovered from a monocular proxy (hand size / marker size), normalised
between "at table" (grasp/release) and "peak lift" and mapped to [z_table, z_lift].

Gripper convention follows LIBERO: -1 = open, +1 = closed.

Usage:
  python retarget.py out/clip01.npz wp/clip01_anchored.npz --strategy anchored \
      --pick-xy -0.10 0.0 --place-xy 0.10 0.0 --z-table 0.90 --z-lift 1.00
"""
import argparse

import numpy as np
from scipy.signal import savgol_filter


def fill_nan(x):
    x = x.copy()
    idx = np.arange(len(x))
    for k in range(x.shape[1]):
        good = ~np.isnan(x[:, k])
        x[:, k] = np.interp(idx, idx[good], x[good, k])
    return x


def robust_smooth(x, valid, med=7, win=9, poly=2):
    """Outlier-robust: rolling median over VALID samples only, then interpolate gaps,
    then a light Savitzky-Golay. Plain SG over noisy tracks shifted contacts by up to 11 cm."""
    x = np.asarray(x, float).copy()
    idx = np.where(valid)[0]
    out = x.copy()
    h = med // 2
    for j, i in enumerate(idx):
        nb = idx[max(j - h, 0): j + h + 1]
        out[i] = np.median(x[nb], axis=0)
    out[~valid] = np.nan
    return smooth(fill_nan(out), win, poly)


def smooth(x, win=15, poly=3):
    win = min(win, len(x) - (1 - len(x) % 2))
    return savgol_filter(x, win, poly, axis=0)


def detect_grasp(aperture):
    lo, hi = np.percentile(aperture, [10, 90])
    closed = aperture < (lo + hi) / 2
    idx = np.where(closed)[0]
    if len(idx) == 0:
        raise ValueError("no grasp detected - check aperture signal / hand visibility")
    return closed, idx[0], idx[-1]


def similarity(a0, a1, b0, b1):
    """2D similarity mapping a0->b0 and a1->b1. Returns f(points)."""
    va, vb = a1 - a0, b1 - b0
    s = np.linalg.norm(vb) / (np.linalg.norm(va) + 1e-9)
    th = np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return lambda p: b0 + s * (p - a0) @ R.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--strategy", choices=["naive", "anchored"], default="anchored")
    ap.add_argument("--pick-xy", type=float, nargs=2, required=True, help="sim pick point (m)")
    ap.add_argument("--place-xy", type=float, nargs=2, required=True, help="sim place point (m)")
    ap.add_argument("--z-table", type=float, required=True)
    ap.add_argument("--z-lift", type=float, required=True)
    ap.add_argument("--flip-y", action="store_true", help="mirror the table y axis")
    ap.add_argument("--n", type=int, default=120, help="number of output waypoints")
    args = ap.parse_args()

    d = np.load(args.traj)
    valid = d["valid"]
    if valid.sum() < 10:
        raise SystemExit("too few valid frames")
    xy = robust_smooth(d["xy"], valid.astype(bool))
    hp = smooth(fill_nan(d["height_proxy"][:, None]))[:, 0]
    apert = smooth(fill_nan(d["aperture"][:, None]))[:, 0]
    if args.flip_y:
        xy[:, 1] *= -1

    closed, g, r = detect_grasp(apert)
    pick, place = np.array(args.pick_xy), np.array(args.place_xy)

    if args.strategy == "anchored":
        f = similarity(xy[g], xy[r], pick, place)
        sim_xy = f(xy)
    else:  # naive: cm -> m, no rotation, grasp pinned to pick point
        sim_xy = pick + 0.01 * (xy - xy[g])

    h_ref = np.median(np.r_[hp[max(g - 3, 0): g + 3], hp[max(r - 3, 0): r + 3]])
    h_peak = np.percentile(hp[g:r + 1], 95)
    hn = np.clip((hp - h_ref) / (h_peak - h_ref + 1e-9), 0, 1)
    z = args.z_table + hn * (args.z_lift - args.z_table)
    z[:g] = np.maximum(z[:g], args.z_table)  # approach stays above table

    grip = np.where(closed, 1.0, -1.0)
    pos = np.c_[sim_xy, z]

    # trim to the useful segment (a little before grasp to a little after release)
    a, b = max(g - 20, 0), min(r + 20, len(pos))
    sel = np.linspace(a, b - 1, args.n).astype(int)
    np.savez(args.out, pos=pos[sel], grip=grip[sel], grasp_idx=g, release_idx=r,
             strategy=args.strategy)
    print(f"{args.strategy}: grasp@{g} release@{r} -> {args.out} ({args.n} waypoints)")


if __name__ == "__main__":
    main()

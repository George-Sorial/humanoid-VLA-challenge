"""Video -> camera-motion-free hand trajectory + object pick/place points (table frame, cm).

No learned hand model needed. Works because two flat ArUco markers give a per-frame
homography image -> table plane.

  1. Detect ArUco markers (4x4) in every frame.
  2. AUTO-CALIBRATE the layout: marker A (default: the left one in the image) defines the
     table frame (its centre = origin, axes aligned with the image, units cm). Marker B's
     four corners are expressed in A's frame using A's own homography, median over all
     frames. So you never measure the marker spacing, and B may be rotated/offset.
  3. Warp every frame into a rectified top-down "table view" (camera shake disappears).
  4. Object pick / place = object-sized blobs that differ between the first and last
     rectified frames; the one that is darker at the START is the pick (either direction).
  5. Hand = foreground that differs from BOTH the start and end scene (removes the static
     object "ghosts"); tip = blob point farthest from where the arm enters the view.
  6. Grasp / release frames = closest approach of the outlier-robust smoothed tip to the
     pick / place point (release must come after grasp).

Outputs  <out>.npz : t, xy [N,2] (cm), valid, pick_cm, place_cm, grasp_idx, release_idx,
                     aperture (binary surrogate), height_proxy (surrogate bump), marker_b_cm
and optional debug images (rectified montage with tip / pick / place overlaid).

Usage:
  python track_hand.py data/L2R_1.mp4 out/L2R_1/traj.npz --debug-dir out/L2R_1
"""
import argparse
import os

import cv2
import numpy as np

from retarget import robust_smooth

SCALE = 8.0                                   # rectified px per cm
X0, X1, Y0, Y1 = -14.0, 54.0, -24.0, 28.0     # rectified crop of the table frame (cm)


def detect_all(path):
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
    )
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, marks = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        c, ids, _ = det.detectMarkers(f)
        frames.append(f)
        marks.append({int(i): x.reshape(4, 2) for x, i in zip(c, ids.flatten())} if ids is not None else {})
    return frames, marks, fps


def square(size, rot):
    """Model corners of a marker centred at 0, in DETECTOR order, image-aligned."""
    h = size / 2.0
    std = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], np.float32)  # TL, TR, BR, BL
    return np.array([std[(k - rot) % 4] for k in range(4)], np.float32)


def calibrate(marks, size, a_id=None):
    """Return {id: 4x2 model corners in table frame (cm)} for the two markers."""
    ids = sorted({i for m in marks for i in m})
    if len(ids) < 2:
        raise SystemExit(f"need two markers, found ids {ids}")
    if a_id is None:  # A = marker that is on the left in the image
        xs = {i: np.mean([m[i][:, 0].mean() for m in marks if i in m]) for i in ids}
        a_id = min(xs, key=xs.get)
    b_id = [i for i in ids if i != a_id][0]
    first_a = next(m[a_id] for m in marks if a_id in m)
    A = square(size, int(np.argmin(first_a.sum(1))))
    bs = []
    for m in marks:
        if a_id in m and b_id in m:
            H, _ = cv2.findHomography(m[a_id].astype(np.float32), A, 0)
            bs.append(cv2.perspectiveTransform(m[b_id].reshape(1, 4, 2).astype(np.float32), H)[0])
    B = np.median(np.stack(bs), 0).astype(np.float32)
    return {a_id: A, b_id: B}, a_id, b_id


def cm_from_px(p):
    return np.asarray(p) / SCALE + np.array([X0, Y0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--marker-size", type=float, default=10.0, help="printed marker side, cm")
    ap.add_argument("--a-id", type=int, default=None, help="marker defining the origin (default: left one)")
    ap.add_argument("--thr", type=int, default=80, help="colour-difference threshold")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    frames, marks, fps = detect_all(args.video)
    model, a_id, b_id = calibrate(marks, args.marker_size, args.a_id)
    b_centre = model[b_id].mean(0)

    S = np.array([[SCALE, 0, -X0 * SCALE], [0, SCALE, -Y0 * SCALE], [0, 0, 1]])
    W, Hh = int((X1 - X0) * SCALE), int((Y1 - Y0) * SCALE)
    warped, valids = [], []
    for f, m in zip(frames, marks):
        m = {i: c for i, c in m.items() if i in model}
        if not m:
            warped.append(None); valids.append(None)
            continue
        src = np.concatenate([m[i] for i in m]).astype(np.float32)
        dst = np.concatenate([model[i] for i in m])
        H, _ = cv2.findHomography(src, dst, 0)
        M = S @ H
        warped.append(cv2.warpPerspective(f, M, (W, Hh)))
        valids.append(cv2.warpPerspective(np.full(f.shape[:2], 255, np.uint8), M, (W, Hh)) > 0)

    n = len(warped)
    ok = [i for i in range(n) if warped[i] is not None]
    first, last = ok[0], ok[-1]
    blur = lambda im: cv2.GaussianBlur(im, (9, 9), 0).astype(np.int16)
    bg, bgL = blur(warped[first]), blur(warped[last])

    # ---- object pick / place from first-vs-last diff -------------------------------
    d = np.abs(bgL - bg).sum(2)
    dm = ((d > args.thr) & valids[first] & valids[last]).astype(np.uint8)
    dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(dm)
    g0 = cv2.cvtColor(warped[first], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gL = cv2.cvtColor(warped[last], cv2.COLOR_BGR2GRAY).astype(np.float32)
    best_pick, best_place = (None, 0.0), (None, 0.0)
    for k in range(1, nlab):
        area = stats[k, cv2.CC_STAT_AREA] / SCALE**2
        wcm, hcm = stats[k, cv2.CC_STAT_WIDTH] / SCALE, stats[k, cv2.CC_STAT_HEIGHT] / SCALE
        if not (8 < area < 200 and 3 < wcm < 16 and 3 < hcm < 16):
            continue  # not object-sized (shadows, lighting drift, paper edges)
        score = float((gL[lab == k] - g0[lab == k]).mean()) * area  # >0: darker at start
        if score > best_pick[1]:
            best_pick = (k, score)
        if score < best_place[1]:
            best_place = (k, score)
    if best_pick[0] is None or best_place[0] is None:
        raise SystemExit("object not found at start/end: keep it fully visible in first & last frame")
    pick_px, place_px = cents[best_pick[0]], cents[best_place[0]]
    pick_cm, place_cm = cm_from_px(pick_px), cm_from_px(place_px)

    # ---- hand tip per frame ----------------------------------------------------------
    xy = np.full((n, 2), np.nan)
    dbg = []
    for i in ok:
        wi = blur(warped[i])
        vm = valids[i]
        fg = ((np.abs(wi - bg).sum(2) > args.thr) & (np.abs(wi - bgL).sum(2) > args.thr) & vm).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        nl, lb, st, _ = cv2.connectedComponentsWithStats(fg)
        if nl <= 1:
            continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < 2500:
            continue  # no hand in view
        blob = lb == k
        # arm enters through the camera-FOV border OR the rectified-crop border
        inner = cv2.erode(vm.astype(np.uint8), np.ones((41, 41), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        edge = blob & vm & ~inner
        ys, xs = np.nonzero(blob)
        pts = np.c_[xs, ys].astype(np.float32)
        if edge.sum() > 50:
            dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)[ys, xs]
            tip = pts[dist >= np.percentile(dist, 98)].mean(0)
        else:
            tip = pts.mean(0)
        xy[i] = cm_from_px(tip)
        if args.debug_dir:
            vis = warped[i].copy()
            vis[blob] = (0.5 * vis[blob] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.circle(vis, tuple(int(v) for v in tip), 14, (0, 255, 0), 3)
            cv2.circle(vis, tuple(int(v) for v in pick_px), 10, (255, 0, 0), 2)
            cv2.circle(vis, tuple(int(v) for v in place_px), 10, (255, 255, 0), 2)
            cv2.putText(vis, f"f{i}", (10, 30), 0, 1, (255, 255, 255), 2)
            dbg.append(vis)

    valid = ~np.isnan(xy[:, 0])
    if valid.sum() < 10:
        raise SystemExit(f"hand found in only {valid.sum()} frames")

    # ---- contacts on the robust-smoothed track -----------------------------------------
    sxy = robust_smooth(xy, valid)
    fv, lv = np.where(valid)[0][[0, -1]]
    dp = np.linalg.norm(sxy - pick_cm, axis=1); dp[:fv] = dp[lv + 1:] = np.nan
    dq = np.linalg.norm(sxy - place_cm, axis=1); dq[:fv] = dq[lv + 1:] = np.nan
    g = int(np.nanargmin(dp))
    after = np.where(np.arange(n) > g, dq, np.nan)
    r = int(np.nanargmin(after)) if np.isfinite(after).any() else g
    aperture = np.where((np.arange(n) >= g) & (np.arange(n) <= r), 0.2, 0.8)
    t = np.clip((np.arange(n) - g) / max(r - g, 1), 0, 1)
    height_proxy = 1.0 + 0.5 * np.sin(np.pi * t)  # surrogate, NOT measured

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, t=np.arange(n) / fps, xy=xy, valid=valid, pick_cm=pick_cm, place_cm=place_cm,
             grasp_idx=g, release_idx=r, aperture=aperture, height_proxy=height_proxy,
             marker_b_cm=b_centre, a_id=a_id, b_id=b_id,
             grasp_err_cm=float(np.linalg.norm(sxy[g] - pick_cm)),
             release_err_cm=float(np.linalg.norm(sxy[r] - place_cm)))
    direction = "image left->right" if place_cm[0] > pick_cm[0] else "image right->left"
    print(f"{os.path.basename(args.video)}: frames={n} hand={valid.sum()} grasp=f{g} release=f{r} "
          f"| markers {a_id}->{b_id} {np.linalg.norm(b_centre):.1f} cm apart "
          f"| pick={pick_cm.round(1)} place={place_cm.round(1)} ({direction}) "
          f"| contact err {np.linalg.norm(sxy[g]-pick_cm):.1f}/{np.linalg.norm(sxy[r]-place_cm):.1f} cm")

    if args.debug_dir and dbg:
        os.makedirs(args.debug_dir, exist_ok=True)
        idx = np.linspace(0, len(dbg) - 1, 8).astype(int)
        tiles = [cv2.resize(dbg[j], (480, int(480 * Hh / W))) for j in idx]
        cv2.imwrite(f"{args.debug_dir}/rectified_montage.jpg",
                    np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))


if __name__ == "__main__":
    main()

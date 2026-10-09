"""Behaviour cloning: train a goal-conditioned MLP policy on the retargeted demo rollouts.

Input  : 12-D state features (policy.features): eef pos, gripper, cube-eef, target-eef, cube-target.
Output : 4-D action [dx, dy, dz, gripper] in the OSC delta-pose action space (rot. deltas = 0).
Data   : states/<clip>_s<seed>.npz from `sim_replay.py --save-states` (successful anchored
         replays of MY demos on randomised layouts, with DART-style action noise).
Split  : by demo clip (whole clips held out) so validation measures generalisation across demos.

  python train_policy.py --data states --out out/policy
"""
import argparse
import glob
import json
import os
import re
import time

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOOKAHEAD = 5  # control steps (0.25 s at 20 Hz)


def anticipate(g, k):
    """Shift every gripper switch k steps earlier.

    In the replays the gripper only starts closing AFTER the arm has arrived, so the state
    'at the cube, gripper open' is labelled both 'open' (approach) and 'close' (one step later).
    A regressor averages the two and sits at ~0 forever. Labelling the last k approach steps
    with the upcoming command removes that ambiguity (same for release)."""
    g = g.copy()
    for t in np.where(np.diff(g) != 0)[0]:          # g[t] != g[t+1]
        g[max(t + 1 - k, 0): t + 1] = g[t + 1]
    return g


def load(files, k=LOOKAHEAD):
    X, Y = [], []
    for f in files:
        d = np.load(f)
        X.append(d["feats"]); a = d["actions"]
        Y.append(np.c_[a[:, :3], anticipate(a[:, 6], k) if k else a[:, 6]])
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="states")
    ap.add_argument("--out", default="out/policy")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD, help="anticipatory gripper labels (steps); 0 = raw")
    ap.add_argument("--val-clips", nargs="*", default=["L2R_10", "R2L_10"], help="clips held out for validation")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data}/*.npz"))
    clip_of = lambda f: re.sub(r"_s\d+\.npz$", "", os.path.basename(f))
    clips = sorted({clip_of(f) for f in files})
    val = [f for f in files if clip_of(f) in args.val_clips]
    tr = [f for f in files if f not in val]
    Xtr, Ytr = load(tr, args.lookahead)
    print(f"{len(files)} episodes from {len(clips)} demos | train {len(tr)} eps / {len(Xtr)} steps | "
          f"val {len(val)} eps (held-out demos {args.val_clips})")

    t0 = time.time()
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(256, 256, 256), activation="relu", batch_size=256, learning_rate_init=1e-3,
        max_iter=args.epochs, alpha=1e-5, random_state=0, early_stopping=False, verbose=False))
    model.fit(Xtr, Ytr)
    mlp = model[-1]
    metrics = dict(lookahead=args.lookahead, train_episodes=len(tr), train_steps=int(len(Xtr)), demos=len(clips),
                   train_time_s=round(time.time() - t0, 1), loss_curve=[float(x) for x in mlp.loss_curve_])
    if val:
        Xv, Yv = load(val, args.lookahead)
        P = model.predict(Xv)
        metrics.update(val_episodes=len(val), val_steps=int(len(Xv)),
                       val_mse_xyz=float(np.mean((np.clip(P[:, :3], -1, 1) - Yv[:, :3]) ** 2)),
                       val_grip_acc=float(np.mean(np.sign(P[:, 3]) == np.sign(Yv[:, 3]))))
    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, f"{args.out}/policy.joblib")
    json.dump(metrics, open(f"{args.out}/train_metrics.json", "w"), indent=1)
    print({k: v for k, v in metrics.items() if k != "loss_curve"})


if __name__ == "__main__":
    main()

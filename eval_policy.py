"""Closed-loop evaluation of the trained policy in sim on layouts it never saw.

Each episode: pick a recorded demo's table layout (cube start / target, mirrored into sim),
shift both by up to --perturb cm with a held-out random seed, then let the policy act from
state alone for --steps control steps (20 Hz). Same success test as sim_replay.

  python eval_policy.py --model out/policy/policy.joblib --episodes 20 --perturb 6 --tag indist
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np

from policy import features, twin_layout
from sim_replay import make_env


def episode(env, model, cube_xy, target_xy, steps, frames=None):
    env.reset()
    cj = env.cube.joints[0]
    qp = env.sim.data.get_joint_qpos(cj).copy()
    qp[:2] = cube_xy; qp[3:] = [1, 0, 0, 0]
    env.sim.data.set_joint_qpos(cj, qp); env.sim.forward()
    for _ in range(10):
        o, *_ = env.step(np.zeros(7))
    cube_z = o["cube_pos"][2]; max_z = cube_z
    first_success = None  # step at which the task was first achieved (LIBERO-style termination)
    for t in range(steps):
        y = model.predict(features(o, target_xy)[None])[0]
        a = np.zeros(7); a[:3] = np.clip(y[:3], -1, 1); a[6] = 1.0 if y[3] > 0 else -1.0
        o, *_ = env.step(a)
        max_z = max(max_z, o["cube_pos"][2])
        c = o["cube_pos"]
        if first_success is None and max_z > cube_z + 0.03 and c[2] < cube_z + 0.02 \
                and np.linalg.norm(c[:2] - target_xy) < 0.04:
            first_success = t
            if frames is None:
                pass  # keep running so the end-of-episode metric is also measured
        if frames is not None:
            frames.append(np.ascontiguousarray(env.sim.render(camera_name="frontview", width=320, height=240)[::-1]))
    final = o["cube_pos"]
    err = float(np.linalg.norm(final[:2] - target_xy))
    lifted = bool(max_z > cube_z + 0.03)
    return dict(success=first_success is not None,                       # standard: terminate on success
                success_at_end=bool(lifted and err < 0.04 and final[2] < cube_z + 0.02),  # stricter: still there at t=end
                first_success_step=first_success, lifted=lifted, place_err_cm=round(err * 100, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/policy/policy.joblib")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--perturb", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--seed-offset", type=int, default=1000, help="held out: data generation uses seeds < 1000")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], help="i n: run episodes i, i+n, ...")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--videos", type=int, default=2, help="record the first N episodes of shard 0")
    args = ap.parse_args()

    model = joblib.load(args.model)
    trajs = sorted(glob.glob("out/*/traj.npz"))
    env = make_env(cams=False)
    out = []
    for i in range(args.shard[0], args.episodes, args.shard[1]):
        d = dict(np.load(trajs[i % len(trajs)]))
        rng = np.random.default_rng(args.seed_offset + i)
        c0, t0 = twin_layout(d)
        cube = c0 + rng.uniform(-1, 1, 2) * args.perturb * 0.01
        targ = t0 + rng.uniform(-1, 1, 2) * args.perturb * 0.01
        frames = [] if (args.shard[0] == 0 and i < args.videos * args.shard[1]) else None
        r = episode(env, model, cube, targ, args.steps, frames)
        r.update(ep=i, layout_from=os.path.basename(os.path.dirname(trajs[i % len(trajs)])),
                 cube=cube.round(3).tolist(), target=targ.round(3).tolist())
        out.append(r); print(json.dumps(r), flush=True)
        if frames:
            import imageio
            os.makedirs("out/policy/videos", exist_ok=True)
            imageio.mimsave(f"out/policy/videos/{args.tag}_ep{i}.mp4", frames, fps=20)
    os.makedirs("out/policy", exist_ok=True)
    json.dump(out, open(f"out/policy/eval_{args.tag}_{args.shard[0]}.json", "w"))
    print(f"SUMMARY {args.tag}: success {sum(r['success'] for r in out)}/{len(out)} | "
          f"still placed at end {sum(r['success_at_end'] for r in out)}/{len(out)}")


if __name__ == "__main__":
    main()

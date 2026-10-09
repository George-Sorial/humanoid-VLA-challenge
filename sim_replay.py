"""Replay retargeted human demos on a simulated Panda (robosuite, MuJoCo) and score them.

Scene = digital twin of the recorded table: the cube starts where the real pyramid
started and the target is where it ended (both measured from the video via markers).
Actions are 7-D [dx,dy,dz, droll,dpitch,dyaw, gripper] through an operational-space
controller -- the same action space LIBERO / SmolVLA-LIBERO use.

Strategies
  naive    : metric replay. Real table cm -> sim m with one fixed rigid mapping; the
             recorded hand path is executed as-is.
  anchored : same mapping, then a 2-D similarity warp so the hand's GRASP point lands
             on the cube and its RELEASE point on the target (works for any new layout).

--perturb R moves the cube start and target by up to R cm (random) to test whether a
strategy generalises beyond the exact recorded layout (MimicGen-style augmentation).

Height: a top-down monocular view gives no reliable z, so z is synthesised from the
horizontal distance to the next contact ("distance-gated descent"): hover high far from
the object, descend as the hand converges on it. Gripper timing comes from the video.

Usage:
  MUJOCO_GL=egl python sim_replay.py out/clip02/traj.npz --strategy anchored \
      --perturb 0 --seeds 1 --video out/sim/clip02_anchored.mp4 --save-rollout rollouts/
"""
import argparse
import json
import os

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from retarget import robust_smooth, similarity

TABLE_Z = 0.80          # robosuite Lift table top
GRASP_DZ = 0.005        # eef height above cube centre when grasping
HOVER = 0.12            # hover height above grasp height
STEP = 0.05             # OSC max translation per step (m) at |action|=1


def twin_map(pick_cm, place_cm):
    """Fixed rigid map real table (cm) -> sim (m): rotate 90deg, centre between contacts."""
    mid = (pick_cm + place_cm) / 2.0
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    return lambda p: 0.01 * (np.atleast_2d(p) - mid) @ R.T


def build_path(d, strategy, cube_xy, target_xy, mapf, n=160):
    xy = robust_smooth(d["xy"], d["valid"].astype(bool))
    g, r = int(d["grasp_idx"]), int(d["release_idx"])
    sxy = mapf(xy)
    if strategy == "anchored":
        sxy = similarity(sxy[g], sxy[r], cube_xy, target_xy)(sxy)
    a, b = max(g - 25, 0), min(r + 15, len(sxy) - 1)
    idx = np.linspace(a, b, n)
    path = np.c_[np.interp(idx, np.arange(len(sxy)), sxy[:, 0]),
                 np.interp(idx, np.arange(len(sxy)), sxy[:, 1])]
    gi = int(np.argmin(np.abs(idx - g)))
    ri = int(np.argmin(np.abs(idx - r)))
    # distance-gated height: descend as path converges on the contact it's heading to
    z = np.empty(n)
    for k in range(n):
        if k <= gi:
            dist = np.linalg.norm(path[k] - path[gi])
        elif k <= ri:
            dist = min(np.linalg.norm(path[k] - path[gi]), np.linalg.norm(path[k] - path[ri]))
        else:
            dist = np.linalg.norm(path[k] - path[ri])
        z[k] = min(HOVER, 1.5 * dist)
    return path, z, gi, ri


def make_env(cam=128, cams=True):
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    return suite.make(
        "Lift", robots="Panda", controller_configs=cfg, has_renderer=False,
        has_offscreen_renderer=True, use_camera_obs=cams,
        camera_names=["agentview", "robot0_eye_in_hand"] if cams else [], camera_heights=cam, camera_widths=cam,
        control_freq=20, horizon=2000, ignore_done=True,
    )


def run_episode(env, d, strategy, rng, perturb_cm, frames=None, log=None):
    env.reset()
    pick_cm, place_cm = d["pick_cm"], d["place_cm"]
    mapf = twin_map(pick_cm, place_cm)
    cube0 = mapf(pick_cm)[0]
    targ0 = mapf(place_cm)[0]
    jit = lambda: rng.uniform(-1, 1, 2) * perturb_cm * 0.01
    cube_xy, target_xy = cube0 + jit(), targ0 + jit()

    cube_j = env.cube.joints[0]
    qp = env.sim.data.get_joint_qpos(cube_j).copy()
    qp[:2] = cube_xy
    qp[3:] = [1, 0, 0, 0]
    env.sim.data.set_joint_qpos(cube_j, qp)
    env.sim.forward()
    for _ in range(10):
        o, *_ = env.step(np.zeros(7))
    cube_z = o["cube_pos"][2]

    path, zoff, gi, ri = build_path(d, strategy, cube_xy, target_xy, mapf)
    grasp_z = cube_z + GRASP_DZ

    def goto(p, grip, max_steps=25, tol=0.008):
        nonlocal o
        for _ in range(max_steps):
            err = p - o["robot0_eef_pos"]
            a = np.zeros(7)
            a[:3] = np.clip(err / STEP * 1.5, -1, 1)
            a[6] = grip
            if log is not None:
                log.append((o["agentview_image"][::-1].copy(), o["robot0_eye_in_hand_image"][::-1].copy(),
                            np.r_[o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]], a.copy()))
            o, *_ = env.step(a)
            if frames is not None:
                frames.append(np.ascontiguousarray(env.sim.render(camera_name="frontview", width=320, height=240)[::-1]))
            if np.linalg.norm(err) < tol:
                break

    def hold(grip, steps):
        p = o["robot0_eef_pos"].copy()
        for _ in range(steps):
            goto(p, grip, max_steps=1, tol=0)

    # start above the first waypoint
    goto(np.r_[path[0], grasp_z + HOVER], -1, max_steps=60)
    max_cube_z = cube_z
    for k in range(len(path)):
        grip = 1.0 if gi <= k < ri else -1.0
        if k == gi:
            goto(np.r_[path[k], grasp_z], -1, max_steps=40, tol=0.004)
            hold(1.0, 15)
        if k == ri:
            goto(np.r_[path[k], grasp_z + 0.01], 1.0, max_steps=40, tol=0.005)
            hold(-1.0, 12)
        goto(np.r_[path[k], grasp_z + zoff[k]], grip)
        max_cube_z = max(max_cube_z, o["cube_pos"][2])
    hold(-1.0, 20)

    final = o["cube_pos"]
    place_err = float(np.linalg.norm(final[:2] - target_xy))
    lifted = bool(max_cube_z > cube_z + 0.03)
    success = bool(lifted and place_err < 0.04 and final[2] < cube_z + 0.02)
    return dict(success=success, lifted=lifted, place_err_cm=round(place_err * 100, 2),
                cube_start=cube_xy.round(3).tolist(), target=target_xy.round(3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--strategy", choices=["naive", "anchored"], default="anchored")
    ap.add_argument("--perturb", type=float, default=0.0, help="cm, random shift of cube & target")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--video", default=None)
    ap.add_argument("--save-rollout", default=None, help="dir: save successful (img,state,action) rollouts")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = dict(np.load(args.traj))
    env = make_env(cams=bool(args.save_rollout))
    res = []
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        frames = [] if (args.video and s == 0) else None
        log = [] if args.save_rollout else None
        r = run_episode(env, d, args.strategy, rng, args.perturb, frames, log)
        r["seed"] = s
        res.append(r)
        print(json.dumps(r))
        if frames:
            import imageio
            os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
            imageio.mimsave(args.video, frames, fps=20)
        if log and r["success"]:
            os.makedirs(args.save_rollout, exist_ok=True)
            name = os.path.splitext(os.path.basename(os.path.dirname(args.traj)))[0]
            np.savez_compressed(
                f"{args.save_rollout}/{name}_{args.strategy}_p{int(args.perturb)}_s{s}.npz",
                agentview=np.stack([x[0] for x in log]), wrist=np.stack([x[1] for x in log]),
                state=np.stack([x[2] for x in log]), action=np.stack([x[3] for x in log]),
                task="pick up the pyramid and place it next to the other marker")
    sr = np.mean([r["success"] for r in res])
    print(f"SUMMARY strategy={args.strategy} perturb={args.perturb}cm success={sr:.2f} n={len(res)}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()

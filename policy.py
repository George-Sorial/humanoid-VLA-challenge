"""Shared bits for the learned policy: observation features + layout sampling.

The policy is goal-conditioned and state-based: it sees the end-effector, the gripper,
the cube and the target, never which demo it came from.
"""
import numpy as np

FEAT_DIM = 12


def features(o, target_xy):
    eef = o["robot0_eef_pos"]
    grip = o["robot0_gripper_qpos"]
    cube = o["cube_pos"]
    return np.r_[eef, grip, cube - eef, target_xy - eef[:2], cube[:2] - target_xy].astype(np.float32)


def twin_layout(d):
    """Cube start / target in sim (m) mirroring the recorded table (see sim_replay.twin_map)."""
    pick, place = d["pick_cm"], d["place_cm"]
    mid = (pick + place) / 2.0
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    f = lambda p: 0.01 * (p - mid) @ R.T
    return f(pick), f(place)

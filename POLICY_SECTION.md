## Learned policy: behaviour cloning on my demos

The anchored replays are scripted: they are told where the cube and target are, and they follow
my recorded path. To get a **reactive policy**, I trained one on those replays and let it control
the robot by itself.

![learned policy on an unseen layout](out/policy/policy_success.gif)

![policy results](out/policy/policy_results.png)

| | naive replay | anchored replay | **learned BC policy** |
|---|---|---|---|
| control | open loop | open loop, scripted | **closed loop, reacts every step** |
| success, layouts shifted ≤ 6 cm | 0/20 | 20/20 | **15/40 (38%)** |

* **Data.** I ran anchored replays of all 20 demos, each on 6 random layouts (≤ 6 cm shift), giving
  117 successful episodes and about 40k steps. They use DART-style noise: the robot executes noisy
  actions while the label stays the clean one, so the data also covers recovering from small
  mistakes.
* **Model** (`train_policy.py`). A goal-conditioned MLP (3×256). Its input is 12-D state:
  end-effector position, gripper, cube relative to the end-effector, target relative to the
  end-effector, and cube relative to the target. Its output is a 4-D action (dx, dy, dz, gripper)
  in the same OSC action space as above. It trains in about 90 s on CPU. Whole demos (`L2R_10`,
  `R2L_10`) are held out for validation, where it reaches 98% gripper accuracy.
* **Evaluation** (`eval_policy.py`). Closed loop on 40 layouts drawn with **held-out random seeds**,
  450 steps (22 s) each. Episodes terminate on success, as in LIBERO. When it succeeds it places
  the cube with a **median error of 0.6 cm**, after about 10 s. Two independent 40-episode runs
  gave 15/40 and 16/40; robosuite randomises the robot's start pose, so runs differ slightly.
* **Reproduce:** `bash run_policy.sh` (about 30 min on 2 CPUs).

### What I learned from the failures
Almost every failure (22 of 25) is "never grasped". I traced individual episodes step by step:

1. **Gripper decision on the fence.** The replay starts closing the gripper only *after* arriving,
   so "at the cube, gripper open" is labelled both open and close. The regressor averaged them and
   sat at −0.08 forever, just below the close threshold. *Fix:* anticipatory gripper labels, where
   each open/close switch is labelled 0.25 s earlier. This cured those episodes, but the overall
   rate stayed the same (10/20 before and after on the same 20 layouts), because…
2. **Stalls from averaging multimodal demonstrations.** Near the cube my recorded paths wobble in
   different directions in different demos. A single-step regressor averages them into roughly zero
   motion, and the arm hovers 1–4 cm above the cube.
3. **No notion of "done".** After placing, the scene can look like a fresh pre-grasp state, and the
   policy sometimes picks the cube up again.

All three are known limits of single-step behaviour cloning, and they are exactly what
**action-chunking, generative policies** (ACT, Diffusion Policy, and SmolVLA's flow-matching action
expert) are designed to fix. So the natural next step is to train SmolVLA on image versions of the
same rollouts (`run_pipeline.py` without `--no-rollouts`). Its policy would also have to find the
cube from pixels instead of being given its position.

# Marker-anchored retargeting: my webcam demos → a simulated Panda

Intern challenge submission. I recorded myself moving a pyramid between two printed ArUco markers.
Each video becomes a camera-motion-free hand trajectory in a metric **table frame**. That trajectory
is retargeted to a Franka Panda in MuJoCo/robosuite, using OSC delta-pose + gripper actions (the
same 7-D action space as LIBERO). Successful replays are saved as a training dataset for a VLA.

**Run everything in Colab:** open `colab_pipeline.ipynb` (GPU runtime) and run all cells.
Locally, run `pip install -r requirements.txt && MUJOCO_GL=egl python run_pipeline.py`.

<!-- After the Colab run, paste out/sim/summary.md here and add out/sim/summary.png and a GIF:
![results](out/sim/summary.png)
![demo vs naive vs anchored](out/sim/gifs/L2R_1_compare.gif)
-->

## Data
There are 9 webcam clips in `data/` (1080p, about 16 fps, 3–9 s each). The camera looks down at a desk
with two printed 4×4 ArUco markers (IDs 0 and 1, 10 cm). Clip names give the direction **from my
viewpoint**. I sit opposite the camera, so `L2R` moves the pyramid right → left in the image.

| clips | direction | recorded |
|---|---|---|
| `L2R_1`, `L2R_2`, `R2L_1`, `R2L_2` | both | session 2 (markers about 35 cm apart) |
| `L2R_3` … `L2R_6`, `R2L_3` | both | session 1 (markers about 38 cm apart) |

## Idea
A moving camera makes raw hand pixels meaningless. Two flat markers give a per-frame homography
from image to table plane, so every frame becomes the same rectified top-down view in cm.

1. **Auto-calibration.** The left marker defines the table frame. The right marker's corners are
   measured in that frame using the left marker's own homography, taking the median over all
   frames. Nothing is measured by hand, and the markers can be rotated or offset.
2. **Object start/end.** I diff the first and last rectified frames and keep only object-sized
   blobs. The pick is the blob that is darker at the start, so both directions work.
3. **Hand path.** I take the foreground that differs from *both* the start and the end scene, which
   removes the static object "ghosts". The tip is the blob point farthest from where the arm enters.
4. **Contacts.** Grasp and release are the closest approaches of the robust-smoothed tip to the
   pick and place points.

The sim scene is a **digital twin** of the recorded table: the cube starts where my pyramid
started, and the target is where it ended. I compare two retargeting strategies:

* **naive** maps table cm → sim m with one fixed rigid transform and executes my path as is.
* **anchored** applies the same transform, then a 2-D similarity warp that pins my grasp point onto
  the cube and my release point onto the target. The path shape, timing and approach in between are
  still mine. The same warp re-targets a demo onto **new layouts**, as a MimicGen-style augmentation.
  `--perturb` tests this by shifting the cube and target by up to 3 or 6 cm.

The view is top-down and monocular, so height is not observable. It is synthesised by
**distance-gated descent**: the gripper hovers 12 cm up and descends as the path converges on the
next contact.

## Pipeline
| step | script | output |
|---|---|---|
| 1. video → table-frame trajectory and contacts | `track_hand.py` | `out/<clip>/traj.npz`, `rectified_montage.jpg` |
| 2. trajectory → Panda end-effector waypoints | `retarget.py` | `wp/<clip>_{naive,anchored}.npz` |
| 3. sim replay, scoring, rollouts | `sim_replay.py` | `out/sim/json/*.json`, `out/sim/videos/*.mp4`, `rollouts/*.npz` |
| 4. tables, figures, GIFs | `summarize.py`, `make_media.py` | `out/sim/summary.{md,png}`, `out/tracks.png`, `out/sim/gifs/` |
| all of the above | `run_pipeline.py` | |

```bash
python run_pipeline.py --quick --clips L2R_1     # smoke test (2 episodes)
python run_pipeline.py                           # all clips x {naive, anchored} x shift {0,3,6} cm
python run_pipeline.py --perturb 0 6 --seeds 2   # faster variant
```

Success means the cube was lifted more than 3 cm and then came to rest within 4 cm of the target.

Each episode in `rollouts/*.npz` holds the `agentview` and `wrist` images (128×128), `state`
(end-effector position, quaternion and gripper, 9-D), `action` (7-D) and the `task` instruction.

## Earlier validation run
I ran an earlier version on the first 5 clips (`L2R_3–6`, `R2L_3`). Anchored replay succeeded in
**45/45** episodes (layout shifts of 0, 3 and 6 cm). Naive replay succeeded **60% → 30% → 5%**,
because it is only as accurate as the hand tracking (1–8 cm error).

## What worked
* **Markers as a moving-camera anchor.** They were detected in practically every frame of every clip.
  Auto-calibration recovers the marker spacing to within about 1 cm of what I measured from pixels.
* **The object is the most reliable signal.** Pick and place come from the scene diff, not the hand.
* **Anchoring** makes tracking noise irrelevant at the contacts, while keeping the human's path and timing.

## What didn't (and what I changed)
* **MediaPipe.** The current MediaPipe has dropped its legacy hand API, and its model download was
  blocked in my build environment. I replaced it with the marker-rectified background-subtraction
  tracker above. It has no learned model, but it gives no pinch aperture and no height.
* **The printed markers are rotated 180° in the image.** This produced a distorted homography until
  marker orientation was measured. Auto-calibration now handles any rotation.
* **Savitzky-Golay smoothing over outlier-prone tracks shifted contacts by up to 11 cm.** I replaced
  it with a rolling median over valid samples, then interpolation, then light smoothing, and detect
  contacts on the smoothed track.
* **Shadows and lighting changes** were mistaken for the object, so blobs must now be object-sized.
* **Fast, motion-blurred clips** (`R2L_1/2` are about 3 s long) give fewer hand frames and larger
  contact errors. These are exactly the cases where naive replay fails and anchoring helps.
* **LIBERO itself** pins old numpy/robosuite versions that would not build in my environment
  (Python 3.13). I used robosuite 1.5, the simulator LIBERO is built on, with the same Panda, OSC
  controller and action convention. Porting means swapping `make_env` and the object/target poses.
* **Honest caveat.** Anchored success is partly by construction, because the sim knows the cube and
  target poses. What it shows is that my *recorded path shape, timing and gripper schedule* turn into
  feasible robot motion. A policy trained on the rollouts must still find the cube from pixels.

## Next: SmolVLA
Convert `rollouts/` to a LeRobot dataset and fine-tune `lerobot/smolvla_base`. Then evaluate base vs
fine-tuned in the same scene at layout shifts of 0, 3 and 6 cm.

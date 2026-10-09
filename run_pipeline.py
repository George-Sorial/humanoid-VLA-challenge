"""One command: every video in data/ -> tracking -> retargeting -> sim replay -> results.

  python run_pipeline.py                       # full run (all clips, perturb 0/3/6 cm, 3 seeds)
  python run_pipeline.py --quick               # smoke test: perturb 0 only, 1 seed
  python run_pipeline.py --clips L2R_1 R2L_1   # subset

Outputs
  out/<clip>/traj.npz, rectified_montage.jpg   tracking
  wp/<clip>_{naive,anchored}.npz               retargeted end-effector waypoints
  out/sim/json/<clip>__<strategy>__p<cm>.json  per-episode sim results
  out/sim/videos/<clip>_<strategy>.mp4         replay videos (unperturbed layout)
  out/sim/summary.{png,md}, out/tracks.png, out/sim/gifs/<clip>_compare.gif
  rollouts/*.npz                               successful anchored episodes (VLA training data)
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable


def run(cmd, log=None):
    with open(log, "w") if log else open(os.devnull, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--clips", nargs="*", default=None, help="clip names (default: all *.mp4 in data/)")
    ap.add_argument("--perturb", nargs="*", type=float, default=[0, 3, 6], help="layout shifts (cm)")
    ap.add_argument("--seeds", type=int, default=3, help="random layouts per perturbation > 0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--marker-size", type=float, default=10.0)
    ap.add_argument("--no-rollouts", action="store_true", help="don't save VLA training rollouts")
    ap.add_argument("--quick", action="store_true", help="smoke test: perturb 0, 1 seed")
    ap.add_argument("--skip-track", action="store_true", help="reuse existing out/<clip>/traj.npz")
    args = ap.parse_args()
    if args.quick:
        args.perturb, args.seeds = [0], 1
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

    clips = args.clips or sorted(os.path.splitext(os.path.basename(p))[0]
                                 for p in glob.glob(f"{args.data}/*.mp4"))
    print(f"clips ({len(clips)}): {' '.join(clips)}")
    t0 = time.time()

    # 1. tracking ---------------------------------------------------------------------------
    if not args.skip_track:
        print("\n[1/4] tracking hand + object (marker-rectified table frame)")
        good = []
        for c in clips:
            rc = subprocess.run([PY, "track_hand.py", f"{args.data}/{c}.mp4", f"out/{c}/traj.npz",
                                 "--marker-size", str(args.marker_size), "--debug-dir", f"out/{c}"],
                                capture_output=True, text=True)
            line = (rc.stdout.strip().splitlines() or [""])[-1]
            print("  " + (line if rc.returncode == 0 else f"{c}: FAILED {rc.stdout[-300:]}{rc.stderr[-300:]}"))
            if rc.returncode == 0:
                good.append(c)
        clips = good

    # 2. retarget (standalone waypoints, for inspection) --------------------------------------
    print("\n[2/4] retargeting to Panda end-effector waypoints")
    os.makedirs("wp", exist_ok=True)
    for c in clips:
        for s in ["naive", "anchored"]:
            run([PY, "retarget.py", f"out/{c}/traj.npz", f"wp/{c}_{s}.npz", "--strategy", s,
                 "--pick-xy", "-0.15", "0", "--place-xy", "0.15", "0", "--z-table", "0.90", "--z-lift", "1.00"])
    print(f"  wrote wp/<clip>_{{naive,anchored}}.npz for {len(clips)} clips")

    # 3. sim replay ---------------------------------------------------------------------------
    print(f"\n[3/4] sim replay: {len(clips)} clips x 2 strategies x perturb {args.perturb} "
          f"(seeds: 1 at 0 cm, {args.seeds} otherwise), {args.workers} workers")
    for d in ["out/sim/json", "out/sim/videos", "out/sim/logs"]:
        os.makedirs(d, exist_ok=True)
    jobs = []
    for c in clips:
        for s in ["anchored", "naive"]:
            for p in args.perturb:
                n = 1 if p == 0 else args.seeds
                tag = f"{c}__{s}__p{int(p)}"
                cmd = [PY, "sim_replay.py", f"out/{c}/traj.npz", "--strategy", s, "--perturb", str(p),
                       "--seeds", str(n), "--json", f"out/sim/json/{tag}.json"]
                if p == 0:
                    cmd += ["--video", f"out/sim/videos/{c}_{s}.mp4"]
                if s == "anchored" and not args.no_rollouts:
                    cmd += ["--save-rollout", "rollouts"]
                jobs.append((tag, cmd))
    done = [0]

    def work(job):
        tag, cmd = job
        rc = run(cmd, f"out/sim/logs/{tag}.log")
        done[0] += 1
        print(f"  [{done[0]}/{len(jobs)}] {tag} {'ok' if rc == 0 else 'FAILED (see out/sim/logs)'}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))

    # 4. results ------------------------------------------------------------------------------
    print("\n[4/4] summarising + media")
    subprocess.run([PY, "summarize.py"])
    subprocess.run([PY, "make_media.py", "--data", args.data])
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> out/sim/summary.md, out/sim/summary.png, out/sim/gifs/")


if __name__ == "__main__":
    main()

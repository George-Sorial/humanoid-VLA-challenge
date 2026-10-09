"""Generates colab_pipeline.ipynb (kept as a script so the notebook is reproducible)."""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": s.strip("\n")})

md("""
# Webcam demos → simulated Panda (Colab)

Runs the whole pipeline on the clips in `data/`:
**marker-rectified hand tracking → retargeting (naive vs anchored) → MuJoCo/robosuite Panda replay
with layout perturbations → results + VLA training rollouts.**

**Use a GPU runtime** (Runtime → Change runtime type → T4): MuJoCo then renders headless via EGL, which is the\ntested path. The physics itself runs on CPU (≈ 30–60 s per episode per core). A CPU-only runtime falls back to\nOSMesa software rendering (untested).
""")
code("""
!git clone -q https://github.com/George-Sorial/humanoid-VLA-challenge.git 2>/dev/null || (cd humanoid-VLA-challenge && git pull -q)
%cd humanoid-VLA-challenge
# Colab's own Python may be too new for MuJoCo 3.3.x + robosuite 1.5.2 wheels (pip would try to build MuJoCo
# from source). So the pipeline runs in its own Python 3.12 env, created with uv in ~1 minute.
!pip -q install uv
!uv venv -q -p 3.12 /content/venv
!VIRTUAL_ENV=/content/venv uv pip install -q -r requirements.txt
PY = "/content/venv/bin/python"
!{PY} -c "import sys, mujoco, robosuite, cv2; print('python', sys.version.split()[0], '| mujoco', mujoco.__version__, '| robosuite', robosuite.__version__, '| aruco', hasattr(cv2, 'aruco'))"
""")
code("""
# headless rendering backend: EGL on a GPU runtime, OSMesa (software) on CPU
import os, subprocess
gpu = subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode == 0
if not gpu:
    subprocess.run("apt-get -qq install -y libosmesa6-dev > /dev/null", shell=True)
os.environ["MUJOCO_GL"] = "egl" if gpu else "osmesa"
os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
print("GPU:", gpu, "| MUJOCO_GL =", os.environ["MUJOCO_GL"], "| CPUs:", os.cpu_count())
""")
md("""
### (Optional) add your own clips
Two flat 4×4 ArUco markers (IDs 0 and 1, 10 cm) on the table, camera looking down, object
fully visible in the first and last frame. Marker spacing is calibrated automatically.
""")
code("""
UPLOAD = False   # set True to upload extra .mp4 clips into data/
if UPLOAD:
    from google.colab import files
    for name, data in files.upload().items():
        open(f"data/{name}", "wb").write(data)
!ls data
""")
md("### 1. Smoke test (one clip, unperturbed, ~2 min)")
code("""
!{PY} run_pipeline.py --quick --clips L2R_1 --workers 2 --no-rollouts
""")
md("""
### 2. Full experiment
All clips × {naive, anchored} × layout shift {0, 3, 6} cm, one random layout per shift.
With 20 clips that is 120 episodes (~25 min on 2 CPUs). `--no-rollouts` skips saving VLA training data
(much faster); drop it, and raise `--seeds`, for a bigger run.
""")
code("""
!{PY} run_pipeline.py --perturb 0 3 6 --seeds 1 --no-rollouts --workers {os.cpu_count()}
""")
md("### 3. Results")
code("""
from IPython.display import Markdown, Image, display
display(Markdown(open("out/sim/summary.md").read()))
display(Image("out/sim/summary.png"))
display(Image("out/tracks.png"))
""")
code("""
import glob
for g in sorted(glob.glob("out/sim/gifs/*_compare.gif"))[:3]:
    print(g); display(Image(g))
""")
md("### 4. VLA training data (successful anchored rollouts)")
code("""
# only if you ran without --no-rollouts
import numpy as np, glob
fs = sorted(glob.glob("rollouts/*.npz"))
assert fs, "no rollouts: re-run run_pipeline.py without --no-rollouts"
print(len(fs), "episodes")
d = np.load(fs[0])
print({k: d[k].shape for k in ["agentview", "wrist", "state", "action"]}, "| task:", str(d["task"]))
import matplotlib.pyplot as plt
idx = np.linspace(0, len(d["action"]) - 1, 6).astype(int)
fig, ax = plt.subplots(2, 6, figsize=(15, 5))
for j, i in enumerate(idx):
    ax[0, j].imshow(d["agentview"][i]); ax[1, j].imshow(d["wrist"][i])
    ax[0, j].set_title(f"t={i}"); ax[0, j].axis("off"); ax[1, j].axis("off")
plt.show()
""")
code("""
# download everything except the (large) rollouts
!zip -qr results.zip out wp
from google.colab import files; files.download("results.zip")
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("colab_pipeline.ipynb", "w"), indent=1)
print("wrote colab_pipeline.ipynb")

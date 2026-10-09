#!/usr/bin/env bash
# Learned policy on top of the retargeted demos (needs out/<clip>/traj.npz from run_pipeline.py).
#   1. data: anchored replays of every demo on 6 random layouts (<= 6 cm shift) with DART-style
#      action noise (robot executes noisy actions, labels stay clean) -> states/<clip>_s<k>.npz
#   2. train: goal-conditioned MLP behaviour cloning, demos L2R_10 + R2L_10 held out for validation
#   3. eval: closed loop on 40 layouts from held-out random seeds (terminate on success, LIBERO-style)
set -euo pipefail
PY=${PY:-python}
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
W=${WORKERS:-2}

ls out/*/traj.npz | xargs -P "$W" -I{} $PY sim_replay.py {} --strategy anchored --perturb 6 --seeds 6 \
    --save-states states --action-noise 0.1 > /dev/null
$PY train_policy.py --data states --out out/policy --epochs 80
for i in $(seq 0 $((W - 1))); do
  $PY eval_policy.py --model out/policy/policy.joblib --episodes 40 --perturb 6 --tag p6 \
      --shard "$i" "$W" --videos $([ "$i" = 0 ] && echo 3 || echo 0) &
done
wait

#!/usr/bin/env bash
# Start training in the background with nohup (keeps running after you log out).
#   bash run.sh                          # all GPUs
#   bash run.sh --resume                 # continue from the last checkpoint
#   bash run.sh --set train.epochs=1     # any train.py arguments
#   bash run.sh --config config_auto.yaml   # continue the finished model with the auto-source prompt
# Log: <output.dir>/logs/train_<time>.log  (latest: <output.dir>/logs/train.log)
# Stop: pkill -f "<this folder>/train.py"   (printed below; doesn't touch ttt/riva_train or stt runs)
# Refuses to start while another multilingual training run is still alive.
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="$PWD/train.py"
MATCH="(python|torchrun).* $SCRIPT"

RUNNING=$(pgrep -f "$MATCH" || true)
if [ -n "$RUNNING" ]; then
  echo "Training is already running (pid: $(echo $RUNNING | tr '\n' ' ')). Not starting another one."
  echo "Stop it first:  pkill -f '$MATCH'"
  exit 1
fi

# output.dir as train.py will see it (--config and --set output.dir=... included)
OUT=$(python - "$@" <<'PY'
import argparse, sys
sys.path.insert(0, "../..")
from general.config import load_config
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--set", nargs="*", default=[])
a, _ = ap.parse_known_args()
print(load_config(a.config, a.set)["output"]["dir"])
PY
)
mkdir -p "$OUT/logs"
LOG="$OUT/logs/train_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "$OUT/logs/train.log"

NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l)
if [ "$NGPU" -gt 1 ]; then
  CMD=(torchrun --nproc_per_node="$NGPU" "$SCRIPT" "$@")
else
  CMD=(python "$SCRIPT" "$@")
fi

PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"
echo "Started on $NGPU GPU(s), pid $(cat "$OUT/train.pid"): ${CMD[*]}"
echo "Log:   tail -f $OUT/logs/train.log"
echo "Stop:  pkill -f '$MATCH'"

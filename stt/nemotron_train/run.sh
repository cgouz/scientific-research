#!/usr/bin/env bash
# Start training in the background with nohup (keeps running after you log out).
#   bash run.sh                          # all GPUs (train.gpus in config.yaml)
#   bash run.sh --resume                 # continue from the last checkpoint
#   bash run.sh --set train.epochs=1     # any train.py arguments
# Log: <output.dir>/logs/train_<time>.log  (latest: <output.dir>/logs/train.log)
# Stop: pkill -f "python.* <this folder>/train.py"   (printed below; doesn't touch ttt/riva_train runs)
# Refuses to start while another Nemotron training run is still alive.
# Lightning starts one process per GPU itself, so this runs plain python, not torchrun.
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="$PWD/train.py"
MATCH="python.* $SCRIPT"

RUNNING=$(pgrep -f "$MATCH" || true)
if [ -n "$RUNNING" ]; then
  echo "Training is already running (pid: $(echo $RUNNING | tr '\n' ' ')). Not starting another one."
  echo "Stop it first:  pkill -f '$MATCH'"
  exit 1
fi

OUT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['output']['dir'])")
mkdir -p "$OUT/logs"
LOG="$OUT/logs/train_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "$OUT/logs/train.log"

PYTHONUNBUFFERED=1 nohup python "$SCRIPT" "$@" > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"
echo "Started, pid $(cat "$OUT/train.pid"): python $SCRIPT $*"
echo "Log:   tail -f $OUT/logs/train.log"
echo "Stop:  pkill -f '$MATCH'"

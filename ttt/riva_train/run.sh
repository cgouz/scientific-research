#!/usr/bin/env bash
# Start training in the background with nohup (keeps running after you log out).
#   bash run.sh                          # all GPUs
#   bash run.sh --resume                 # continue from the last checkpoint
#   bash run.sh --set train.epochs=1     # any train.py arguments
# Log: <output.dir>/logs/train_<time>.log  (latest: <output.dir>/logs/train.log)
# Stop: pkill -f "python.* train.py|torchrun.* train.py"
# Refuses to start while another training run is still alive.
set -euo pipefail
cd "$(dirname "$0")"

RUNNING=$(pgrep -f "python.* train.py|torchrun.* train.py" || true)
if [ -n "$RUNNING" ]; then
  echo "Training is already running (pid: $(echo $RUNNING | tr '\n' ' ')). Not starting another one."
  echo "Stop it first:  pkill -f 'python.* train.py|torchrun.* train.py'"
  exit 1
fi

OUT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['output']['dir'])")
mkdir -p "$OUT/logs"
LOG="$OUT/logs/train_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "$OUT/logs/train.log"

NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l)
if [ "$NGPU" -gt 1 ]; then
  CMD=(torchrun --nproc_per_node="$NGPU" train.py "$@")
else
  CMD=(python train.py "$@")
fi

PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"
echo "Started on $NGPU GPU(s), pid $(cat "$OUT/train.pid"): ${CMD[*]}"
echo "Log:   tail -f $OUT/logs/train.log"
echo "Stop:  pkill -f 'python.* train.py|torchrun.* train.py'"

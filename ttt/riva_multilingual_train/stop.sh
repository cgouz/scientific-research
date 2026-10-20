#!/usr/bin/env bash
# Stop a training run of this folder's train.py (started by run.sh, torchrun or python), with its dataloader workers.
#   bash stop.sh          # asks first; SIGTERM, then SIGKILL after 30 s if something is still alive
#   bash stop.sh -y       # don't ask
# Doesn't touch ttt/riva_train or stt runs.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd -P)"

# "<pid> <command>" of every python/torchrun process running this train.py: by full path, or a plain
# "train.py" started in this folder (an editor, tail or shell with the path in its command line is not matched)
list() {
  (pgrep -af "train.py" || true) | while read -r pid cmd; do
    exe="${cmd%% *}"
    case "${exe##*/}" in python*|torchrun) ;; *) continue ;; esac
    case "$cmd" in
      *riva_multilingual_train/train.py*) echo "$pid $cmd" ;;
      *" train.py"*) if [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$HERE" ]; then echo "$pid $cmd"; fi ;;
    esac
  done
}

PROCS=$(list)
if [ -z "$PROCS" ]; then
  echo "No riva_multilingual training is running."
  exit 0
fi
echo "Running ($(echo "$PROCS" | wc -l) processes: the main one + dataloader workers):"
echo "$PROCS" | cut -c1-150

if [ "${1:-}" != "-y" ]; then
  read -r -p "Stop them? [y/N] " ok
  if [ "$ok" != "y" ] && [ "$ok" != "Y" ]; then echo "Nothing stopped."; exit 0; fi
fi

kill $(echo "$PROCS" | awk '{print $1}') 2>/dev/null || true
echo -n "Stopping"
for _ in $(seq 30); do
  [ -z "$(list)" ] && break
  echo -n "."
  sleep 1
done
echo

LEFT=$(list | awk '{print $1}')
if [ -n "$LEFT" ]; then
  echo "Still alive after 30 s, killing: $(echo $LEFT | tr '\n' ' ')"
  kill -9 $LEFT 2>/dev/null || true
  sleep 3
fi

OUT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['output']['dir'])" 2>/dev/null || true)
if [ -n "$OUT" ]; then rm -f "$OUT/train.pid"; fi

if [ -n "$(list)" ]; then
  echo "Could not stop:"; list; exit 1
fi
echo "Stopped."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
echo "Start again:  bash run.sh            (fresh run)"
echo "              bash run.sh --resume   (continue from the latest checkpoint-*/)"

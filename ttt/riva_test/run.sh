#!/usr/bin/env bash
# Run riva_test.py in the background with nohup (keeps running after you log out).
#   bash run.sh --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json
#   bash run.sh --model /data/experiments/riva/final --test-file ... --out finetuned.json
# Results: /data/experiments/riva/results/   Log: /data/experiments/riva/logs/test_<time>.log
set -euo pipefail
cd "$(dirname "$0")"

LOGS=/data/experiments/riva/logs
mkdir -p "$LOGS"
LOG="$LOGS/test_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "$LOGS/test.log"

PYTHONUNBUFFERED=1 nohup python riva_test.py "$@" > "$LOG" 2>&1 &
echo "Started, pid $!: python riva_test.py $*"
echo "Log:   tail -f $LOGS/test.log"

#!/usr/bin/env bash
# Run nemotron_test.py in the background with nohup (keeps running after you log out).
#   bash run.sh --bench --split test --out finetuned.json
#   bash run.sh --model nvidia/nemotron-3.5-asr-streaming-0.6b --bench --split test --out base.json
# Results: /data/experiments/nemotron/results/   Log: /data/experiments/nemotron/logs/test_<time>.log
set -euo pipefail
cd "$(dirname "$0")"

LOGS=/data/experiments/nemotron/logs
mkdir -p "$LOGS"
LOG="$LOGS/test_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" "$LOGS/test.log"

PYTHONUNBUFFERED=1 nohup python nemotron_test.py "$@" > "$LOG" 2>&1 &
echo "Started, pid $!: python nemotron_test.py $*"
echo "Log:   tail -f $LOGS/test.log"

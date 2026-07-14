#!/usr/bin/env bash
# wait_log.sh <logfile> <target_step>  -- poll until target step val, run end, or error
log="$1"; target="${2:-700}"; i=0
until grep -qaE "step ${target}  val_minus=|^BEST:|Traceback|RuntimeError|Killed|CUDA error" "$log" 2>/dev/null; do
  sleep 10; i=$((i+1)); [ $i -gt 300 ] && { echo "TIMEOUT after ~$((i*10))s"; break; }
done
echo "=== $log val trajectory ==="
grep -aE "step [0-9]+  val_minus=" "$log"
echo "=== end/errors ==="
grep -aE "^BEST:|^CONFIG:|Traceback|RuntimeError|Killed|CUDA error|Error" "$log" | tail -5

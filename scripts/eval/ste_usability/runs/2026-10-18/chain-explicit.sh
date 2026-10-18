#!/bin/bash
# Local-model runs for 2026-10-18, explicit routes only. The device is shared:
# run_local.py writes <id>.SWAP.txt whenever another chat model was launched
# during a call, and score.py's caller must exclude or flag those programs.
set -u
cd /home/teague/Projects/lackpy
R=scripts/eval/ste_usability/runs/2026-10-18
PY=.venv/bin/python
Q30=tiiny-fast/Qwen/Qwen3-30B-A3B-Instruct
Q36=tiiny-fast/Qwen/Qwen3.6-35B-A3B
run() { echo "== $1 $2 ($3) variant=${PROMPT_VARIANT:-guide} $(date -Is)"; $PY scripts/eval/ste_usability/run_local.py $R "$1" "$2" "$3"; }
unset PROMPT_VARIANT
run qwen3-30b r1 $Q30; run qwen3-30b r2 $Q30
export PROMPT_VARIANT=pre
run qwen3-30b-pre r1 $Q30; run qwen3-30b-pre r2 $Q30
unset PROMPT_VARIANT
run qwen3.6-35b r1 $Q36; run qwen3.6-35b r2 $Q36
export PROMPT_VARIANT=pre
run qwen3.6-35b-pre r1 $Q36; run qwen3.6-35b-pre r2 $Q36
echo "== restore 30B-Instruct (the device default per ttt) $(date -Is)"
timeout 600 ttt load fast 2>&1 | tail -2; timeout 60 ttt models 2>&1 | tail -6
echo "== swap files:"; find $R -name '*.SWAP.txt' | sort
echo "== done $(date -Is)"

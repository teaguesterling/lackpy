#!/bin/bash
# Qwen3.6-35B-A3B in instruct mode (THINKING=off), explicit route, both prompt variants.
set -u
cd /home/teague/Projects/lackpy
R=scripts/eval/ste_usability/runs/2026-10-18
PY=.venv/bin/python
Q36=tiiny-fast/Qwen/Qwen3.6-35B-A3B
export THINKING=off
run() { echo "== $1 $2 ($3) variant=${PROMPT_VARIANT:-guide} thinking=off $(date -Is)"; $PY scripts/eval/ste_usability/run_local.py $R "$1" "$2" "$3"; }
unset PROMPT_VARIANT
run qwen3.6-35b r1 $Q36; run qwen3.6-35b r2 $Q36
export PROMPT_VARIANT=pre
run qwen3.6-35b-pre r1 $Q36; run qwen3.6-35b-pre r2 $Q36
echo "== restore 30B-Instruct $(date -Is)"
timeout 600 ttt load fast 2>&1 | tail -2; timeout 60 ttt models 2>&1 | tail -6
echo "== swap files:"; find $R -name '*.SWAP.txt' | sort
echo "== empty replies:"; find $R -name '*.reply.txt' -size 0 | sort
echo "== done $(date -Is)"

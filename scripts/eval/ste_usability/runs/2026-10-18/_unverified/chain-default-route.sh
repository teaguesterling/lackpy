#!/bin/bash
# Local-model chain for the 2026-10-18 run. Each step logs what it did.
set -u
cd /home/teague/Projects/lackpy
R=scripts/eval/ste_usability/runs/2026-10-18
PY=.venv/bin/python
echo "== qwen3-30b r2 (route: ttt default, inferred Qwen3-30B-A3B-Instruct) $(date -Is)"
$PY scripts/eval/ste_usability/run_local.py $R qwen3-30b r2
echo "== load Qwen/Qwen3.6-35B-A3B $(date -Is)"
timeout 600 ttt load Qwen/Qwen3.6-35B-A3B 2>&1 | tail -3
timeout 60 ttt models 2>&1 | tail -6
echo "== qwen3.6-35b r1 (route: tiiny-fast/Qwen/Qwen3.6-35B-A3B) $(date -Is)"
$PY scripts/eval/ste_usability/run_local.py $R qwen3.6-35b r1 tiiny-fast/Qwen/Qwen3.6-35B-A3B
echo "== qwen3.6-35b r2 $(date -Is)"
$PY scripts/eval/ste_usability/run_local.py $R qwen3.6-35b r2 tiiny-fast/Qwen/Qwen3.6-35B-A3B
echo "== restore device default (ttt load fast) $(date -Is)"
timeout 600 ttt load fast 2>&1 | tail -3
timeout 60 ttt models 2>&1 | tail -6
echo "== done $(date -Is)"

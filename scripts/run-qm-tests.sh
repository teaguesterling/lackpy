#!/usr/bin/env bash
# Run all quartermaster experiments in batch.
# Results saved to results/ directory.
set -euo pipefail

HOST="${OLLAMA_HOST:-http://localhost:11435}"
TIMEOUT="${QM_TIMEOUT:-60}"
SCRIPT="$(dirname "$0")/pluckit-quartermaster.py"
PYTHON="${PYTHON:-$(dirname "$0")/../../venv/bin/python}"
OUTDIR="$(dirname "$0")/../results"
COMPARE="$(dirname "$0")/pluckit-model-compare.py"

mkdir -p "$OUTDIR"

echo "Host: $HOST"
echo "Timeout: ${TIMEOUT}s per call"
echo "Output: $OUTDIR/"
echo ""

# ── Phase 1: Model comparison (all models, no QM) ──

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Model comparison (raw, no quartermaster)      ║"
echo "╚══════════════════════════════════════════════════════════╝"
"$PYTHON" "$COMPARE" --host "$HOST" --timeout "$TIMEOUT" \
    --models qwen2.5-coder:0.5b,qwen2.5-coder:1.5b,qwen2.5-coder:3b,qwen2.5-coder:7b,qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b,qwen3:0.6b,smollm2:latest,phi4-mini:latest,llama3.2:1b,llama3.2:latest,granite-code:3b \
    --output "$OUTDIR/model-compare.json" \
    2>&1 | tee "$OUTDIR/model-compare.log"
echo ""

# ── Phase 2: Rule-based QM with top assemblers ──

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Rule-based QM → assembler (with baseline)     ║"
echo "╚══════════════════════════════════════════════════════════╝"

for asm in qwen2.5:1.5b qwen2.5:3b qwen2.5:7b qwen2.5-coder:1.5b qwen2.5-coder:3b qwen2.5-coder:7b smollm2:latest phi4-mini:latest granite-code:3b; do
    slug="${asm//[:.]/-}"
    echo "━━━ rules → $asm ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode rules --asm-model "$asm" --baseline \
        --output "$OUTDIR/qm-rules-${slug}.json" \
        2>&1 | tee "$OUTDIR/qm-rules-${slug}.log"
    echo ""
done

# ── Phase 3: Model-based QM (small QM → medium assembler) ──

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: Model-based QM → assembler                    ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Small QMs → 7B assembler
for qm in qwen2.5-coder:0.5b qwen2.5-coder:1.5b qwen2.5:1.5b llama3.2:1b; do
    slug="${qm//[:.]/-}"
    echo "━━━ model($qm) → qwen2.5:7b ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode model --qm-model "$qm" --asm-model qwen2.5:7b --baseline \
        --output "$OUTDIR/qm-model-${slug}-to-7b.json" \
        2>&1 | tee "$OUTDIR/qm-model-${slug}-to-7b.log"
    echo ""
done

# Small QMs → 3B assembler
for qm in qwen2.5-coder:0.5b qwen2.5:1.5b; do
    slug="${qm//[:.]/-}"
    echo "━━━ model($qm) → qwen2.5:3b ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode model --qm-model "$qm" --asm-model qwen2.5:3b --baseline \
        --output "$OUTDIR/qm-model-${slug}-to-3b.json" \
        2>&1 | tee "$OUTDIR/qm-model-${slug}-to-3b.log"
    echo ""
done

# ── Summary ──

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL RUNS COMPLETE                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
ls -lh "$OUTDIR"/qm-*.json "$OUTDIR"/model-compare.json 2>/dev/null | awk '{printf "  %-50s %s\n", $NF, $5}'
echo ""
echo "Analyze with:"
echo "  python3 -c \"import json; ...\""

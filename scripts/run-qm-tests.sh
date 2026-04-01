#!/usr/bin/env bash
# Run all quartermaster experiments in batch.
# Results saved to results/ directory.
set -euo pipefail

HOST="${OLLAMA_HOST:-http://localhost:11435}"
TIMEOUT="${QM_TIMEOUT:-60}"
SCRIPT="$(dirname "$0")/pluckit-quartermaster.py"
PYTHON="${PYTHON:-$(dirname "$0")/../../venv/bin/python}"
OUTDIR="$(dirname "$0")/../results"

mkdir -p "$OUTDIR"

echo "Host: $HOST"
echo "Timeout: ${TIMEOUT}s per call"
echo "Output: $OUTDIR/"
echo ""

# ── Rule-based QM with different assemblers ──

for asm in qwen2.5:1.5b qwen2.5-coder:1.5b smollm2:latest phi4-mini:latest llama3.2:latest llama3.2:1b; do
    slug="${asm//[:.]/-}"
    echo "━━━ rules → $asm ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode rules --asm-model "$asm" --baseline \
        --output "$OUTDIR/qm-rules-${slug}.json" \
        2>&1 | tee "$OUTDIR/qm-rules-${slug}.log"
    echo ""
done

# ── Model-based QM (improved few-shot) with qwen2.5:1.5b assembler ──

for qm in qwen2.5-coder:0.5b qwen2.5:1.5b; do
    slug="${qm//[:.]/-}"
    echo "━━━ model($qm) → qwen2.5:1.5b ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode model --qm-model "$qm" --asm-model qwen2.5:1.5b --baseline \
        --output "$OUTDIR/qm-model-${slug}.json" \
        2>&1 | tee "$OUTDIR/qm-model-${slug}.log"
    echo ""
done

# ── No QM (pure baseline) with top assemblers ──

for asm in qwen2.5:1.5b smollm2:latest; do
    slug="${asm//[:.]/-}"
    echo "━━━ none → $asm ━━━"
    "$PYTHON" "$SCRIPT" --host "$HOST" --timeout "$TIMEOUT" \
        --qm-mode none --asm-model "$asm" \
        --output "$OUTDIR/qm-none-${slug}.json" \
        2>&1 | tee "$OUTDIR/qm-none-${slug}.log"
    echo ""
done

# ── Summary ──

echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL RUNS COMPLETE. Results in $OUTDIR/"
echo "════════════════════════════════════════════════════════════"
ls -la "$OUTDIR"/qm-*.json 2>/dev/null | awk '{print $NF, $5}'

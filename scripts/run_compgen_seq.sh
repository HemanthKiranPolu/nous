#!/usr/bin/env bash
# SCAN-mini compositional-generalization sweep.
#   ./scripts/run_compgen_seq.sh                 # full 32-seed sweep → results/scan_mini_32s.json
#   ./scripts/run_compgen_seq.sh --quick         # 4-seed CI smoke → results/scan_mini_quick.json
#   ./scripts/run_compgen_seq.sh --quick --ci-gate 0   # smoke + tripwire: fail if NOUS < best baseline
# Extra args after --quick are forwarded to the module (e.g. --ci-gate N).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--quick" ]]; then
    shift
    # Small but not flaky: a Δ≥25pp *claim* needs the full 32-seed sweep; the
    # quick run is a regression tripwire, so the default gate is lenient (≥0).
    exec "$PY" -m nous.train_compgen_seq --seeds 0 1 2 3 4 --epochs 200 \
        --out results/scan_mini_quick.json "$@"
fi

exec "$PY" -m nous.train_compgen_seq --seeds $(seq 0 31) --epochs 200 \
    --out results/scan_mini_32s.json "$@"

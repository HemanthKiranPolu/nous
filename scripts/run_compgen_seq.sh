#!/usr/bin/env bash
# SCAN-mini compositional-generalization sweep.
#   ./scripts/run_compgen_seq.sh            # full 32-seed sweep → results/scan_mini_32s.json
#   ./scripts/run_compgen_seq.sh --quick    # 2-seed / 40-epoch CI sanity (no claim)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--quick" ]]; then
    exec "$PY" -m nous.train_compgen_seq --seeds 0 1 --epochs 40 \
        --out results/scan_mini_quick.json
fi

exec "$PY" -m nous.train_compgen_seq --seeds $(seq 0 31) --epochs 200 \
    --out results/scan_mini_32s.json

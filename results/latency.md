# Inference latency — NOUS vs Transformer (SCAN-mini)

Reproduce: `python -m scripts.profile_latency --queries 200 --out results/latency.csv`

Single-thread CPU, freshly-initialised models (latency is architecture, not
training), identical `SeqDecoder` for both. `n_steps=60`, `dt=0.1`.

| model              | early-exit | params | p50 (ms) | p95 (ms) |
| ------------------ | ---------- | ------ | -------- | -------- |
| NOUS               | on         | 22,918 | 8.50     | 8.68     |
| NOUS               | off        | 22,918 | 8.50     | 9.80     |
| Transformer        | n/a        | 19,989 | 0.145    | 0.176    |

## Honest interpretation

- **NOUS is ~58× slower per query** than the param-comparable transformer. The
  cost is the iterative Euler–Lagrange relaxation: ~60 ODE steps, each an
  autograd force evaluation. This is the real, expected EBM latency tax.
- **Early-exit helps only the tail.** It shaves p95 from 9.8 → 8.7 ms (~11 %)
  but leaves p50 unchanged — on this task the equilibrium seldom reaches the
  `delta` force-norm threshold before `n_steps`, so most queries run the full
  budget. Early-exit is not a fix for the constant-factor gap.
- **Absolute bar:** at 8.7 ms p95 NOUS clears a 45 ms latency gate comfortably;
  the concern is relative cost, not absolute, at this scale.

## Caveat

This is the SCAN-mini decode (tiny state, 8 inputs). Full-SCAN autoregressive
decode multiplies NOUS cost by output length (one relaxation per token), so the
gap widens. The generalization upside (see `README.md`) is what must justify
this latency — quantified here so the trade-off is explicit, not hidden.

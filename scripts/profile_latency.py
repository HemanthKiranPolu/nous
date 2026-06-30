"""
Inference-latency profile: NOUS vs Transformer on the SCAN-mini decode.

Pre-empts the "energy-based models are slow" critique with numbers. Latency is
an architecture property (independent of training), so we time inference on
freshly-initialised models reused from nous.train_compgen_seq — the same
SeqDecoder for both, so the comparison is apples-to-apples.

NOUS is profiled with early-exit ON (solver stops once the force norm drops
below `delta`) and OFF (full n_steps), quantifying what early-exit buys.

  python -m scripts.profile_latency --queries 200 --out results/latency.csv
"""

import argparse
import csv
import os
import time

import torch

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.train_compgen_seq import (TRAIN_PAIRS, HELDOUT_PAIRS, INPUT_DIM, SeqDecoder,
                                    TransformerBody, encode, decode_correct)

ALL_PAIRS = TRAIN_PAIRS + HELDOUT_PAIRS


def _params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def _percentiles(times_ms):
    s = sorted(times_ms)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return p(0.50), p(0.95)


def _bench(fn, queries, warmup=10):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(queries):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="results/latency.csv")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)            # single-thread → comparable, stable timings

    # -- NOUS: energy net + solver + shared decoder --
    E = EnergyNet(input_dim=INPUT_DIM, state_dim=args.state_dim, n_rbf=8)
    nous_dec = SeqDecoder(args.state_dim)
    solver_ee = EulerLagrangeSolver(E, dt=args.dt, n_steps=args.n_steps, delta=1e-3)
    solver_no = EulerLagrangeSolver(E, dt=args.dt, n_steps=args.n_steps, delta=0.0)

    def nous_query(solver):
        def run():
            v, c = ALL_PAIRS[run.i % len(ALL_PAIRS)]
            run.i += 1
            q = solver.solve(encode(v, c), torch.zeros(args.state_dim))
            decode_correct(nous_dec, q, v, c)
        run.i = 0
        return run

    # -- Transformer: encoder + same decoder --
    tr = TransformerBody(args.state_dim)
    tr_dec = SeqDecoder(args.state_dim)

    def tr_query():
        v, c = ALL_PAIRS[tr_query.i % len(ALL_PAIRS)]
        tr_query.i += 1
        with torch.no_grad():
            r = tr(v, c)
            decode_correct(tr_dec, r, v, c)
    tr_query.i = 0

    rows = [
        ("nous", "early_exit_on",  _params(E, nous_dec), _bench(nous_query(solver_ee), args.queries)),
        ("nous", "early_exit_off", _params(E, nous_dec), _bench(nous_query(solver_no), args.queries)),
        ("transformer", "n/a",     _params(tr, tr_dec),  _bench(tr_query, args.queries)),
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "early_exit", "params", "p50_ms", "p95_ms", "queries"])
        print(f"{'model':>12} {'early_exit':>15} {'params':>8} {'p50_ms':>8} {'p95_ms':>8}")
        print("─" * 56)
        for name, ee, params, times in rows:
            p50, p95 = _percentiles(times)
            w.writerow([name, ee, params, f"{p50:.3f}", f"{p95:.3f}", args.queries])
            print(f"{name:>12} {ee:>15} {params:>8} {p50:>8.3f} {p95:>8.3f}")
    print(f"\nWrote {args.out}  (n_steps={args.n_steps}, single-thread CPU)")


if __name__ == "__main__":
    main()

"""
Controlled demo: NOUS as a GENERATIVE controller for an LLM-style generator.

Premise (grounded in the merged result, not the inert verifier): NOUS's edge is
GENERATIVE compositional construction, not accept/reject scoring. So the useful
role is a controller that OWNS the structured/compositional sub-query and
produces the valid output — where the LLM-style generator fails on novel
combinations.

Pipeline (controlled):
    query ──► router ──► structured/novel? ──► NOUS (compositional generator)
                         else               ──► LLM (transformer baseline)

We measure, on novel-composition queries (the held-out SCAN-mini pairs):
    - LLM alone  (transformer)              → expected to fail
    - LLM + NOUS controller (route to NOUS) → expected to recover

This reuses the proven SCAN-mini task/models (no new claims): NOUS central with
factored decode is the component that generalizes.

Run: python -m nous.controller_demo --seeds 0 1 2
"""

import argparse

import torch

from nous.train_compgen_seq import (TRAIN_PAIRS, HELDOUT_PAIRS, ACTIONS, encode,
                                    decode_correct, SeqDecoder, TransformerBody,
                                    nous_eqprop_step, _fit_backprop, seq_loss)
from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
import torch.nn as nn


def train_nous(seed, state_dim=32, n_rbf=8, dt=0.1, n_steps=60, eps=0.3, epochs=200, lr=0.02, dh=32):
    torch.manual_seed(seed)
    E = EnergyNet(input_dim=encode(0, 0).numel(), state_dim=state_dim, n_rbf=n_rbf)
    solver = EulerLagrangeSolver(E, dt=dt, n_steps=n_steps)
    dec = SeqDecoder(state_dim, hidden=dh)
    opt = torch.optim.Adam(list(E.parameters()) + list(dec.parameters()), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(TRAIN_PAIRS)):
            v, c = TRAIN_PAIRS[i]
            nous_eqprop_step(E, solver, dec, opt, encode(v, c), v, c, eps)
    def correct(v, c):
        q = solver.solve(encode(v, c), torch.zeros(state_dim))
        return decode_correct(dec, q, v, c)
    return correct


def train_llm(seed, rep_dim=32, epochs=200, lr=0.005, dh=32):
    """Transformer baseline = stand-in 'LLM' generator."""
    torch.manual_seed(seed)
    body = TransformerBody(rep_dim)
    dec = SeqDecoder(rep_dim, hidden=dh)
    _fit_backprop([body], lambda v, c: body(v, c), dec, lr=lr, cap=max(epochs, 600))
    def correct(v, c):
        with torch.no_grad():
            return decode_correct(dec, body(v, c), v, c)
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print("Controlled demo — NOUS as generative controller")
    print(f"novel-composition queries (held out): "
          f"{[(ACTIONS[v], 'twice' if c==1 else 'thrice') for v, c in HELDOUT_PAIRS]}")
    print("─" * 60)

    llm_acc, sys_acc = [], []
    for s in args.seeds:
        nous = train_nous(s); llm = train_llm(s)
        # route every novel-composition query to NOUS; LLM handles the rest
        llm_hits = [llm(v, c) for v, c in HELDOUT_PAIRS]
        sys_hits = [nous(v, c) for v, c in HELDOUT_PAIRS]      # controller intervenes
        la = sum(llm_hits) / len(llm_hits); sa = sum(sys_hits) / len(sys_hits)
        llm_acc.append(la); sys_acc.append(sa)
        print(f"seed {s}: LLM-alone={la*100:4.0f}%   LLM+NOUS-controller={sa*100:4.0f}%")

    la = sum(llm_acc) / len(llm_acc); sa = sum(sys_acc) / len(sys_acc)
    print("─" * 60)
    print(f"On novel-composition queries (n={len(HELDOUT_PAIRS)}/seed, {len(args.seeds)} seeds):")
    print(f"  LLM alone               : {la*100:.0f}% exact")
    print(f"  LLM + NOUS controller   : {sa*100:.0f}% exact   (Δ {(sa-la)*100:+.0f} pp)")
    print("NOUS owns the structured sub-query it can compose; LLM keeps the rest.")


if __name__ == "__main__":
    main()

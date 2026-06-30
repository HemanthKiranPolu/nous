"""
NOUS — Compositional Generalization (toy, 2-slot)

Smallest honest test of the core "novel combination" claim, on home turf
(no SCAN yet). Question: does the NOUS energy manifold COMPOSE — i.e. encode
two factors near-independently so an unseen combination of *known* primitives
decodes correctly?

Task
  - Input  : two slots — verb v ∈ {v0,v1,v2,v3}, count c ∈ {once, twice}.
             Encoded as one-hot(verb,4) ⊕ one-hot(count,2) → x ∈ R^6.
  - Output : FACTORED. Two heads read the SAME equilibrium q*:
               head_v : q* → 4-way verb logits
               head_c : q* → 2-way count logits
             Factoring is what makes "compositional" measurable: a joint
             8-way classifier would structurally score 0 on a held-out class
             and tell us nothing.

Split
  - Held out ENTIRELY from training: (v3, twice).
    v3 is seen only with `once`; `twice` is seen only with v0..v2.
    The verb head has therefore seen v3, the count head has seen twice —
    so each head CAN be right. Composition succeeds iff the free-phase q*
    for the never-seen pair lands where BOTH heads read out correctly.

Honesty
  - No preset accuracy gate. We run several seeds and report the actual
    distribution (the repo's XOR result is 4/4 on only 8% of seeds, so
    seed variance is expected and reported, not hidden).
  - A param-matched MLP baseline trained by backprop on the same 7 pairs is
    reported alongside, so the NOUS number is never shown in isolation.

Run:
  python -m nous.train_compgen --seeds 0 1 2 3 4 --out results/compgen_toy.json
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver

# ── Task definition ─────────────────────────────────────────────────────────
N_VERB   = 4
N_COUNT  = 2
INPUT_DIM = N_VERB + N_COUNT          # one-hot ⊕ one-hot
HELDOUT  = (3, 1)                     # (v3, twice) — never trained
ALL_PAIRS   = [(v, c) for v in range(N_VERB) for c in range(N_COUNT)]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p != HELDOUT]

CE = nn.functional.cross_entropy


def encode(v: int, c: int) -> torch.Tensor:
    """one-hot(verb) ⊕ one-hot(count) → R^6"""
    x = torch.zeros(INPUT_DIM)
    x[v] = 1.0
    x[N_VERB + c] = 1.0
    return x


def _ce(logits: torch.Tensor, target: int) -> torch.Tensor:
    return CE(logits.unsqueeze(0), torch.tensor([target]))


# ── NOUS: inline dual-head EqProp step ────────────────────────────────────────
# Mirrors nous.equilibrium_prop.EquilibriumProp.step exactly, but the decoder
# is two heads and the contrast/nudge energy is the SUM of the two CEs. The
# shared single-head class is left untouched.
def nous_eqprop_step(E, solver, head_v, head_c, opt, x, tv, tc, eps):
    state_dim = E.state_dim
    q0 = torch.zeros(state_dim)

    # -- Free phase --
    q_free = solver.solve(x, q0)
    loss = _ce(head_v(q_free), tv) + _ce(head_c(q_free), tc)
    grads_free = E.param_grad_at(x, q_free)

    # -- Nudge phase: gently clamp BOTH outputs --
    def extra_energy(q):
        return eps * (_ce(head_v(q), tv) + _ce(head_c(q), tc))

    q_nudge = solver.solve(x, q0, extra_energy_fn=extra_energy)
    grads_nudge = E.param_grad_at(x, q_nudge)

    # -- EqProp update for energy params (no backprop through ODE) --
    opt.zero_grad()
    for n, p in E.named_parameters():
        if p.requires_grad:
            p.grad = (1.0 / eps) * (grads_nudge[n] - grads_free[n])

    # -- Decoder heads: standard CE at the nudged (detached) equilibrium --
    ce_nudge = _ce(head_v(q_nudge), tv) + _ce(head_c(q_nudge), tc)
    ce_nudge.backward()

    opt.step()
    return loss.item()


def make_heads(state_dim):
    head_v = nn.Linear(state_dim, N_VERB)
    head_c = nn.Linear(state_dim, N_COUNT)
    for h in (head_v, head_c):
        nn.init.xavier_uniform_(h.weight, gain=0.3)
        nn.init.zeros_(h.bias)
    return head_v, head_c


def evaluate(decode_fn, pairs):
    """decode_fn(v,c) → (verb_pred, count_pred). Returns per-pair correctness."""
    out = {}
    for (v, c) in pairs:
        pv, pc = decode_fn(v, c)
        out[f"v{v}_c{c}"] = {
            "verb_ok":  int(pv == v),
            "count_ok": int(pc == c),
            "both_ok":  int(pv == v and pc == c),
        }
    return out


def run_nous(seed, state_dim, n_rbf, dt, n_steps, eps, epochs, lr):
    torch.manual_seed(seed)
    E = EnergyNet(input_dim=INPUT_DIM, state_dim=state_dim, n_rbf=n_rbf)
    solver = EulerLagrangeSolver(E, dt=dt, n_steps=n_steps)
    head_v, head_c = make_heads(state_dim)
    opt = torch.optim.Adam(
        list(E.parameters()) + list(head_v.parameters()) + list(head_c.parameters()),
        lr=lr,
    )

    for _ in range(epochs):
        order = torch.randperm(len(TRAIN_PAIRS))
        for i in order:
            v, c = TRAIN_PAIRS[i]
            nous_eqprop_step(E, solver, head_v, head_c, opt, encode(v, c), v, c, eps)

    def decode_fn(v, c):
        q = solver.solve(encode(v, c), torch.zeros(state_dim))
        return head_v(q).argmax().item(), head_c(q).argmax().item()

    return {
        "train":   evaluate(decode_fn, TRAIN_PAIRS),
        "heldout": evaluate(decode_fn, [HELDOUT]),
    }


# ── Baseline: param-matched feed-forward, trained by backprop ─────────────────
class MLPBaseline(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.body   = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.head_v = nn.Linear(hidden, N_VERB)
        self.head_c = nn.Linear(hidden, N_COUNT)

    def forward(self, x):
        h = self.body(x)
        return self.head_v(h), self.head_c(h)


def run_mlp(seed, hidden, epochs, lr):
    torch.manual_seed(seed)
    net = MLPBaseline(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for _ in range(epochs):
        order = torch.randperm(len(TRAIN_PAIRS))
        for i in order:
            v, c = TRAIN_PAIRS[i]
            lv, lc = net(encode(v, c))
            loss = _ce(lv, v) + _ce(lc, c)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def decode_fn(v, c):
        with torch.no_grad():
            lv, lc = net(encode(v, c))
        return lv.argmax().item(), lc.argmax().item()

    return {
        "train":   evaluate(decode_fn, TRAIN_PAIRS),
        "heldout": evaluate(decode_fn, [HELDOUT]),
    }


def _heldout_both(seed_results):
    """Fraction of seeds where the held-out pair was decoded fully correctly."""
    hits = [r["heldout"][f"v{HELDOUT[0]}_c{HELDOUT[1]}"]["both_ok"] for r in seed_results]
    return sum(hits) / len(hits), hits


def _train_both(seed_results):
    accs = []
    for r in seed_results:
        cells = r["train"].values()
        accs.append(sum(c["both_ok"] for c in cells) / len(cells))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--n-rbf", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.3)        # nudge β; memory: 0.1–0.5 for CE
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mlp-hidden", type=int, default=32)
    ap.add_argument("--out", type=str, default="results/compgen_toy.json")
    args = ap.parse_args()

    print("NOUS — Compositional Generalization (toy, 2-slot)")
    print("─" * 60)
    print(f"Train pairs : {TRAIN_PAIRS}")
    print(f"Held out    : (v{HELDOUT[0]}, c{HELDOUT[1]})  [never trained]")
    print(f"Seeds       : {args.seeds} | epochs={args.epochs} | eps={args.eps}")
    print("─" * 60)

    nous_runs, mlp_runs = [], []
    for s in args.seeds:
        nous_runs.append(run_nous(s, args.state_dim, args.n_rbf, args.dt,
                                  args.n_steps, args.eps, args.epochs, args.lr))
        mlp_runs.append(run_mlp(s, args.mlp_hidden, args.epochs, args.lr))
        n_h = nous_runs[-1]["heldout"][f"v{HELDOUT[0]}_c{HELDOUT[1]}"]
        m_h = mlp_runs[-1]["heldout"][f"v{HELDOUT[0]}_c{HELDOUT[1]}"]
        print(f"seed {s}: NOUS heldout both={n_h['both_ok']} "
              f"(v={n_h['verb_ok']},c={n_h['count_ok']})  |  "
              f"MLP heldout both={m_h['both_ok']} "
              f"(v={m_h['verb_ok']},c={m_h['count_ok']})")

    nous_ho, nous_hits = _heldout_both(nous_runs)
    mlp_ho,  mlp_hits  = _heldout_both(mlp_runs)

    summary = {
        "config": vars(args),
        "task": {"verbs": N_VERB, "counts": N_COUNT,
                 "heldout": list(HELDOUT), "train_pairs": [list(p) for p in TRAIN_PAIRS]},
        "nous": {"train_both_acc": _train_both(nous_runs),
                 "heldout_both_rate": nous_ho, "heldout_both_per_seed": nous_hits,
                 "per_seed": nous_runs},
        "mlp":  {"train_both_acc": _train_both(mlp_runs),
                 "heldout_both_rate": mlp_ho, "heldout_both_per_seed": mlp_hits,
                 "per_seed": mlp_runs},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("─" * 60)
    print(f"NOUS : train(both)={summary['nous']['train_both_acc']:.2f}  "
          f"heldout(both) on {nous_ho*100:.0f}% of seeds  {nous_hits}")
    print(f"MLP  : train(both)={summary['mlp']['train_both_acc']:.2f}  "
          f"heldout(both) on {mlp_ho*100:.0f}% of seeds  {mlp_hits}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

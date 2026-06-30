"""
NOUS as a VERIFIER — controlled compositional-validity test.

Premise to test (do NOT assume): does NOUS's energy verify the validity of
NOVEL compositions better than a param-matched discriminative net? This is the
only axis where NOUS has shown an edge (central EqProp + structured task). If it
wins here, a "NOUS verifier" system is worth building; if it ties the MLP, the
verifier is inert (same lesson as the SCAN-latent ablation).

Task — judge (command, candidate_output) → VALID / INVALID
  - command  : verb v ∈ {0..5}, count c ∈ {once,twice,thrice}.
  - correct output = action(v) repeated (c+1) times.
  - candidate encoded as a multiset over the 6 actions ⊕ its length.
  - VALID iff candidate == correct output for the command.

Negatives (deterministic): wrong-verb and wrong-count corruptions.

Split — the comp-gen trap
  - Held-out commands = verbs {3,4,5} × {twice, thrice}: the verifier must judge
    validity of compositions (known action, novel repeat-count) it never trained
    on. Verbs {3,4,5} are trained only at `once`; counts twice/thrice only with
    verbs {0,1,2}.

NOUS verifier = central EqProp energy classifier (the config that won the toy).
Baseline = param-matched MLP. Train-fit-gated, multi-seed.

Run: python -m nous.verify_compgen --seeds 0 1 2 3 4 5 6 7
"""

import argparse
import json

import torch
import torch.nn as nn

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp

N_VERB = 6
N_COUNT = 3                                   # 0=once,1=twice,2=thrice  → length c+1
TRAIN_VERBS = [0, 1, 2]
HELDOUT_VERBS = [3, 4, 5]
MULTI = [1, 2]                                # twice, thrice
FEAT = N_VERB + N_COUNT + N_VERB + 1          # cmd onehot ⊕ output multiset ⊕ length
CE = nn.functional.cross_entropy


def correct_out(v, c):
    return [v] * (c + 1)


def featurize(v, c, out):
    x = torch.zeros(FEAT)
    x[v] = 1.0                                # verb
    x[N_VERB + c] = 1.0                       # count
    for a in out:
        x[N_VERB + N_COUNT + a] += 1.0        # action multiset
    x[-1] = len(out) / N_COUNT                # normalized length
    return x


def candidates(v, c):
    """(feature, label) list: 1 correct + 2 corruptions for command (v, c)."""
    pos = correct_out(v, c)
    wrong_verb  = [(v + 1) % N_VERB] * (c + 1)
    wrong_count = [v] * (((c + 1) % N_COUNT) + 1)
    return [(featurize(v, c, pos), 1),
            (featurize(v, c, wrong_verb), 0),
            (featurize(v, c, wrong_count), 0)]


def build_split():
    train_cmds = ([(v, 0) for v in range(N_VERB)]
                  + [(v, c) for v in TRAIN_VERBS for c in MULTI])
    heldout_cmds = [(v, c) for v in HELDOUT_VERBS for c in MULTI]
    train = [ex for (v, c) in train_cmds for ex in candidates(v, c)]
    held  = [ex for (v, c) in heldout_cmds for ex in candidates(v, c)]
    return train, held


def acc(decode_fn, data):
    hits = [int(decode_fn(x) == y) for x, y in data]
    return sum(hits) / len(hits)


def balanced(decode_fn, data):
    """Balanced acc + per-class recall (chance = 0.5; not gameable by always-reject)."""
    pos = [decode_fn(x) == 1 for x, y in data if y == 1]
    neg = [decode_fn(x) == 0 for x, y in data if y == 0]
    rp = sum(pos) / len(pos); rn = sum(neg) / len(neg)
    return 0.5 * (rp + rn), rp, rn


def run_nous(seed, train, held, state_dim, n_rbf, dt, n_steps, eps, epochs, lr):
    torch.manual_seed(seed)
    E = EnergyNet(input_dim=FEAT, state_dim=state_dim, n_rbf=n_rbf)
    solver = EulerLagrangeSolver(E, dt=dt, n_steps=n_steps)
    dec = nn.Linear(state_dim, 2)
    nn.init.xavier_uniform_(dec.weight, gain=0.3); nn.init.zeros_(dec.bias)
    opt = torch.optim.Adam(list(E.parameters()) + list(dec.parameters()), lr=lr)
    eqp = EquilibriumProp(E, solver, dec, opt, eps=eps, phi_distance=0.05, phi_curvature=1.2)
    for _ in range(epochs):
        for i in torch.randperm(len(train)):
            x, y = train[i]
            eqp.step(x, torch.tensor(y), q0_override=torch.zeros(state_dim))
    def decode(x):
        q = solver.solve(x, torch.zeros(state_dim))
        return dec(q).argmax().item()
    return (acc(decode, train), *balanced(decode, held))


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def run_mlp(seed, train, held, h, epochs, lr):
    torch.manual_seed(seed)
    net = MLP(h)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(max(epochs, 400)):
        for i in torch.randperm(len(train)):
            x, y = train[i]
            loss = CE(net(x).unsqueeze(0), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
    def decode(x):
        with torch.no_grad():
            return net(x).argmax().item()
    return (acc(decode, train), *balanced(decode, held))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--n-rbf", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--gate", type=float, default=0.95)
    ap.add_argument("--out", type=str, default="results/verify_compgen.json")
    args = ap.parse_args()

    train, held = build_split()
    print("NOUS verifier — compositional-validity (held-out novel compositions)")
    print(f"train candidates={len(train)}  held-out candidates={len(held)} "
          f"(verbs {HELDOUT_VERBS} × twice/thrice)")
    print("─" * 60)

    nous, mlp = [], []
    for s in args.seeds:
        ntr, nbal, nrp, nrn = run_nous(s, train, held, args.state_dim, args.n_rbf, args.dt,
                                       args.n_steps, args.eps, args.epochs, args.lr)
        mtr, mbal, mrp, mrn = run_mlp(s, train, held, args.mlp_hidden, args.epochs, args.lr)
        nous.append((ntr, nbal, nrp, nrn)); mlp.append((mtr, mbal, mrp, mrn))
        print(f"seed {s}: NOUS tr={ntr:.2f} bal={nbal:.2f}(✓{nrp:.2f}/✗{nrn:.2f})  |  "
              f"MLP tr={mtr:.2f} bal={mbal:.2f}(✓{mrp:.2f}/✗{mrn:.2f})")

    kept = [i for i in range(len(args.seeds)) if nous[i][0] >= args.gate and mlp[i][0] >= args.gate]
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    nbal = mean([nous[i][1] for i in kept]); mbal = mean([mlp[i][1] for i in kept])
    nvr  = mean([nous[i][2] for i in kept]); mvr  = mean([mlp[i][2] for i in kept])
    summary = {"kept": [args.seeds[i] for i in kept], "n_kept": len(kept),
               "nous_bal": nbal, "mlp_bal": mbal, "delta_pp": (nbal - mbal) * 100,
               "nous_valid_recall": nvr, "mlp_valid_recall": mvr,
               "nous_raw": nous, "mlp_raw": mlp}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("─" * 60)
    print(f"train-fit gate ≥{args.gate}: kept {len(kept)}/{len(args.seeds)}   (chance bal = 50%)")
    print(f"NOUS held-out balanced={nbal*100:.1f}%  (accept-valid recall {nvr*100:.1f}%)")
    print(f"MLP  held-out balanced={mbal*100:.1f}%  (accept-valid recall {mvr*100:.1f}%)")
    print(f"Δ balanced = {(nbal-mbal)*100:+.1f} pp")


if __name__ == "__main__":
    main()

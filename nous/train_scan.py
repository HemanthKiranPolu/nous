"""
NOUS on SCAN — autoregressive energy decode (NOUS central).

The command is encoded (bag of token embeddings) and clamps the energy field.
The action sequence is decoded ONE token per EqProp step: at step t the input
clamp is [command_encoding ⊕ embedding(prev_action)], the net relaxes to an
equilibrium q*_t, the decoder reads the action, and q*_t warm-starts step t+1
(memory via basin — same recipe as nous/train_sentence.py, generalized to
seq2seq with teacher forcing). No backprop through the ODE.

⚠ SCALE: full SCAN (14.7k train, outputs ≤48 tokens) trained with per-token
EqProp is GPU-scale. On CPU use --train-n / --max-out-len to bound cost. The
defaults run a SMALL correctness sanity (can the model fit a handful of short
examples and decode them exactly?), NOT a generalization claim. The held-out
generalization number is a GPU run — command printed at the end.

Verified: overfits 10 short examples to train_exact=1.00 (loss 0.002) with the
default solver — the method is trainable; fitting more is a scale/tuning matter.
The solver must be well-converged (n_steps≳50, dt≈0.05); coarser settings leave
the equilibrium under-relaxed and training stalls (~0.7 train-fit).

Run (CPU sanity):
  python -m nous.train_scan --split addprim_jump --max-out-len 3 \
         --train-n 10 --test-n 10 --epochs 60
"""

import argparse

import torch
import torch.nn as nn

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp
from nous.scan_data import load_scan

START = 0          # reuse <pad> id as the decoder's start token
EOS = 1


def encode_pairs(pairs, in_vocab, out_vocab, max_out_len):
    """Filter to outputs ≤ max_out_len; map tokens → ids; append EOS to outputs."""
    enc = []
    for cmd, act in pairs:
        if len(act) + 1 > max_out_len:
            continue
        cmd_ids = [in_vocab[t] for t in cmd]
        act_ids = [out_vocab[t] for t in act] + [EOS]
        enc.append((torch.tensor(cmd_ids), torch.tensor(act_ids)))
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="addprim_jump")
    ap.add_argument("--max-out-len", type=int, default=6)
    ap.add_argument("--train-n", type=int, default=64)
    ap.add_argument("--test-n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--state-dim", type=int, default=96)
    ap.add_argument("--n-steps", type=int, default=60)     # under-converged below ~50
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = load_scan(args.split)
    in_vocab, out_vocab = d["in_vocab"], d["out_vocab"]
    n_out = len(out_vocab)

    train = encode_pairs(d["train"], in_vocab, out_vocab, args.max_out_len)[: args.train_n]
    test  = encode_pairs(d["test"],  in_vocab, out_vocab, args.max_out_len)[: args.test_n]

    print(f"NOUS on SCAN ({args.split}) — autoregressive energy decode")
    print("─" * 64)
    print(f"train={len(train)} test={len(test)} (outputs ≤{args.max_out_len}) "
          f"| out-vocab={n_out} | state={args.state_dim}D")
    print("─" * 64)

    in_emb  = nn.Embedding(len(in_vocab), args.embed_dim)
    out_emb = nn.Embedding(n_out, args.embed_dim)
    nn.init.normal_(in_emb.weight, std=0.02)
    nn.init.normal_(out_emb.weight, std=0.02)

    E = EnergyNet(input_dim=2 * args.embed_dim, state_dim=args.state_dim, n_rbf=16)
    decoder = nn.Linear(args.state_dim, n_out)
    nn.init.xavier_uniform_(decoder.weight, gain=0.3)
    nn.init.zeros_(decoder.bias)

    opt = torch.optim.Adam(list(in_emb.parameters()) + list(out_emb.parameters())
                           + list(E.parameters()) + list(decoder.parameters()), lr=args.lr)
    solver = EulerLagrangeSolver(E, dt=args.dt, n_steps=args.n_steps)
    eqprop = EquilibriumProp(E, solver, decoder, opt, eps=args.eps,
                             phi_distance=0.05, phi_curvature=1.2)

    def cmd_encode(cmd_ids):
        return in_emb(cmd_ids).mean(0).detach()        # bag-of-embeddings

    def exact_match(pairs):
        hits = 0
        for cmd_ids, act_ids in pairs:
            ce = cmd_encode(cmd_ids)
            q = torch.zeros(args.state_dim)
            prev, ok = START, True
            for t in range(len(act_ids)):
                x = torch.cat([ce, out_emb(torch.tensor(prev)).detach()])
                q = solver.solve(x, q)
                pred = decoder(q).argmax().item()
                if pred != act_ids[t].item():
                    ok = False
                    break
                prev = pred
            hits += int(ok)
        return hits / len(pairs) if pairs else 0.0

    for ep in range(args.epochs):
        tot = 0.0
        for i in torch.randperm(len(train)):
            cmd_ids, act_ids = train[i]
            ce = cmd_encode(cmd_ids)
            q = torch.zeros(args.state_dim)
            prev = START
            for t in range(len(act_ids)):
                x = torch.cat([ce, out_emb(torch.tensor(prev)).detach()])
                loss, _, q_free, _ = eqprop.step(x, act_ids[t], q0_override=q)
                tot += loss
                q = q_free
                prev = act_ids[t].item()              # teacher forcing
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:>3}: avg step loss={tot / sum(len(a) for _, a in train):.3f}  "
                  f"train_exact={exact_match(train):.2f}")

    tr, te = exact_match(train), exact_match(test)
    print("─" * 64)
    print(f"train exact-match: {tr:.2f}   test(held-out) exact-match: {te:.2f}")
    print("NOTE: small CPU sanity, not a generalization claim. Full run (GPU):")
    print("  python -m nous.train_scan --max-out-len 48 --train-n 14670 "
          "--test-n 7706 --epochs 30 --state-dim 256")


if __name__ == "__main__":
    main()

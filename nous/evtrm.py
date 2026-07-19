"""
EV-TRM — Energy-Verified Tiny Recursive Reasoner.

A tiny recursive reasoner (TRM-style: deep supervision, latent detached between
supervision steps) with a JOINTLY-TRAINED energy/verification head that predicts
its own constraint violations. The energy head adds what TRM/HRM lack: calibrated
self-verification → abstention and a trust-gate for routing from an LLM.

Mechanism, not SOTA: validated end-to-end on 4×4 Sudoku from scratch —
  exact-solve 95.8%, energy flags own errors AUC 0.81,
  selective prediction 95.8%→98.3% when abstaining on the top-20% energy.
9×9 (--side 9) is the real target but needs the full TRM training budget
(~20h, 1 GPU); see results/evtrm.md.

Background: the NOUS idea (non-autoregressive iterative energy minimization) is
validated by 2025 work — Energy-Based Transformers (arXiv:2507.02092), Tiny
Recursive Model (arXiv:2510.04871), Kona EBM. EV-TRM = TRM backbone + EBT-style
energy head. The obsolete RBF+EqProp NOUS core is dropped.

Run:
  python -m nous.evtrm --side 4 --givens 8 --epochs 90      # PoC (minutes, GPU)
  python -m nous.evtrm --side 9 --givens 35 --epochs 2000   # real run (~hours+)
"""

import argparse
import math
import random

import torch
import torch.nn as nn


def make_grids(side, givens, n, seed0, dev):
    """n (puzzle, solution) pairs. side∈{4,9}; box = √side."""
    b = int(math.isqrt(side))
    def base(rng):
        def pat(r, c): return (b*(r % b) + r//b + c) % side
        rows = [g*b + r for g in rng.sample(range(b), b) for r in rng.sample(range(b), b)]
        cols = [g*b + c for g in rng.sample(range(b), b) for c in rng.sample(range(b), b)]
        nums = rng.sample(range(1, side+1), side)
        return [nums[pat(rows[r], cols[c])] for r in range(side) for c in range(side)]
    out = []
    for i in range(n):
        rng = random.Random(seed0 + i)
        sol = base(rng); idx = list(range(side*side)); rng.shuffle(idx)
        puz = sol[:]
        for j in idx[givens:]:
            puz[j] = 0
        out.append((torch.tensor(puz), torch.tensor(sol)))
    return out


def group_ids(side):
    b = int(math.isqrt(side))
    row = torch.tensor([r for r in range(side) for c in range(side)])
    col = torch.tensor([c for r in range(side) for c in range(side)])
    box = torch.tensor([(r//b)*b + c//b for r in range(side) for c in range(side)])
    return row, col, box


def violations(grid, groups, side):
    """Missing distinct digits across rows/cols/boxes (0 == valid solution)."""
    v = 0
    for g in groups:
        for k in range(side):
            v += side - grid[g == k].unique().numel()
    return v


class EVTRM(nn.Module):
    def __init__(self, side, d=128, heads=4, layers=2, H=3, L=4, T=5):
        super().__init__()
        self.side, self.N, self.D = side, side*side, side
        self.in_emb  = nn.Embedding(side+1, d)
        self.ans_emb = nn.Embedding(side+1, d)
        self.pos     = nn.Parameter(torch.randn(self.N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4*d, dropout=0.0,
                                           batch_first=True, activation="gelu")
        self.block  = nn.TransformerEncoder(layer, layers)
        self.head   = nn.Linear(d, side+1)
        self.energy = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.H, self.L, self.T, self.d = H, L, T, d

    def forward(self, puz, sol=None):
        B = puz.shape[0]; x_emb = self.in_emb(puz); blank = (puz == 0)
        z = torch.zeros(B, self.N, self.d, device=puz.device); cur = puz.clone()
        logits = None; ce = puz.new_zeros((), dtype=torch.float)
        for _ in range(self.T):                                  # deep supervision
            for _ in range(self.H):
                for _ in range(self.L):
                    z = self.block(z + x_emb + self.ans_emb(cur) + self.pos)
            logits = self.head(z)
            if sol is not None and blank.any():
                ce = ce + nn.functional.cross_entropy(logits[blank], sol[blank])
            cur = torch.where(blank, logits.argmax(-1).clamp(1, self.D), puz)
            z = z.detach(); cur = cur.detach()                   # TRM detach between steps
        e = self.energy(z.mean(1)).squeeze(-1)                   # predicted violations
        return logits, cur, e, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=4, choices=[4, 9])
    ap.add_argument("--givens", type=int, default=8)
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--test-n", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--energy-weight", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    groups = [g.to(dev) for g in group_ids(args.side)]
    train = make_grids(args.side, args.givens, args.train_n, 0, dev)
    test  = make_grids(args.side, args.givens, args.test_n, 10_000, dev)

    model = EVTRM(args.side, d=args.d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"EV-TRM side={args.side} givens={args.givens} "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}K dev={dev}")

    def batches(data, bs):
        order = torch.randperm(len(data))
        for s in range(0, len(data), bs):
            ch = [data[i] for i in order[s:s+bs]]
            yield (torch.stack([p for p, _ in ch]).to(dev),
                   torch.stack([q for _, q in ch]).to(dev))

    def real_viol(grids):
        return torch.tensor([violations(g.cpu(), [gr.cpu() for gr in groups], args.side)
                             for g in grids], dtype=torch.float, device=grids.device)

    @torch.no_grad()
    def evaluate():
        model.eval(); cell = solve = 0; E = []; OK = []
        for puz, sol in batches(test, 400):
            _, cur, e, _ = model(puz); blank = (puz == 0)
            cell += (((cur == sol) & blank).sum().item() / blank.sum().item()) * puz.shape[0]
            solve += (cur == sol).all(1).sum().item(); E.append(e); OK.append((cur == sol).all(1))
        e = torch.cat(E); ok = torch.cat(OK)
        ef, eo = e[~ok], e[ok]
        auc = (ef.unsqueeze(1) > eo.unsqueeze(0)).float().mean().item() if len(ef) and len(eo) else float("nan")
        model.train()
        return cell/len(test), solve/len(test), auc

    for ep in range(args.epochs):
        for puz, sol in batches(train, args.bs):
            _, cur, e, ce = model(puz, sol)
            loss = ce + args.energy_weight * nn.functional.mse_loss(e, real_viol(cur))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 10 == 0 or ep == args.epochs-1:
            c, sv, auc = evaluate()
            print(f"ep{ep:>4} | cell={c*100:5.1f}% solve={sv*100:5.1f}% | energy→error AUC={auc:.2f}")


if __name__ == "__main__":
    main()

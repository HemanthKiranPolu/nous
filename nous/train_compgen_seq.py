"""
NOUS — Compositional Generalization, SCAN-mini (sequence output)

Difficulty lives in the OUTPUT binding (mirrors SCAN's "jump twice"), not in
the input encoding. A precursor toy with one-hot⊕one-hot input AND independent
per-slot heads was confounded — every model got the held-out pair for free
(MLP 8/8). Here a known symbol must be produced in a structurally NOVEL slot.

Task
  - Input : verb v ∈ {0..N_VERB-1}, count c ∈ {once,twice,thrice} (lengths 1,2,3).
  - Output: the verb's action symbol emitted (c+1) times.
              (JUMP, thrice) → [JUMP, JUMP, JUMP]
  - Decoder: ONE shared, position-conditioned head over q*:
               tok(q*, pos) → action logits   (weights tied across positions)
               len(q*)      → length {1,2,3}

Split — the SCAN trap (rich held-out for honest mean±std)
  - Every verb is trained at `once`. Only TRAIN_VERBS are ever trained with
    twice/thrice. HELDOUT_VERBS appear ONLY at `once`.
  - Held out: HELDOUT_VERBS × {twice, thrice} → each requires emitting a KNOWN
    symbol at positions (1, 2) that were never trained for that verb.
  - Multiple held-out points per seed ⇒ per-seed held-out accuracy is a real
    fraction; the 32-seed mean±std and pooled Wilson CI are meaningful.

Honesty
  - No preset gate. Report the actual distribution.
  - Train-fit gating: the headline comparison uses ONLY seeds where BOTH NOUS
    and the param-matched MLP reach train ≥ --train-fit-gate, so a held-out
    miss is never confounded by a model that failed to fit train.
  - Identical decoder for both models; any NOUS edge is measured, not asserted.

Run:
  python -m nous.train_compgen_seq --seeds $(seq 0 31) --epochs 200 \
         --out results/scan_mini_32s.json
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver

ACTIONS  = ["WALK", "LOOK", "RUN", "JUMP", "TURN", "STAY"]
N_VERB   = len(ACTIONS)               # 6
N_COUNT  = 3                          # 0=once(len1), 1=twice(len2), 2=thrice(len3)
MAX_LEN  = N_COUNT
INPUT_DIM = N_VERB + N_COUNT

TRAIN_VERBS   = [0, 1, 2]             # seen with every count
HELDOUT_VERBS = [3, 4, 5]            # seen ONLY at `once`
MULTI_COUNTS  = [1, 2]               # twice, thrice — the novel-slot counts

# pairs trained: every verb at `once`, plus TRAIN_VERBS at twice/thrice
TRAIN_PAIRS = ([(v, 0) for v in range(N_VERB)]
               + [(v, c) for v in TRAIN_VERBS for c in MULTI_COUNTS])
# pairs held out entirely: HELDOUT_VERBS at twice/thrice
HELDOUT_PAIRS = [(v, c) for v in HELDOUT_VERBS for c in MULTI_COUNTS]

CE = nn.functional.cross_entropy


def encode(v: int, c: int) -> torch.Tensor:
    x = torch.zeros(INPUT_DIM)
    x[v] = 1.0
    x[N_VERB + c] = 1.0
    return x


def _ce(logits: torch.Tensor, target: int) -> torch.Tensor:
    return CE(logits.unsqueeze(0), torch.tensor([target]))


# ── Shared position-conditioned sequence decoder ──────────────────────────────
class SeqDecoder(nn.Module):
    def __init__(self, state_dim, hidden=32, pos_dim=4):
        super().__init__()
        self.pos_emb = nn.Embedding(MAX_LEN, pos_dim)
        self.tok = nn.Sequential(nn.Linear(state_dim + pos_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, N_VERB))
        self.length = nn.Linear(state_dim, MAX_LEN)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def tok_logits(self, q, pos):
        pe = self.pos_emb(torch.tensor(pos))
        return self.tok(torch.cat([q, pe], dim=-1))

    def len_logits(self, q):
        return self.length(q)


def seq_loss(dec, q, v, c):
    loss = _ce(dec.len_logits(q), c)
    for pos in range(c + 1):
        loss = loss + _ce(dec.tok_logits(q, pos), v)        # token id == verb id
    return loss


def decode_correct(dec, q, v, c):
    """Exact-match: predicted length AND every token must match."""
    pred_len = dec.len_logits(q).argmax().item() + 1
    toks = [dec.tok_logits(q, pos).argmax().item() for pos in range(pred_len)]
    return int(pred_len == c + 1 and toks == [v] * (c + 1))


# ── NOUS: inline EqProp step with the sequence decoder ────────────────────────
def nous_eqprop_step(E, solver, dec, opt, x, v, c, eps):
    q0 = torch.zeros(E.state_dim)

    q_free = solver.solve(x, q0)
    grads_free = E.param_grad_at(x, q_free)

    def extra_energy(q):
        return eps * seq_loss(dec, q, v, c)

    q_nudge = solver.solve(x, q0, extra_energy_fn=extra_energy)
    grads_nudge = E.param_grad_at(x, q_nudge)

    opt.zero_grad()
    for n, p in E.named_parameters():
        if p.requires_grad:
            p.grad = (1.0 / eps) * (grads_nudge[n] - grads_free[n])

    seq_loss(dec, q_nudge, v, c).backward()                 # decoder via CE at nudge
    opt.step()


def run_nous(seed, state_dim, n_rbf, dt, n_steps, eps, epochs, lr, dec_hidden):
    torch.manual_seed(seed)
    E = EnergyNet(input_dim=INPUT_DIM, state_dim=state_dim, n_rbf=n_rbf)
    solver = EulerLagrangeSolver(E, dt=dt, n_steps=n_steps)
    dec = SeqDecoder(state_dim, hidden=dec_hidden)
    opt = torch.optim.Adam(list(E.parameters()) + list(dec.parameters()), lr=lr)

    for _ in range(epochs):
        for i in torch.randperm(len(TRAIN_PAIRS)):
            v, c = TRAIN_PAIRS[i]
            nous_eqprop_step(E, solver, dec, opt, encode(v, c), v, c, eps)

    def correct(v, c):
        q = solver.solve(encode(v, c), torch.zeros(state_dim))
        return decode_correct(dec, q, v, c)

    return _report(correct)


# ── Baseline: MLP body feeding the IDENTICAL decoder, backprop-trained ────────
class MLPBody(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out_dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


# Generic backprop trainer for the baselines. `rep(v, c)` returns the body's
# representation of the input (differentiable). Trains until the train set is
# perfectly fit (early stop) or `cap` epochs elapse — every baseline gets its
# best shot at fitting train, so the train-fit gate isolates generalization.
def _fit_backprop(modules, rep, dec, lr, cap):
    params = [p for m in modules for p in m.parameters()] + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(cap):
        for i in torch.randperm(len(TRAIN_PAIRS)):
            v, c = TRAIN_PAIRS[i]
            loss = seq_loss(dec, rep(v, c), v, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 25 == 24:
            with torch.no_grad():
                if all(decode_correct(dec, rep(v, c), v, c) for (v, c) in TRAIN_PAIRS):
                    break


def run_mlp(seed, rep_dim, hidden, epochs, lr, dec_hidden):
    torch.manual_seed(seed)
    body = MLPBody(hidden, rep_dim)
    dec = SeqDecoder(rep_dim, hidden=dec_hidden)
    _fit_backprop([body], lambda v, c: body(encode(v, c)), dec, lr, cap=max(epochs, 600))

    def correct(v, c):
        with torch.no_grad():
            r = body(encode(v, c))
        return decode_correct(dec, r, v, c)

    return _report(correct)


# ── Baseline: small Transformer encoder feeding the IDENTICAL decoder ─────────
# Input is the natural 2-token sequence [verb, count] with learned embeddings
# (not one-hot) — the encoding that gives a transformer its best shot. Self-
# attention pools the two tokens; the shared SeqDecoder reads the pooled rep.
class TransformerBody(nn.Module):
    def __init__(self, rep_dim, d_model=32, nhead=4, layers=2, ff=64):
        super().__init__()
        self.verb_emb  = nn.Embedding(N_VERB, d_model)
        self.count_emb = nn.Embedding(N_COUNT, d_model)
        self.pos = nn.Parameter(torch.randn(2, d_model) * 0.1)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(d_model, rep_dim)

    def forward(self, v, c):
        toks = torch.stack([self.verb_emb(torch.tensor(v)),
                            self.count_emb(torch.tensor(c))]) + self.pos   # (2, d)
        h = self.tr(toks.unsqueeze(0)).squeeze(0)                          # (2, d)
        return torch.tanh(self.proj(h.mean(0)))


def run_transformer(seed, rep_dim, epochs, lr, dec_hidden):
    torch.manual_seed(seed)
    body = TransformerBody(rep_dim)
    dec = SeqDecoder(rep_dim, hidden=dec_hidden)
    # transformers favour a smaller step on tiny data; give them the same
    # train-until-fit budget as the MLP so the gate is the only filter.
    _fit_backprop([body], lambda v, c: body(v, c), dec, lr=5e-3, cap=max(epochs, 600))

    def correct(v, c):
        with torch.no_grad():
            r = body(v, c)
        return decode_correct(dec, r, v, c)

    return _report(correct)


def _report(correct_fn):
    train_hits   = [correct_fn(v, c) for (v, c) in TRAIN_PAIRS]
    heldout_hits = [correct_fn(v, c) for (v, c) in HELDOUT_PAIRS]
    return {
        "train_exact":   sum(train_hits) / len(train_hits),
        "heldout_exact": sum(heldout_hits) / len(heldout_hits),
        "heldout_hits":  heldout_hits,        # per-pair 0/1, for pooled CI
    }


# ── Statistics ────────────────────────────────────────────────────────────────
def mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


def wilson_ci(successes, total, z=1.96):
    """Wilson 95% CI for a binomial proportion (correct for pooled 0/1 trials)."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--n-rbf", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--dec-hidden", type=int, default=32)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--train-fit-gate", type=float, default=0.99)
    ap.add_argument("--out", type=str, default="results/scan_mini_32s.json")
    args = ap.parse_args()

    print("NOUS — Compositional Generalization, SCAN-mini (sequence output)")
    print("─" * 64)
    print(f"Verbs={N_VERB} counts={N_COUNT} | train pairs={len(TRAIN_PAIRS)} "
          f"heldout pairs={len(HELDOUT_PAIRS)} (verbs {HELDOUT_VERBS} × twice/thrice)")
    print(f"Seeds={len(args.seeds)} epochs={args.epochs} eps={args.eps} "
          f"gate≥{args.train_fit_gate}")
    print("─" * 64)

    runs = {"nous": [], "mlp": [], "transformer": []}
    for s in args.seeds:
        runs["nous"].append(run_nous(s, args.state_dim, args.n_rbf, args.dt, args.n_steps,
                                     args.eps, args.epochs, args.lr, args.dec_hidden))
        runs["mlp"].append(run_mlp(s, args.state_dim, args.mlp_hidden, args.epochs,
                                   args.lr, args.dec_hidden))
        runs["transformer"].append(run_transformer(s, args.state_dim, args.epochs,
                                                    args.lr, args.dec_hidden))
        print(f"seed {s:>2}: " + "  ".join(
            f"{k.upper()[:4]} tr={runs[k][-1]['train_exact']:.2f} "
            f"ho={runs[k][-1]['heldout_exact']:.2f}" for k in runs))

    # -- train-fit-gated comparison: keep seeds where ALL models fit train --
    gate = args.train_fit_gate
    kept = [i for i in range(len(args.seeds))
            if all(runs[k][i]["train_exact"] >= gate for k in runs)]
    kept_seeds = [args.seeds[i] for i in kept]

    def gated_stats(model_runs):
        per_seed = [model_runs[i]["heldout_exact"] for i in kept]
        m, sd = mean_std(per_seed)
        pooled = [h for i in kept for h in model_runs[i]["heldout_hits"]]
        lo, hi = wilson_ci(sum(pooled), len(pooled))
        return {"per_seed_mean": m, "per_seed_std": sd,
                "pooled_n": len(pooled),
                "pooled_acc": (sum(pooled) / len(pooled)) if pooled else 0.0,
                "wilson95": [lo, hi], "per_seed": per_seed}

    gated = {k: gated_stats(v) for k, v in runs.items()}

    summary = {
        "config": vars(args),
        "task": {"actions": ACTIONS, "train_verbs": TRAIN_VERBS,
                 "heldout_verbs": HELDOUT_VERBS, "multi_counts": MULTI_COUNTS,
                 "heldout_pairs": [list(p) for p in HELDOUT_PAIRS],
                 "note": "held-out = known symbol emitted in a novel position"},
        "gating": {"gate": gate, "kept_seeds": kept_seeds, "n_kept": len(kept),
                   "n_total": len(args.seeds)},
        "delta_pp": {k: (gated["nous"]["pooled_acc"] - gated[k]["pooled_acc"]) * 100
                     for k in ("mlp", "transformer")},
    }
    for k in runs:
        summary[k] = {"gated": gated[k], "per_seed_raw": runs[k]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("─" * 64)
    print(f"Train-fit gate ≥{gate}: kept {len(kept)}/{len(args.seeds)} seeds {kept_seeds}")
    for k in ("nous", "mlp", "transformer"):
        g = gated[k]
        print(f"{k.upper():>11} heldout: {g['per_seed_mean']*100:5.1f} ± {g['per_seed_std']*100:4.1f} %"
              f"  (pooled {g['pooled_acc']*100:.1f}%, 95%CI "
              f"[{g['wilson95'][0]*100:.1f},{g['wilson95'][1]*100:.1f}], n={g['pooled_n']})")
    print(f"Δ  NOUS−MLP = {summary['delta_pp']['mlp']:+.1f} pp   "
          f"NOUS−Transformer = {summary['delta_pp']['transformer']:+.1f} pp")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

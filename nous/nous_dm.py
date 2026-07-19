"""
NOUS-DM PoC v2 — Growth TRIGGER, not growth target.

v1 (see results/nous_dm.md) found the interesting axis isn't "targeted vs
untargeted growth" (targeting only sped up recovery, didn't fix false growth)
-- it's WHEN to grow. The plateau trigger fires during ordinary optimization
difficulty (small-XOR training genuinely stalls before escaping on its own),
not just genuine representational insufficiency.

v2 adds a PATIENCE PROBE before growing: pause, train P more epochs at a
lower LR, and only grow if held-out validation loss still doesn't improve.
That's the falsifiable claim under test:

  "NOUS-DM grows only when representation is insufficient,
   not when optimization is temporarily slow."

Five variants (same task/model as v1 -- 4 binary factors, XOR label, 2 hidden
units can't represent stage 2/3's added XOR term):
  DEN + plateau       -- v1 baseline (untargeted growth, dumb trigger)
  NOUS-DM + plateau   -- v1 result (targeted growth, dumb trigger)
  DEN + probe         -- does the probe alone (no targeting) fix false growth?
  NOUS-DM + probe     -- the actual new claim (targeting + probe)
  NOUS-DM + probe + prune -- does pruning unhelpful grown dims fix calibration?

Metrics: recovery epochs, final accuracy, false grows (no-shift control),
growth precision (useful grows / total grows), forgetting (accuracy on
earlier stages' data re-measured at the end).

Honest scope: toy synthetic task, not a real benchmark.

Run:
  python -m nous.nous_dm
"""

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F_

FACTORS = ["color", "size", "shape", "material"]
F = len(FACTORS) * 2
STAGE_FACTORS = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]     # stage 2 adds shape, stage 3 adds material


def make_data(n, active_factors, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, len(FACTORS)), generator=g)
    x = torch.zeros(n, F)
    for i in range(len(FACTORS)):
        x[torch.arange(n), 2 * i + bits[:, i]] = 1.0
    y = torch.zeros(n, dtype=torch.long)
    for i in active_factors:
        y = y ^ bits[:, i]
    return x, y.float()


class GrowingModel(nn.Module):
    def __init__(self, d0=2):
        super().__init__()
        self.W  = nn.Parameter(torch.randn(d0, F) * 0.5)
        self.b  = nn.Parameter(torch.zeros(d0))
        self.a  = nn.Parameter(torch.randn(d0) * 0.3)
        self.a0 = nn.Parameter(torch.zeros(1))
        self.grown_mask = [False] * d0                    # True for dims added by growth (prunable)

    def forward(self, x):
        q = torch.tanh(x @ self.W.t() + self.b)
        return torch.sigmoid(q @ self.a + self.a0).squeeze(-1)

    def grow(self, w_new, b_new, a_new):
        self.W = nn.Parameter(torch.cat([self.W.data, w_new[None]]))
        self.b = nn.Parameter(torch.cat([self.b.data, b_new.reshape(1)]))
        self.a = nn.Parameter(torch.cat([self.a.data, a_new.reshape(1)]))
        self.grown_mask.append(True)

    def prune(self, idx):
        keep = [i for i in range(self.W.shape[0]) if i != idx]
        self.W = nn.Parameter(self.W.data[keep])
        self.b = nn.Parameter(self.b.data[keep])
        self.a = nn.Parameter(self.a.data[keep])
        self.grown_mask = [self.grown_mask[i] for i in keep]


def fit_candidate(x, residual, steps=200, lr=0.1, weight_decay=0.05):
    """Cascade-Correlation candidate: unit maximizing |corr(unit(x), residual)|.

    weight_decay keeps the unit off tanh's saturated tails -- correlation is
    scale-invariant for a linear unit, so unconstrained Adam inflates ||w||
    into saturation, which then starves the readout of gradient once grown.
    """
    w = torch.randn(F, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    r = residual.detach() - residual.detach().mean()
    for _ in range(steps):
        g = torch.tanh(x @ w + b)
        g = g - g.mean()
        corr = (g * r).sum() / (g.norm() * r.norm() + 1e-8)
        opt.zero_grad(); (-corr.abs()).backward(); opt.step()
    with torch.no_grad():
        g = torch.tanh(x @ w + b); g = g - g.mean()
        corr = ((g * r).sum() / (g.norm() * r.norm() + 1e-8)).abs().item()
    return w.detach(), b.detach(), corr


def probe_ok_to_grow(model, x, y, xv, yv, epochs=30, lr=0.015, eps=5e-3):
    """Pause growth: train P more epochs at a lower LR. If val loss improves,
    the plateau was ordinary optimization difficulty -- don't grow, keep the
    free training. If it still stalls, restore pre-probe weights and grow."""
    snapshot = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        val_before = F_.binary_cross_entropy(model(xv), yv).item()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = F_.binary_cross_entropy(model(x), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        val_after = F_.binary_cross_entropy(model(xv), yv).item()
    if val_after < val_before - eps:
        return False                                       # kept the extra training, no grow
    model.load_state_dict(snapshot)                          # revert probe, then grow
    return True


def run_stage(model, x, y, xv, yv, mode, trigger, epochs, patience, plateau_eps, plateau_bad, tau, stats):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    best, stall, recovered_at = float("inf"), 0, None
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = F_.binary_cross_entropy(pred, y)
        loss.backward(); opt.step()

        acc = ((pred.detach() > 0.5).float() == y).float().mean().item()
        if recovered_at is None and acc >= 0.95:
            recovered_at = ep

        l = loss.item()
        stall = stall + 1 if l >= best - plateau_eps else 0
        best = min(best, l)

        if mode != "fixed" and stall >= patience and best > plateau_bad:
            do_grow = True if trigger == "plateau" else probe_ok_to_grow(model, x, y, xv, yv)
            if do_grow:
                with torch.no_grad():
                    val_before_grow = F_.binary_cross_entropy(model(xv), yv).item()
                    residual = y - model(x)
                if mode == "targeted":
                    w, b, corr = fit_candidate(x, residual)
                    grow_it = corr > tau
                else:
                    w, b, corr = torch.randn(F) * 0.5, torch.zeros(1), None
                    grow_it = True
                if grow_it:
                    a_new = torch.tensor(0.5 if residual.mean() >= 0 else -0.5) if mode == "targeted" \
                        else torch.randn(1).squeeze() * 0.3
                    model.grow(w, b, a_new)
                    stats["grows"] += 1
                    stats["val_before_grow"].append(val_before_grow)
            opt = torch.optim.Adam(model.parameters(), lr=0.05)
            best, stall = float("inf"), 0

    with torch.no_grad():
        final_acc = ((model(x) > 0.5).float() == y).float().mean().item()
        final_val = F_.binary_cross_entropy(model(xv), yv).item()
    for v in stats["val_before_grow"]:
        stats["useful"] += final_val < v - 5e-3
    stats["val_before_grow"] = []
    return final_acc, recovered_at


def maybe_prune(model, xv, yv, eps=2e-3):
    with torch.no_grad():
        base = F_.binary_cross_entropy(model(xv), yv).item()
        removed = 0
        for i in reversed(range(model.W.shape[0])):
            if not model.grown_mask[i]:
                continue
            saved_a = model.a.data[i].clone()
            model.a.data[i] = 0.0
            loss_without = F_.binary_cross_entropy(model(xv), yv).item()
            if loss_without < base + eps:
                model.prune(i)
                removed += 1
            else:
                model.a.data[i] = saved_a
    return removed


def run_condition(mode, trigger, prune, factor_schedule, seed, epochs=250, d0=2):
    torch.manual_seed(seed)
    model = GrowingModel(d0)
    stats = {"grows": 0, "useful": 0, "val_before_grow": []}
    val_sets = []                                            # (stage, xv, yv) for forgetting check
    for stage, active in enumerate(factor_schedule):
        x, y = make_data(400, active, seed=1000 * seed + stage)
        xv, yv = make_data(200, active, seed=2000 * seed + stage)
        val_sets.append((xv, yv))
        acc, recovered_at = run_stage(
            model, x, y, xv, yv, mode, trigger, epochs=epochs, patience=40,
            plateau_eps=1e-3, plateau_bad=0.05, tau=0.5, stats=stats,
        )
        if prune:
            maybe_prune(model, xv, yv)
    with torch.no_grad():
        forget = None
        if len(val_sets) > 1:
            earlier_accs = []
            for xv, yv in val_sets[:-1]:
                earlier_accs.append(((model(xv) > 0.5).float() == yv).float().mean().item())
            forget = sum(earlier_accs) / len(earlier_accs)
    return acc, recovered_at, stats["grows"], stats["useful"], forget


def summarize(mode, trigger, prune, schedule, label, seeds, track_forgetting):
    accs, recs, grows, usefuls, forgets = [], [], [], [], []
    for s in seeds:
        acc, rec, grow, useful, forget = run_condition(mode, trigger, prune, schedule, s)
        accs.append(acc); recs.append(rec if rec is not None else float("nan"))
        grows.append(grow); usefuls.append(useful)
        if forget is not None:
            forgets.append(forget)
    n = len(seeds)
    mean_acc = sum(accs) / n
    valid_recs = [r for r in recs if r == r]
    mean_rec = sum(valid_recs) / len(valid_recs) if valid_recs else float("nan")
    mean_grow = sum(grows) / n
    precision = (sum(usefuls) / sum(grows)) if sum(grows) else float("nan")
    line = (f"{label:<26} final_acc={mean_acc*100:5.1f}%  recover_ep={mean_rec:6.1f}  "
            f"dims={mean_grow:4.1f}  precision={precision*100:5.1f}%")
    if track_forgetting and forgets:
        line += f"  earlier_stage_acc={sum(forgets)/len(forgets)*100:5.1f}%"
    print(line)


VARIANTS = [
    ("generic",  "plateau", False, "DEN + plateau trigger"),
    ("targeted", "plateau", False, "NOUS-DM + plateau trigger"),
    ("generic",  "probe",   False, "DEN + probe trigger"),
    ("targeted", "probe",   False, "NOUS-DM + probe trigger"),
    ("targeted", "probe",   True,  "NOUS-DM + probe + prune"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    shift_schedule = STAGE_FACTORS                              # color+size -> +shape -> +material
    no_shift_schedule = [STAGE_FACTORS[0]] * 3                  # control: nothing ever changes

    print("=== hidden-factor SHIFT (color+size -> +shape -> +material) ===")
    for mode, trigger, prune, label in VARIANTS:
        summarize(mode, trigger, prune, shift_schedule, label, seeds, track_forgetting=True)

    print("\n=== NO-SHIFT control (false-growth check) ===")
    for mode, trigger, prune, label in VARIANTS:
        summarize(mode, trigger, prune, no_shift_schedule, label, seeds, track_forgetting=False)


if __name__ == "__main__":
    main()

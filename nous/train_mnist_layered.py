"""
NOUS-Layered: Proper multi-layer EqProp on full MNIST.

Architecture: x(784) → h1(512) → h2(256) → y(10)
Training: Centered EqProp (C-EP) with block coordinate descent solver.
Dataset: Full 60k MNIST (or --subset N for testing).

This is the architecture class that achieves 99%+ in the literature.
The single-q NOUS has structural capacity ceiling ~93-95%.
Layered EqProp has no such ceiling — depth = representational power.

How it works:
  Free phase:   find (h1*, h2*, y*) = argmin E(x, h1, h2, y)
  Pos nudge:    find (h1+, h2+, y+) = argmin E + β·CE(y, target)  (y pulled toward correct)
  Neg nudge:    find (h1-, h2-, y-) = argmin E - β·CE(y, target)  (y pushed away)
  C-EP update:  ΔW_l = (1/2β)[h_{l-1}^+ h_l^+T - h_{l-1}^- h_l^-T]  (local Hebbian!)

The C-EP update is LOCAL: each weight only needs the pre/post activities
at the two equilibria. No global error signal required.

Usage:
  python -m nous.train_mnist_layered                    # 10k subset default
  python -m nous.train_mnist_layered --full             # full 60k
  python -m nous.train_mnist_layered --subset 5000
"""

import argparse, os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--full",    action="store_true")
parser.add_argument("--subset",  type=int, default=10000)
parser.add_argument("--epochs",  type=int, default=30)
parser.add_argument("--beta",    type=float, default=0.1)
parser.add_argument("--lr",      type=float, default=0.05)   # SGD works better for EqProp
parser.add_argument("--n-steps", type=int, default=30)
parser.add_argument("--dims",    nargs="+", type=int, default=[512, 256, 10])
args = parser.parse_args()

N_TRAIN   = None if args.full else args.subset
EPOCHS    = args.epochs
LAYER_DIMS = args.dims          # hidden + output dimensions
INPUT_DIM  = 784
BETA       = args.beta
LR         = args.lr
N_STEPS    = args.n_steps
OUT_DIR    = "nous_output_layered"
os.makedirs(OUT_DIR, exist_ok=True)

print("NOUS-Layered: multi-layer C-EP on MNIST")
print(f"  Arch: {INPUT_DIM} → {' → '.join(str(d) for d in LAYER_DIMS)}")
print(f"  Train: {'60k' if args.full else N_TRAIN}  |  Epochs: {EPOCHS}")
print(f"  β={BETA}  lr={LR}  n_steps={N_STEPS}")
print()

# ── Data ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
tr_full = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf)
te_full = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)

if N_TRAIN:
    per = N_TRAIN // 10
    idx, c = [], {i: 0 for i in range(10)}
    for i, (_, l) in enumerate(tr_full):
        if c[l] < per: idx.append(i); c[l] += 1
        if len(idx) >= N_TRAIN: break
    train_ds = torch.utils.data.Subset(tr_full, idx)
else:
    train_ds = tr_full

test_ds = te_full
print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

# ── Layered Hopfield Network ──────────────────────────────────────────────────
torch.manual_seed(42)

dims = [INPUT_DIM] + LAYER_DIMS
n_layers = len(LAYER_DIMS)

# Weight matrices W[l]: dims[l] → dims[l+1]  (stored as (in, out))
W = [nn.Parameter(torch.zeros(dims[l], dims[l+1])) for l in range(n_layers)]
b = [nn.Parameter(torch.zeros(dims[l+1])) for l in range(n_layers)]
for w in W:
    nn.init.xavier_uniform_(w.data, gain=1.0)

all_params = W + b

# SGD momentum (classic EqProp uses SGD, not Adam — more stable for local rules)
optimizer = torch.optim.SGD(all_params, lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Block coordinate descent solver ──────────────────────────────────────────
def hardtanh(x): return x.clamp(-1, 1)
act = hardtanh   # bounded activation → energy-bounded basins, stable dynamics


def solve(x: torch.Tensor, beta: float = 0.0, target: int = None,
          state0: list = None) -> list:
    """
    Find energy minimum via block coordinate descent.
    Updates each layer sequentially until convergence.
    beta=0: free phase. beta≠0: nudged phase.
    """
    state = [torch.zeros(d) for d in LAYER_DIMS] if state0 is None else \
            [h.clone().detach() for h in state0]

    for _ in range(N_STEPS):
        max_delta = 0.0
        for l in range(n_layers):
            # Input from layer below
            h_prev = x if l == 0 else state[l-1]
            pre    = h_prev @ W[l] + b[l]

            # Feedback from layer above (if not output)
            if l < n_layers - 1:
                post = state[l+1] @ W[l+1].t()
            else:
                post = torch.zeros_like(state[l])

            if l < n_layers - 1:
                h_new = act(pre + post)
            else:
                # Output layer: linear + nudge
                h_new = pre + post
                if beta != 0.0 and target is not None:
                    with torch.enable_grad():
                        h_g = h_new.detach().requires_grad_(True)
                        ce  = F.cross_entropy(h_g.unsqueeze(0), torch.tensor([target]))
                        grad = torch.autograd.grad(ce, h_g)[0]
                    h_new = h_new - beta * grad.detach()

            delta = (h_new - state[l]).abs().max().item()
            max_delta = max(max_delta, delta)
            state[l] = h_new

        if max_delta < 1e-4:
            break

    return state


# ── C-EP weight update ────────────────────────────────────────────────────────
# ΔW_l = (1/2β) [h_{l-1}^+⊗h_l^+ - h_{l-1}^-⊗h_l^-]
# This is purely LOCAL: weight update = outer product of adjacent layer activities.
# No error signal needs to travel the full network. Biologically plausible.

ACCUMULATE = 8   # mini-batch accumulation for variance reduction
_accum_count = [0]

def ceqprop_step(x: torch.Tensor, target: int, q_free: list):
    global _accum_count

    if _accum_count[0] == 0:
        optimizer.zero_grad()

    q_pos = solve(x, beta=+BETA, target=target, state0=q_free)
    q_neg = solve(x, beta=-BETA, target=target, state0=q_free)

    layers_pos = [x] + q_pos
    layers_neg = [x] + q_neg
    scale = 1.0 / (2.0 * BETA * ACCUMULATE)

    # C-EP Hebbian update: ΔW_l ∝ h_{l-1}^+ h_l^+T - h_{l-1}^- h_l^-T
    with torch.no_grad():
        for l in range(n_layers):
            pre_pos, post_pos = layers_pos[l], q_pos[l]
            pre_neg, post_neg = layers_neg[l], q_neg[l]

            # Outer product gradient
            dW = scale * (torch.outer(pre_pos, post_pos) -
                          torch.outer(pre_neg, post_neg))
            db = scale * (post_pos - post_neg)

            if W[l].grad is None:
                W[l].grad = -dW    # negative: we want to minimize energy
                b[l].grad = -db
            else:
                W[l].grad -= dW
                b[l].grad -= db

    _accum_count[0] += 1
    if _accum_count[0] >= ACCUMULATE:
        # Clip and step
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        _accum_count[0] = 0

    return F.cross_entropy(
        torch.tensor(q_free[-1]).unsqueeze(0),
        torch.tensor([target])
    ).item()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(ds, max_n=None):
    correct = total = 0
    n = min(len(ds), max_n or len(ds))
    for i in range(n):
        img, lbl = ds[i]
        lbl_i = lbl.item() if hasattr(lbl, 'item') else int(lbl)
        state = solve(img)
        pred  = state[-1].argmax().item()
        correct += (pred == lbl_i)
        total   += 1
    return correct / total * 100


# ── Training loop ─────────────────────────────────────────────────────────────
loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
acc_hist, loss_hist = [], []
best = 0.0

print(f"{'Ep':>3}  {'Loss':>7}  {'Test%':>6}  {'LR':>7}  {'T':>5}")
print("─" * 36)

for epoch in range(EPOCHS):
    t0 = time.time()
    losses = []

    for batch in loader:
        img, lbl = batch
        img, lbl = img.squeeze(0), lbl.squeeze(0)
        lbl_i = lbl.item()

        # Free phase from zeros
        q_free = solve(img)

        # C-EP update
        loss = ceqprop_step(img, lbl_i, q_free)
        losses.append(loss)

    scheduler.step()
    avg_loss = float(np.mean(losses))
    loss_hist.append(avg_loss)

    # Evaluate on 2k test examples (fast)
    acc = evaluate(test_ds, max_n=2000)
    acc_hist.append(acc)
    best = max(best, acc)

    print(f"{epoch:3d}  {avg_loss:7.4f}  {acc:6.1f}  "
          f"{scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nBest: {best:.2f}%")
print("\n── vs baselines ──")
print("  Single-q NOUS (1k):      ~82%  (structural ceiling)")
print("  Logistic regression:     ~92%  (10k), ~92% (60k)")
print("  MLP 784→256→10:          ~97%  (60k)")
print("  Layered EqProp lit:      ~99.56% (60k, convolutional)")
print(f"  NOUS-Layered:            {best:.2f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_hist, color="#9060ff", lw=1.5, marker="o", ms=3)
axes[0].set_title("C-EP Loss"); axes[0].grid(alpha=0.3)
axes[1].plot(acc_hist, color="#60d0a0", lw=2, marker="o", ms=3, label="NOUS-Layered")
axes[1].axhline(99.56, color="gray",   lw=1, ls=":", label="C-EP literature")
axes[1].axhline(97.0,  color="orange", lw=1, ls="--", label="MLP ~97%")
axes[1].axhline(92.0,  color="red",    lw=1, ls="--", label="Logistic ~92%")
axes[1].set_ylim(60, 101); axes[1].legend(fontsize=8)
axes[1].set_title("Test Accuracy"); axes[1].grid(alpha=0.3)
plt.suptitle(f"NOUS-Layered {INPUT_DIM}→{'→'.join(str(d) for d in LAYER_DIMS)} | best={best:.2f}%", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nous_layered_mnist.png", dpi=130)
plt.close()
print(f"Plot: {OUT_DIR}/nous_layered_mnist.png")

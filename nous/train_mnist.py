"""
NOUS Phase 3B — MNIST Benchmark

Standard benchmark: 1k-sample subset first, then full 60k.
Compares EqProp landscape-sculpting against known baselines:
  - Random: 10%
  - Logistic regression: ~92%
  - MLP (target to beat): ~98%

Architecture:
  - Input: 784D flattened MNIST → projected to EMBED_DIM
  - NOUS state: q ∈ R^STATE_DIM (semantic manifold)
  - Output: 10-class decoder

Speed strategy for EqProp on real data:
  - Fast ODE: dt=0.1, n_steps=50 (vs 300 in toy runs)
  - Batch EqProp: one update per sample, shuffled each epoch
  - State warm-start: q0 = previous q* for same class (not zeros)

Run:
  python -m nous.train_mnist              # 1k subset, fast
  python -m nous.train_mnist --full       # full 60k, slow
"""

import argparse
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import datasets, transforms

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp
from nous.annealing import AnnealingScheduler

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Use full 60k training set")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--subset", type=int, default=1000, help="Samples if not --full")
args = parser.parse_args()

N_TRAIN    = None if args.full else args.subset
EPOCHS     = args.epochs
EMBED_DIM  = 64
STATE_DIM  = 64
N_CLASSES  = 10
OUT_DIR    = "nous_output_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

print("NOUS Phase 3B — MNIST Benchmark")
print("─" * 60)
print(f"Mode: {'full 60k' if args.full else f'subset {N_TRAIN}'} | "
      f"Epochs: {EPOCHS} | State: {STATE_DIM}D")

# ── Data ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          transforms.Lambda(lambda x: x.view(-1))])  # flatten to 784

train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)

if N_TRAIN is not None:
    # Stratified subset: equal samples per class
    per_class = N_TRAIN // N_CLASSES
    indices = []
    class_counts = {c: 0 for c in range(N_CLASSES)}
    for i, (_, label) in enumerate(train_ds):
        if class_counts[label] < per_class:
            indices.append(i)
            class_counts[label] += 1
        if len(indices) >= N_TRAIN:
            break
    train_ds = torch.utils.data.Subset(train_ds, indices)

# Fixed test subset for fast eval (1k stratified)
test_indices = []
test_counts = {c: 0 for c in range(N_CLASSES)}
for i, (_, label) in enumerate(test_ds):
    if test_counts[label] < 100:
        test_indices.append(i)
        test_counts[label] += 1
    if len(test_indices) >= 1000:
        break
test_subset = torch.utils.data.Subset(test_ds, test_indices)

print(f"Train: {len(train_ds)} | Test eval: {len(test_subset)}")

# ── Components ────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# Project 784D → EMBED_DIM before handing to NOUS
projector = nn.Linear(784, EMBED_DIM)
nn.init.xavier_uniform_(projector.weight, gain=0.5)
nn.init.zeros_(projector.bias)

E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM,
              hidden=128, depth=3, n_rbf=16)

decoder = nn.Linear(STATE_DIM, N_CLASSES)
nn.init.xavier_uniform_(decoder.weight, gain=0.3)
nn.init.zeros_(decoder.bias)

optimizer = torch.optim.Adam(
    list(projector.parameters()) +
    list(E.parameters()) +
    list(decoder.parameters()),
    lr=1e-3
)

# Fast ODE for real-data scale
solver  = EulerLagrangeSolver(E, dt=0.1, n_steps=50, delta=1e-3)
eqprop  = EquilibriumProp(E, solver, decoder, optimizer,
                           eps=0.3, phi_distance=0.5, phi_curvature=0.1)
annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.0002,
                               beta_max=10.0, alpha_0=1e-3)

# Warm-start cache: one prototype state per class
q_cache = {c: torch.zeros(STATE_DIM) for c in range(N_CLASSES)}

# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(dataset, label="Test"):
    correct, total = 0, 0
    with torch.no_grad():
        for img, lbl in dataset:
            x = projector(img).detach()
            q0 = q_cache[lbl.item() if hasattr(lbl, 'item') else lbl]
            q_star = solver.solve(x, q0)
            pred = decoder(q_star).argmax().item()
            correct += (pred == (lbl.item() if hasattr(lbl, 'item') else lbl))
            total += 1
    acc = correct / total * 100
    print(f"  {label} accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

# ── Training ──────────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

epoch_losses, epoch_accs, morpho_per_epoch = [], [], []

print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Test%':>7}  {'Morpho':>7}  {'β':>6}  {'Time':>6}")
print("─" * 55)

for epoch in range(EPOCHS):
    for pg in optimizer.param_groups:
        pg["lr"] = annealer.alpha()

    losses, morpho_count = [], 0
    t0 = time.time()

    for img, label in train_loader:
        img, label = img.squeeze(0), label.squeeze(0)
        x = projector(img).detach()
        lbl = label.item()

        # Warm-start from class prototype
        q0 = q_cache[lbl].clone()
        loss, morpho, q_free, _ = eqprop.step(x, label, q0_override=q0)

        # Update prototype toward new equilibrium (EMA)
        q_cache[lbl] = 0.9 * q_cache[lbl] + 0.1 * q_free.detach()

        losses.append(loss)
        if morpho:
            morpho_count += 1

    annealer.tick()
    avg_loss = np.mean(losses)
    epoch_losses.append(avg_loss)
    morpho_per_epoch.append(morpho_count)

    acc = evaluate(test_subset)
    epoch_accs.append(acc)
    elapsed = time.time() - t0
    print(f"{epoch:5d}  {avg_loss:8.4f}  {acc:7.1f}  {morpho_count:7d}  "
          f"{annealer.beta():6.2f}  {elapsed:5.1f}s")

# ── Baselines ─────────────────────────────────────────────────────────────────
print("\n── Baselines ──")
print("  Random:              10.0%")
print("  Logistic regression: ~92%")
print("  MLP (2-layer):       ~98%")
print(f"  NOUS Phase 3B:       {epoch_accs[-1]:.1f}%  (epoch {EPOCHS})")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(epoch_losses, color="#9060ff", lw=1.5, marker="o", markersize=4)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg CE Loss")
axes[0].set_title("Training Loss"); axes[0].grid(alpha=0.3)

axes[1].plot(epoch_accs, color="#60d0a0", lw=1.5, marker="o", markersize=4)
axes[1].axhline(92, color="orange", lw=1, ls="--", label="Logistic (~92%)")
axes[1].axhline(98, color="red",    lw=1, ls="--", label="MLP (~98%)")
axes[1].axhline(10, color="gray",   lw=1, ls=":",  label="Random (10%)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Test Accuracy vs Baselines"); axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 100); axes[1].grid(alpha=0.3)

axes[2].bar(range(EPOCHS), morpho_per_epoch, color="#f06060", alpha=0.7)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Morphogenesis Events")
axes[2].set_title("Basin Growth per Epoch"); axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mnist_training.png", dpi=120)
plt.close()

print(f"\nOutputs: {OUT_DIR}/mnist_training.png")

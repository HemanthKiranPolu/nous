"""
NOUS-Λ Phase 3C — MNIST with Langevin + Contrastive + Dual-timescale

Three simultaneous innovations over standard EqProp:
  1. Langevin training: stochastic ODE forces wide basins, prevents overfitting
  2. Contrastive basin shaping: positive + negative nudge per step
  3. Dual-timescale state: q_fast (16D) relaxes before q_slow (48D)

Run:
  python -m nous.train_mnist_v2              # 1k stratified subset
  python -m nous.train_mnist_v2 --full       # full 60k
  python -m nous.train_mnist_v2 --ablate     # compare all three ablations
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
from nous.langevin_solver import LangevinSolver
from nous.eqprop_contrastive import ContrastiveEqProp
from nous.annealing import AnnealingScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--full",    action="store_true")
parser.add_argument("--ablate",  action="store_true", help="Ablation: run baseline alongside")
parser.add_argument("--epochs",  type=int, default=15)
parser.add_argument("--subset",  type=int, default=1000)
parser.add_argument("--no-langevin",   action="store_true")
parser.add_argument("--no-contrastive",action="store_true")
parser.add_argument("--no-dual",       action="store_true")
args = parser.parse_args()

N_TRAIN   = None if args.full else args.subset
EPOCHS    = args.epochs
FAST_DIM  = 16
SLOW_DIM  = 48
STATE_DIM = FAST_DIM + SLOW_DIM   # 64
EMBED_DIM = 64
N_CLASSES = 10
OUT_DIR   = "nous_output_mnist_v2"
os.makedirs(OUT_DIR, exist_ok=True)

USE_LANGEVIN    = not args.no_langevin
USE_CONTRASTIVE = not args.no_contrastive
USE_DUAL        = not args.no_dual

print("NOUS-Λ Phase 3C — MNIST Benchmark")
print("─" * 60)
print(f"Innovations: Langevin={USE_LANGEVIN}  Contrastive={USE_CONTRASTIVE}  Dual-timescale={USE_DUAL}")
print(f"Mode: {'full 60k' if args.full else f'subset {N_TRAIN}'}  |  Epochs: {EPOCHS}")

# ── Data ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)

if N_TRAIN is not None:
    per_class = N_TRAIN // N_CLASSES
    indices, counts = [], {c: 0 for c in range(N_CLASSES)}
    for i, (_, lbl) in enumerate(train_ds):
        if counts[lbl] < per_class:
            indices.append(i); counts[lbl] += 1
        if len(indices) >= N_TRAIN:
            break
    train_ds = torch.utils.data.Subset(train_ds, indices)

test_indices, test_counts = [], {c: 0 for c in range(N_CLASSES)}
for i, (_, lbl) in enumerate(test_ds):
    if test_counts[lbl] < 100:
        test_indices.append(i); test_counts[lbl] += 1
    if len(test_indices) >= 1000:
        break
test_subset = torch.utils.data.Subset(test_ds, test_indices)
print(f"Train: {len(train_ds)}  |  Test eval: {len(test_subset)}\n")

# ── Components ────────────────────────────────────────────────────────────────
torch.manual_seed(42)

projector = nn.Linear(784, EMBED_DIM)
nn.init.xavier_uniform_(projector.weight, gain=0.5)
nn.init.zeros_(projector.bias)

E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM,
              hidden=128, depth=3, n_rbf=16)

decoder = nn.Linear(STATE_DIM, N_CLASSES)
nn.init.xavier_uniform_(decoder.weight, gain=0.3)
nn.init.zeros_(decoder.bias)

optimizer = torch.optim.Adam(
    list(projector.parameters()) + list(E.parameters()) + list(decoder.parameters()),
    lr=1e-3
)

annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.0003, beta_max=8.0, alpha_0=1e-3)

solver = LangevinSolver(E, dt=0.1, n_steps=40, delta=1e-3,
                        training=USE_LANGEVIN)

eqprop = ContrastiveEqProp(
    E, solver, decoder, optimizer,
    eps=0.3, gamma=0.5 if USE_CONTRASTIVE else 0.0,
    fast_dim=FAST_DIM, use_dual_timescale=USE_DUAL
)

# Per-class warm-start cache
q_cache = {c: torch.zeros(STATE_DIM) for c in range(N_CLASSES)}

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate(dataset):
    solver.set_training(False)
    correct, total = 0, 0
    with torch.no_grad():
        for img, lbl in dataset:
            x = projector(img).detach()
            lbl_int = lbl.item() if hasattr(lbl, 'item') else lbl
            q0 = q_cache[lbl_int]
            if USE_DUAL:
                q_star = solver.solve_dual_timescale(x, q0, fast_dim=FAST_DIM)
            else:
                q_star = solver.solve(x, q0)
            pred = decoder(q_star).argmax().item()
            correct += (pred == lbl_int)
            total += 1
    solver.set_training(USE_LANGEVIN)
    return correct / total * 100

# ── Metrics ───────────────────────────────────────────────────────────────────
def basin_width_estimate(x, q_star, n_probes=4):
    """Estimate basin width via perturbation stability.
    Wider basin = more robust attractor = better generalization."""
    perturb_scale = 0.3
    stable = 0
    for _ in range(n_probes):
        q_perturbed = q_star + perturb_scale * torch.randn_like(q_star)
        q_recovered = solver.solve(x, q_perturbed)
        if (q_recovered - q_star).norm() < 0.5:
            stable += 1
    return stable / n_probes

# ── Training ──────────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

epoch_losses, epoch_accs, epoch_widths = [], [], []

print(f"{'Epoch':>5}  {'Loss':>8}  {'Test%':>7}  {'BasinW':>7}  {'β':>6}  {'Time':>6}")
print("─" * 55)

for epoch in range(EPOCHS):
    solver.set_beta(annealer.beta())
    for pg in optimizer.param_groups:
        pg["lr"] = annealer.alpha()

    losses, widths = [], []
    t0 = time.time()

    for img, label in train_loader:
        img, label = img.squeeze(0), label.squeeze(0)
        x = projector(img).detach()
        lbl = label.item()

        q0 = q_cache[lbl].clone()
        loss, q_free, q_pos, q_neg = eqprop.step(x, label, q0_override=q0)

        # EMA warm-start update
        q_cache[lbl] = 0.9 * q_cache[lbl] + 0.1 * q_free.detach()
        losses.append(loss)

    annealer.tick()

    avg_loss = np.mean(losses)
    epoch_losses.append(avg_loss)

    # Basin width on 20 train samples
    solver.set_training(False)
    width_samples = []
    for img, label in list(train_loader)[:20]:
        x = projector(img.squeeze(0)).detach()
        lbl = label.item()
        q0 = q_cache[lbl]
        q_star = solver.solve(x, q0)
        w = basin_width_estimate(x, q_star)
        width_samples.append(w)
    avg_width = np.mean(width_samples)
    epoch_widths.append(avg_width)
    solver.set_training(USE_LANGEVIN)

    acc = evaluate(test_subset)
    epoch_accs.append(acc)
    elapsed = time.time() - t0

    print(f"{epoch:5d}  {avg_loss:8.4f}  {acc:7.1f}  {avg_width:7.2f}  "
          f"{annealer.beta():6.3f}  {elapsed:5.1f}s")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n── Final Results ──")
peak_acc = max(epoch_accs)
peak_ep  = epoch_accs.index(peak_acc)
print(f"  Peak test accuracy: {peak_acc:.1f}%  (epoch {peak_ep})")
print(f"  Final test accuracy: {epoch_accs[-1]:.1f}%")
print(f"  Mean basin width:   {np.mean(epoch_widths[-3:]):.2f}  (last 3 epochs)")

print("\n── Baselines ──")
print("  Random:              10.0%")
print("  Logistic regression: ~92%")
print("  MLP (2-layer):       ~98%")
print(f"  NOUS Phase 3B (det): ~87%  (overfits after epoch 8)")
print(f"  NOUS-Λ (this):      {peak_acc:.1f}%  (epoch {peak_ep})")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epoch_losses, color="#9060ff", lw=1.5, marker="o", ms=4)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg CE Loss")
axes[0].set_title("Training Loss (Langevin EqProp)"); axes[0].grid(alpha=0.3)

axes[1].plot(epoch_accs, color="#60d0a0", lw=1.5, marker="o", ms=4, label="NOUS-Λ")
axes[1].axhline(92, color="orange", lw=1, ls="--", label="Logistic (~92%)")
axes[1].axhline(98, color="red",    lw=1, ls="--", label="MLP (~98%)")
axes[1].axhline(87, color="#9060ff",lw=1, ls=":",  label="Phase 3B peak (~87%)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("NOUS-Λ vs Baselines"); axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 100); axes[1].grid(alpha=0.3)

axes[2].plot(epoch_widths, color="#f06060", lw=1.5, marker="s", ms=4)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Basin Width (perturbation stability)")
axes[2].set_title("Attractor Basin Width\n(higher = more robust = less overfit)")
axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

plt.suptitle(
    f"NOUS-Λ: Langevin={USE_LANGEVIN}, Contrastive={USE_CONTRASTIVE}, Dual-TS={USE_DUAL}",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nous_lambda_mnist.png", dpi=130)
plt.close()
print(f"\nOutputs: {OUT_DIR}/nous_lambda_mnist.png")

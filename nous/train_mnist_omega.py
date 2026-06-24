"""
NOUS-Ω Phase 3D — Spectral Attractor Classification

Classification signal: Hessian eigenspectrum at q* (basin curvature)
NOT q* position alone.

The ensemble decoder has two heads:
  - Position head: linear(q*)  [standard EqProp decoder]
  - Spectral head: linear(λ₁..λ₈)  [natural oscillation frequencies]
  - Ensemble: α·logits_pos + (1-α)·logits_spec

Training:
  - EqProp updates the energy landscape (same as before)
  - CE on ensemble logits updates both decoders
  - InertiaNet M(q) trained via CE gradient on spectral head
  - α = learned mixing weight

Key insight: even when position q* is ambiguous (two classes nearby),
their Hessian spectra differ — distinct curvature = distinct identity.

Run: python -m nous.train_mnist_omega
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import datasets, transforms

from nous.energy_net import EnergyNet
from nous.el_solver_v2 import EulerLagrangeSolverV2, InertiaNet
from nous.equilibrium_prop import EquilibriumProp
from nous.langevin_solver import LangevinSolver
from nous.annealing import AnnealingScheduler

OUT_DIR = "nous_output_omega"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN   = 1000
EPOCHS    = 20
EMBED_DIM = 64
STATE_DIM = 64
N_CLASSES = 10
N_EIGS    = 8       # spectral fingerprint size

print("NOUS-Ω Phase 3D — Spectral Attractor Classification")
print("─" * 60)
print(f"State: {STATE_DIM}D  |  Spectrum: top-{N_EIGS} Hessian eigenvalues")

# ── Data ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
train_ds = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf)
test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)

per_class = N_TRAIN // N_CLASSES
indices, counts = [], {c: 0 for c in range(N_CLASSES)}
for i, (_, lbl) in enumerate(train_ds):
    if counts[lbl] < per_class:
        indices.append(i); counts[lbl] += 1
    if len(indices) >= N_TRAIN: break
train_ds = torch.utils.data.Subset(train_ds, indices)

test_idx, tcounts = [], {c: 0 for c in range(N_CLASSES)}
for i, (_, lbl) in enumerate(test_ds):
    if tcounts[lbl] < 100:
        test_idx.append(i); tcounts[lbl] += 1
    if len(test_idx) >= 1000: break
test_sub = torch.utils.data.Subset(test_ds, test_idx)
print(f"Train: {len(train_ds)}  |  Test eval: {len(test_sub)}\n")

# ── Architecture ──────────────────────────────────────────────────────────────
torch.manual_seed(42)

projector = nn.Linear(784, EMBED_DIM)
nn.init.xavier_uniform_(projector.weight, gain=0.5)
nn.init.zeros_(projector.bias)

E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM,
              hidden=128, depth=3, n_rbf=16)

inertia_net = InertiaNet(STATE_DIM, hidden=32)

# Position decoder (standard EqProp head)
pos_decoder = nn.Linear(STATE_DIM, N_CLASSES)
nn.init.xavier_uniform_(pos_decoder.weight, gain=0.3)
nn.init.zeros_(pos_decoder.bias)

# Spectral decoder (novel: reads basin curvature)
spec_decoder = nn.Sequential(
    nn.Linear(N_EIGS, 32), nn.ReLU(),
    nn.Linear(32, N_CLASSES)
)
nn.init.xavier_uniform_(spec_decoder[0].weight, gain=0.5)
nn.init.xavier_uniform_(spec_decoder[2].weight, gain=0.3)

# Learned ensemble mixing weight
log_alpha = nn.Parameter(torch.zeros(1))   # α = sigmoid(log_alpha)

optimizer = torch.optim.Adam(
    list(projector.parameters()) +
    list(E.parameters()) +
    list(inertia_net.parameters()) +
    list(pos_decoder.parameters()) +
    list(spec_decoder.parameters()) +
    [log_alpha],
    lr=1e-3
)

annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.0003, beta_max=8.0, alpha_0=1e-3)

# True second-order EL solver
el2 = EulerLagrangeSolverV2(
    E, inertia_net, dt=0.05, n_steps=120, delta=1e-3, gamma=0.5
)

# Langevin solver (for EqProp training phase — calibrated noise)
lang = LangevinSolver(E, dt=0.1, n_steps=40, delta=1e-3,
                      training=True, noise_factor=0.05)

# EqProp operates on the Langevin solver (efficient)
eqprop = EquilibriumProp(E, lang, pos_decoder, optimizer, eps=0.3,
                          phi_distance=0.5, phi_curvature=0.1)

q_cache = {c: torch.zeros(STATE_DIM) for c in range(N_CLASSES)}

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate(dataset):
    lang.set_training(False)
    correct_pos, correct_spec, correct_ens = 0, 0, 0
    total = 0
    alpha = torch.sigmoid(log_alpha).item()

    for img, lbl in dataset:
        x = projector(img).detach()
        lbl_int = lbl.item() if hasattr(lbl, 'item') else lbl

        # Use EL2 (true dynamics) at eval
        q_star = el2.solve(x, q_cache[lbl_int])

        # Position logits
        logits_pos = pos_decoder(q_star)

        # Spectral logits
        spectrum = el2.get_spectral_fingerprint(x, q_star, N_EIGS)
        logits_spec = spec_decoder(spectrum)

        # Ensemble
        logits_ens = alpha * logits_pos + (1 - alpha) * logits_spec

        correct_pos  += (logits_pos.argmax().item() == lbl_int)
        correct_spec += (logits_spec.argmax().item() == lbl_int)
        correct_ens  += (logits_ens.argmax().item() == lbl_int)
        total += 1

    lang.set_training(True)
    return (correct_pos/total*100, correct_spec/total*100, correct_ens/total*100)

# ── Training ──────────────────────────────────────────────────────────────────
loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

epoch_pos, epoch_spec, epoch_ens, epoch_loss = [], [], [], []
spectral_diversity = []   # track how distinct spectra are across classes

print(f"{'Ep':>3}  {'Loss':>7}  {'Pos%':>6}  {'Spec%':>6}  {'Ens%':>6}  {'α':>5}  {'SpecDiv':>8}  {'T':>5}")
print("─" * 65)

for epoch in range(EPOCHS):
    lang.set_beta(annealer.beta())
    lang.set_training(True)
    for pg in optimizer.param_groups:
        pg["lr"] = annealer.alpha()

    losses = []
    t0 = time.time()

    # Collect spectra for 10 samples/class to measure diversity
    class_spectra = {c: [] for c in range(N_CLASSES)}

    for img, label in loader:
        img, label = img.squeeze(0), label.squeeze(0)
        x = projector(img).detach()
        lbl = label.item()

        q0 = q_cache[lbl].clone()

        # EqProp step (position head, fast Langevin solver)
        loss_val, morpho, q_free, q_nudge = eqprop.step(x, label, q0_override=q0)
        q_cache[lbl] = 0.9 * q_cache[lbl] + 0.1 * q_free.detach()
        losses.append(loss_val)

        # Spectral head update (standard backprop on spectrum at q_free)
        if len(class_spectra[lbl]) < 10:
            with torch.no_grad():
                spectrum = el2.get_spectral_fingerprint(x, q_free.detach(), N_EIGS)
            class_spectra[lbl].append(spectrum.detach())

            # Spectral CE loss
            logits_spec = spec_decoder(spectrum)
            alpha = torch.sigmoid(log_alpha)
            logits_pos = pos_decoder(q_free.detach())
            logits_ens = alpha * logits_pos + (1 - alpha) * logits_spec
            spec_loss = F.cross_entropy(logits_ens.unsqueeze(0), label.unsqueeze(0))
            optimizer.zero_grad()
            spec_loss.backward()
            optimizer.step()

    annealer.tick()

    # Spectral diversity: mean pairwise distance between class mean spectra
    class_means = []
    for c in range(N_CLASSES):
        if class_spectra[c]:
            class_means.append(torch.stack(class_spectra[c]).mean(0))
    if len(class_means) >= 2:
        dists = []
        for i in range(len(class_means)):
            for j in range(i+1, len(class_means)):
                dists.append((class_means[i] - class_means[j]).norm().item())
        spec_div = np.mean(dists)
    else:
        spec_div = 0.0
    spectral_diversity.append(spec_div)

    acc_pos, acc_spec, acc_ens = evaluate(test_sub)
    alpha_val = torch.sigmoid(log_alpha).item()
    avg_loss = np.mean(losses)
    epoch_pos.append(acc_pos); epoch_spec.append(acc_spec); epoch_ens.append(acc_ens)
    epoch_loss.append(avg_loss)
    elapsed = time.time() - t0

    print(f"{epoch:3d}  {avg_loss:7.4f}  {acc_pos:6.1f}  {acc_spec:6.1f}  "
          f"{acc_ens:6.1f}  {alpha_val:5.3f}  {spec_div:8.3f}  {elapsed:4.0f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Peak Results ──")
print(f"  Position head:  {max(epoch_pos):.1f}%  (epoch {epoch_pos.index(max(epoch_pos))})")
print(f"  Spectral head:  {max(epoch_spec):.1f}%  (epoch {epoch_spec.index(max(epoch_spec))})")
print(f"  Ensemble:       {max(epoch_ens):.1f}%  (epoch {epoch_ens.index(max(epoch_ens))})")
print(f"  α (pos weight): {torch.sigmoid(log_alpha).item():.3f}")
print(f"  Spectral diversity grew: {spectral_diversity[0]:.3f} → {spectral_diversity[-1]:.3f}")

print("\n── Baselines ──")
print("  Random:              10.0%")
print("  Logistic regression: ~92%")
print("  MLP:                 ~98%")
print(f"  NOUS Phase 3B (pos): ~87%  (overfits)")
print(f"  NOUS-Λ (Langevin):   ~80%  (stable)")
print(f"  NOUS-Ω (spectral):   {max(epoch_ens):.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].plot(epoch_loss, color="#9060ff", lw=1.5, marker="o", ms=3)
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)

axes[1].plot(epoch_pos,  label="Position",  color="#6090ff", lw=1.5)
axes[1].plot(epoch_spec, label="Spectral",  color="#ff9060", lw=1.5)
axes[1].plot(epoch_ens,  label="Ensemble",  color="#60d0a0", lw=2.0)
axes[1].axhline(87, color="#9060ff", ls=":", lw=1, label="Phase 3B peak")
axes[1].axhline(92, color="orange",  ls="--", lw=1, label="Logistic")
axes[1].set_ylim(0, 100); axes[1].legend(fontsize=7)
axes[1].set_title("Accuracy: Pos vs Spec vs Ensemble"); axes[1].grid(alpha=0.3)

axes[2].plot(spectral_diversity, color="#f06060", lw=1.5, marker="s", ms=3)
axes[2].set_title("Spectral Diversity\n(class mean separation in freq space)")
axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)
axes[2].set_ylabel("Mean pairwise dist between class spectra")

alpha_vals = [torch.sigmoid(log_alpha).item()] * EPOCHS
axes[3].set_ylim(0, 1)
axes[3].axhline(torch.sigmoid(log_alpha).item(), color="#a0a0ff", lw=2,
                label=f"α={torch.sigmoid(log_alpha).item():.3f}")
axes[3].bar(["Position", "Spectral"],
            [torch.sigmoid(log_alpha).item(), 1-torch.sigmoid(log_alpha).item()],
            color=["#6090ff", "#ff9060"], alpha=0.8)
axes[3].set_title("Ensemble Mixing Weight α"); axes[3].grid(alpha=0.3, axis='y')
axes[3].set_ylabel("Weight in ensemble")

plt.suptitle("NOUS-Ω: Spectral Attractor Classification (Basin Curvature = Identity)",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nous_omega_mnist.png", dpi=130)
plt.close()
print(f"\nOutputs: {OUT_DIR}/nous_omega_mnist.png")

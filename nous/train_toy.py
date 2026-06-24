"""
NOUS Phase 1 Prototype — XOR in 2D Semantic Space

Validates:
  1. Input-clamped energy E(x,q) creates distinct equilibria per input
  2. EqProp sculpts the landscape without backprop through the ODE
  3. Dual morphogenesis trigger fires when no basin exists
  4. Thermodynamic annealing deepens correct basins

Run: python -m nous.train_toy
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp
from nous.annealing import AnnealingScheduler

OUT_DIR = "nous_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -- XOR dataset: 2-bit input, 1-bit XOR output --
XOR_X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
XOR_Y = torch.tensor([0, 1, 1, 0])
LABELS = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]

INPUT_DIM = 2
STATE_DIM = 2   # 2D so we can visualize V(q)
N_CLASSES = 2
STEPS = 5000

# -- Components --
torch.manual_seed(0)
E = EnergyNet(input_dim=INPUT_DIM, state_dim=STATE_DIM, hidden=64, depth=3)
decoder = nn.Linear(STATE_DIM, N_CLASSES)
nn.init.xavier_uniform_(decoder.weight)

# Guided initialization: W_in with diagonal structure guarantees linear separability.
# Equilibrium condition: ∂V/∂q = W_in^T·x, so inputs (0,0),(0,1),(1,0),(1,1)
# produce equilibria at ~ (0,0), (-s,s), (s,s), (0,2s) for W_in=[[s,-s],[s,s]].
# Classes 0={(0,0),(0,2s)}, 1={(-s,s),(s,s)} → separable by horizontal line at y=s.
with torch.no_grad():
    s = 0.6
    E.W_in.weight.copy_(torch.tensor([[s, -s], [s, s]]))
    # Pre-seed 4 of the 8 RBF centers near expected equilibria; rest stay random
    E.mu.data[:4] = torch.tensor([
        [0.0,  0.0],   # q*(0,0) class 0
        [-s,   s  ],   # q*(0,1) class 1
        [ s,   s  ],   # q*(1,0) class 1
        [0.0,  2*s],   # q*(1,1) class 0
    ])

optimizer = torch.optim.Adam(
    list(E.parameters()) + list(decoder.parameters()), lr=3e-3
)

solver = EulerLagrangeSolver(E, dt=0.05, n_steps=300, delta=5e-4)
eqprop = EquilibriumProp(E, solver, decoder, optimizer, eps=0.5,
                          phi_distance=0.8, phi_curvature=0.05)
annealer = AnnealingScheduler(beta_0=1.0, lambda_=0.0003, beta_max=20.0, alpha_0=3e-3)


def get_equilibria():
    """Compute equilibrium state for each XOR input."""
    eq = {}
    with torch.no_grad():
        for i, x in enumerate(XOR_X):
            q0 = torch.zeros(STATE_DIM)
            q_star = solver.solve(x, q0)
            eq[i] = q_star.numpy()
    return eq


def visualize(step: int, eq: dict):
    res = 60
    grid_x = np.linspace(-3, 3, res)
    grid_y = np.linspace(-3, 3, res)
    XX, YY = np.meshgrid(grid_x, grid_y)
    pts = torch.tensor(np.stack([XX, YY], -1).reshape(-1, 2), dtype=torch.float32)

    with torch.no_grad():
        Z = E.V(pts).reshape(res, res).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: V(q) landscape
    ax = axes[0]
    ctr = ax.contourf(XX, YY, Z, levels=40, cmap="inferno_r", alpha=0.85)
    plt.colorbar(ctr, ax=ax, label="V(q)")
    ax.contour(XX, YY, Z, levels=15, colors="white", linewidths=0.3, alpha=0.4)

    colors = ["#60f0a0", "#a060f0", "#f06060", "#60c0ff"]
    for i, (x, lbl) in enumerate(zip(XOR_X, LABELS)):
        traj = solver.solve_trajectory(x, torch.zeros(STATE_DIM), n_steps=60)
        tn = traj.numpy()
        ax.plot(tn[:, 0], tn[:, 1], color=colors[i], lw=1.5, alpha=0.8)
        ax.scatter(*x.numpy(), color=colors[i], s=80, zorder=5, marker="o")
        if i in eq:
            ax.scatter(*eq[i], color=colors[i], s=140, zorder=6, marker="*",
                       edgecolors="white", linewidths=0.5)
        ax.text(x[0].item() + 0.1, x[1].item() + 0.1, lbl,
                fontsize=8, color=colors[i], fontweight="bold")

    ax.set_title(f"Energy Landscape V(q) — Step {step}\nβ={annealer.beta():.2f}", fontsize=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("q₀"); ax.set_ylabel("q₁")
    ax.grid(alpha=0.15)

    # Right: force field under each input
    ax2 = axes[1]
    i_show = 1  # show force field for input (0,1)→1
    x_show = XOR_X[i_show]
    Fx = np.zeros((res, res))
    Fy = np.zeros((res, res))
    pts_g = pts.clone()
    for idx in range(0, len(pts_g), 50):
        chunk = pts_g[idx:idx+50]
        with torch.no_grad():
            f = E.force(x_show.expand(len(chunk), -1), chunk)
        Fx.flat[idx:idx+50] = f[:, 0].numpy()
        Fy.flat[idx:idx+50] = f[:, 1].numpy()

    speed = np.sqrt(Fx**2 + Fy**2) + 1e-8
    ax2.streamplot(XX, YY, Fx, Fy, color=np.log1p(speed),
                   cmap="plasma", linewidth=0.8, density=1.2, arrowsize=0.8)
    if i_show in eq:
        ax2.scatter(*eq[i_show], color="#a060f0", s=180, zorder=6, marker="*",
                    edgecolors="white", linewidths=1)
    ax2.set_title(f"Force Field for {LABELS[i_show]}", fontsize=10)
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
    ax2.set_xlabel("q₀"); ax2.set_ylabel("q₁")
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/landscape_{step:05d}.png", dpi=110)
    plt.close()


# -- Training --
print("NOUS Phase 1 — XOR  |  EqProp, no backprop through ODE")
print("─" * 60)

loss_hist = []
morpho_events = []

best_loss = float('inf')
best_E_state = None
best_dec_state = None
best_step = 0

for step in range(STEPS):
    idx = step % 4
    x, y = XOR_X[idx], XOR_Y[idx]

    for pg in optimizer.param_groups:
        pg["lr"] = annealer.alpha()

    loss, morpho, q_free, q_nudge = eqprop.step(x, y)
    loss_hist.append(loss)
    annealer.tick()

    if morpho:
        morpho_events.append(step)

    avg = sum(loss_hist[-100:]) / min(100, len(loss_hist))
    if avg < best_loss:
        best_loss = avg
        best_E_state = {k: v.clone() for k, v in E.state_dict().items()}
        best_dec_state = {k: v.clone() for k, v in decoder.state_dict().items()}
        best_step = step

    if step % 300 == 0:
        eq = get_equilibria()
        disp = (q_nudge - q_free).norm().item()
        print(f"Step {step:4d} | loss={avg:.4f} | disp={disp:.4f} | "
              f"morpho={len(morpho_events)} | {annealer.status()}")
        visualize(step, eq)

# Restore best checkpoint before final eval
print(f"\nRestoring best checkpoint (step={best_step}, loss={best_loss:.4f})")
E.load_state_dict(best_E_state)
decoder.load_state_dict(best_dec_state)

# Final
eq = get_equilibria()
visualize(STEPS, eq)

# Accuracy check
print("\n── Final Accuracy ──")
correct = 0
for x, y in zip(XOR_X, XOR_Y):
    q0 = torch.zeros(STATE_DIM)
    q_star = solver.solve(x, q0)
    pred = decoder(q_star).argmax().item()
    label = y.item()
    mark = "✓" if pred == label else "✗"
    print(f"  {mark}  input={x.tolist()}  pred={pred}  target={label}")
    correct += (pred == label)
print(f"\nAccuracy: {correct}/4")
print(f"Morphogenesis events: {len(morpho_events)}")

# Loss curve
window = 80
smooth = np.convolve(loss_hist, np.ones(window)/window, mode="valid")
plt.figure(figsize=(8, 4))
plt.plot(smooth, color="#9060ff", lw=1.5)
plt.xlabel("Step"); plt.ylabel("Loss")
plt.title("NOUS XOR — EqProp Training (No Backprop Through ODE)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/loss_curve.png", dpi=110)
plt.close()
print(f"\nOutputs: {OUT_DIR}/")

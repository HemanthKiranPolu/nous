"""
NOUS Phase 2 — Language
Sentence: "The cat sat on the mat near the big tree"

Architecture:
  - GPT-2 tokenizer → token ids
  - Learned embedding: vocab → R^64
  - NOUS state: q ∈ R^512  (semantic manifold)
  - EqProp at each token step — NO backprop through time, NO unrolled graphs
  - Stateful: q_{t} initializes dynamics at step t+1 (memory via basin)
  - Task: next-token prediction (causal language modeling)

Training protocol:
  One EqProp step per token position per epoch.
  Free phase starts from previous equilibrium q*_{t-1}.
  Nudge phase adds ε·CE(decoder(q), target_token).
  Δθ = (1/ε)[∂E(q*_nudge)/∂θ − ∂E(q*_free)/∂θ]

Run: python -m nous.train_sentence
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

from transformers import GPT2Tokenizer
from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp
from nous.annealing import AnnealingScheduler

OUT_DIR = "nous_output_lang"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Sentence ──────────────────────────────────────────────────────────────────
SENTENCE = "The cat sat on the mat near the big tree"
EMBED_DIM  = 64
STATE_DIM  = 512
N_RBF      = 32
EPOCHS     = 80   # multiple passes over the 9 prediction pairs
PRINT_EVERY = 10

print("NOUS Phase 2 — Language (512D Semantic Manifold)")
print("─" * 60)
print(f"Sentence : {SENTENCE!r}")

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ids = tokenizer.encode(SENTENCE)
VOCAB_SIZE = tokenizer.vocab_size   # 50257

words = [tokenizer.decode([i]) for i in ids]
n_tokens = len(ids)
print(f"Tokens   : {words}")
print(f"IDs      : {ids}")
print(f"Vocab    : {VOCAB_SIZE}  |  Embed: {EMBED_DIM}D  |  State: {STATE_DIM}D")
print()

tokens = torch.tensor(ids)   # (n_tokens,)

# ── Components ────────────────────────────────────────────────────────────────
torch.manual_seed(7)

embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
nn.init.normal_(embedding.weight, std=0.02)

E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM,
              hidden=256, depth=4, n_rbf=N_RBF)

decoder = nn.Linear(STATE_DIM, VOCAB_SIZE)
nn.init.xavier_uniform_(decoder.weight, gain=0.3)
nn.init.zeros_(decoder.bias)

optimizer = torch.optim.Adam(
    list(embedding.parameters()) +
    list(E.parameters()) +
    list(decoder.parameters()),
    lr=1e-3
)

solver = EulerLagrangeSolver(E, dt=0.02, n_steps=150, delta=1e-3)
eqprop  = EquilibriumProp(E, solver, decoder, optimizer,
                           eps=0.3, phi_distance=0.05, phi_curvature=1.2)
annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.002,
                               beta_max=10.0, alpha_0=1e-3)

# ── Training ──────────────────────────────────────────────────────────────────
all_losses   = []    # (epoch, position) → loss
all_morpho   = []    # list of (epoch, position, token_word)
all_states   = []    # one 512-D state per (epoch, position)
epoch_acc    = []

print(f"{'Epoch':>5}  {'Loss':>8}  {'Morpho':>7}  {'β':>6}  Token losses")
print("─" * 70)

for epoch in range(EPOCHS):
    for pg in optimizer.param_groups:
        pg['lr'] = annealer.alpha()

    epoch_losses = []
    q_state = torch.zeros(STATE_DIM)   # state carries memory across tokens

    for t in range(n_tokens - 1):
        x_t      = embedding(tokens[t]).detach()     # current token embedding
        target_t = tokens[t + 1]                     # predict next token

        loss, morpho, q_free, q_nudge = eqprop.step(x_t, target_t,
                                                      q0_override=q_state)
        epoch_losses.append(loss)
        all_losses.append((epoch, t, loss))

        if morpho:
            all_morpho.append((epoch, t, words[t]))

        # carry state: next step starts from current equilibrium
        q_state = q_free.detach()

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            all_states.append((epoch, t, words[t], q_free.detach().clone()))

    annealer.tick()

    avg = np.mean(epoch_losses)
    if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
        token_str = "  ".join(f"{w.strip()[:4]}:{l:.2f}"
                               for w, l in zip(words[:-1], epoch_losses))
        print(f"{epoch:5d}  {avg:8.4f}  {len(all_morpho):7d}  "
              f"{annealer.beta():6.2f}  {token_str}")

print()
print(f"Total morphogenesis events : {len(all_morpho)}")

# ── Visualize ─────────────────────────────────────────────────────────────────

# 1. Loss curve per epoch
ep_avg = {}
for epoch, t, loss in all_losses:
    ep_avg.setdefault(epoch, []).append(loss)
ep_means = [np.mean(ep_avg[e]) for e in sorted(ep_avg)]

plt.figure(figsize=(9, 3))
plt.plot(ep_means, color="#9060ff", lw=1.5)
plt.xlabel("Epoch"); plt.ylabel("Avg CE Loss")
plt.title("NOUS Language — EqProp Next-Token Loss (No BPTT)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/loss_curve.png", dpi=120)
plt.close()

# 2. Per-token loss (last epoch only)
last_epoch = max(ep_avg.keys())
last_losses = ep_avg[last_epoch]
fig, ax = plt.subplots(figsize=(10, 3))
bars = ax.bar(range(len(last_losses)), last_losses,
              color=plt.cm.plasma(np.linspace(0.1, 0.9, len(last_losses))))
ax.set_xticks(range(len(words) - 1))
ax.set_xticklabels([f"{words[t].strip()}→{words[t+1].strip()}"
                    for t in range(len(words) - 1)],
                   rotation=35, ha='right', fontsize=8)
ax.set_ylabel("CE Loss"); ax.set_title(f"Per-Token Loss (Epoch {last_epoch})")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/token_loss.png", dpi=120)
plt.close()

# 3. PCA of 512D states (final epoch)
final_states = [(t, w, q) for (ep, t, w, q) in all_states if ep == last_epoch]
if len(final_states) >= 2:
    Qs = torch.stack([q for _, _, q in final_states]).numpy()
    pca = PCA(n_components=2)
    Qs_2d = pca.fit_transform(Qs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(final_states)))
    for i, (t, w, _) in enumerate(final_states):
        ax.scatter(Qs_2d[i, 0], Qs_2d[i, 1], color=colors[i], s=180, zorder=5)
        ax.annotate(w.strip(), (Qs_2d[i, 0], Qs_2d[i, 1]),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=10, color=colors[i], fontweight='bold')
    if len(Qs_2d) > 1:
        for i in range(len(Qs_2d) - 1):
            ax.annotate("", xy=Qs_2d[i+1], xytext=Qs_2d[i],
                        arrowprops=dict(arrowstyle="->", color='gray',
                                        lw=0.8, alpha=0.5))
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("512D NOUS States — PCA Projection\n"
                 "(Each point = equilibrium attractor for that token context)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/semantics_pca.png", dpi=130)
    plt.close()

# 4. Morphogenesis timeline
if all_morpho:
    fig, ax = plt.subplots(figsize=(10, 4))
    morpho_epochs = [e for e, t, w in all_morpho]
    morpho_pos    = [t for e, t, w in all_morpho]
    sc = ax.scatter(morpho_epochs, morpho_pos,
                    c=morpho_pos, cmap='tab10', s=15, alpha=0.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Token position")
    ax.set_yticks(range(n_tokens - 1))
    ax.set_yticklabels([f"{words[t].strip()}→{words[t+1].strip()}"
                        for t in range(n_tokens - 1)], fontsize=8)
    ax.set_title(f"Morphogenesis Events — {len(all_morpho)} total\n"
                 "(Each dot = new semantic basin carved at that token × epoch)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/morpho_timeline.png", dpi=120)
    plt.close()

# 5. Pairwise semantic distance matrix (final epoch)
if len(final_states) >= 2:
    Qs_t = torch.stack([q for _, _, q in final_states])
    D = torch.cdist(Qs_t, Qs_t).numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [w.strip() for _, w, _ in final_states]
    im = ax.imshow(D, cmap='viridis_r')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="‖q*_i − q*_j‖ (512D)")
    ax.set_title("Semantic Distance Matrix\n(Lower = more similar context representation)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/semantic_distance.png", dpi=120)
    plt.close()

print(f"\nOutputs: {OUT_DIR}/")
print("  loss_curve.png       — per-epoch average loss")
print("  token_loss.png       — per-token loss at final epoch")
print("  semantics_pca.png    — 512D equilibria projected to 2D via PCA")
print("  morpho_timeline.png  — which tokens triggered basin creation")
print("  semantic_distance.png — pairwise distance between semantic states")

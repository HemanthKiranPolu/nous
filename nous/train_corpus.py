"""
NOUS Phase 3A — Corpus Scaling
WikiText-2, 200 sentences, 15 training epochs.

The scientific question: do attractors for the same word cluster
across different sentence contexts?

If "cat" in sentence 1 and "cat" in sentence 84 converge to nearby
attractors — semantics = basin topology is proven.

Metrics:
  - Perplexity per epoch
  - Intra-word attractor variance (low = consistent representation)
  - Inter-word attractor distance (high = discriminative)
  - Morphogenesis event rate per epoch

Run: python -m nous.train_corpus
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from sklearn.decomposition import PCA

from datasets import load_dataset
from transformers import GPT2Tokenizer

from nous.energy_net import EnergyNet
from nous.el_solver import EulerLagrangeSolver
from nous.equilibrium_prop import EquilibriumProp
from nous.annealing import AnnealingScheduler

OUT_DIR = "nous_output_corpus"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM  = 64
STATE_DIM  = 512
N_RBF      = 32
EPOCHS     = 15
N_SENTS    = 200    # sentences from WikiText-2
MIN_TOKENS = 5
MAX_TOKENS = 12     # short sentences → faster ODE convergence
TRACK_TOPK = 12     # most frequent tokens to track attractor consistency

print("NOUS Phase 3A — Corpus Scaling")
print("─" * 60)

# ── Data ──────────────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
VOCAB_SIZE = tokenizer.vocab_size

print(f"Loading WikiText-2 ...", end=" ", flush=True)
raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
sentences = []
for item in raw:
    text = item["text"].strip()
    if not text or text.startswith("="):
        continue
    ids = tokenizer.encode(text)
    if MIN_TOKENS <= len(ids) <= MAX_TOKENS:
        sentences.append(ids)
    if len(sentences) >= N_SENTS:
        break

print(f"{len(sentences)} sentences  |  avg len {np.mean([len(s) for s in sentences]):.1f} tokens")
print(f"Embed: {EMBED_DIM}D  |  State: {STATE_DIM}D  |  RBF: {N_RBF}")

# Find top-K recurring tokens to track
token_counts = defaultdict(int)
for s in sentences:
    for t in s:
        token_counts[t] += 1

# Exclude very common function tokens for interesting semantics
stopids = set(tokenizer.encode(" the " + " a " + " of " + " in " + " and ")[:10])
track_ids = [tid for tid, cnt in sorted(token_counts.items(), key=lambda x: -x[1])
             if tid not in stopids and cnt >= 3][:TRACK_TOPK]
track_words = {tid: tokenizer.decode([tid]).strip() for tid in track_ids}
print(f"Tracking: {list(track_words.values())}")
print()

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
    lr=8e-4
)

solver   = EulerLagrangeSolver(E, dt=0.02, n_steps=120, delta=1e-3)
eqprop   = EquilibriumProp(E, solver, decoder, optimizer,
                            eps=0.3, phi_distance=0.05, phi_curvature=1.2)
annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.003,
                               beta_max=10.0, alpha_0=8e-4)

# ── Training ──────────────────────────────────────────────────────────────────
epoch_losses   = []
epoch_morpho   = []
# word_attractors[token_id] = list of 512-D states seen across all epochs
word_attractors = defaultdict(list)

print(f"{'Epoch':>5}  {'PPL':>8}  {'Morpho':>7}  {'β':>5}  Notes")
print("─" * 55)

for epoch in range(EPOCHS):
    for pg in optimizer.param_groups:
        pg['lr'] = annealer.alpha()

    losses, morpho_count = [], 0
    np.random.shuffle(sentences)   # random sentence order each epoch

    for sent in sentences:
        q_state = torch.zeros(STATE_DIM)

        for t in range(len(sent) - 1):
            x_t      = embedding(torch.tensor(sent[t])).detach()
            target_t = torch.tensor(sent[t + 1])

            loss, morpho, q_free, q_nudge = eqprop.step(
                x_t, target_t, q0_override=q_state)

            losses.append(loss)
            if morpho:
                morpho_count += 1

            q_state = q_free.detach()

            # Record attractor for tracked tokens
            if sent[t] in track_ids:
                word_attractors[sent[t]].append(q_free.detach().clone())

    annealer.tick()

    avg_loss = np.mean(losses)
    ppl = np.exp(min(avg_loss, 20))
    epoch_losses.append(avg_loss)
    epoch_morpho.append(morpho_count)

    note = ""
    if morpho_count == 0:
        note = "← basins stable"
    elif morpho_count > len(sentences) * 5:
        note = "← active carving"

    print(f"{epoch:5d}  {ppl:8.1f}  {morpho_count:7d}  "
          f"{annealer.beta():5.2f}  {note}")

print()

# ── Attractor consistency analysis ────────────────────────────────────────────
print("── Attractor Consistency ──")
consistency = {}   # word → (intra_std, n_obs)
for tid in track_ids:
    states = word_attractors[tid]
    if len(states) < 2:
        continue
    mat = torch.stack(states).numpy()   # (N, 512)
    centroid = mat.mean(0)
    dists = np.linalg.norm(mat - centroid, axis=1)
    intra_std = dists.std()
    consistency[tid] = (intra_std, len(states))
    word = track_words[tid]
    print(f"  {word:12s}  n={len(states):3d}  intra_std={intra_std:.4f}")

print()

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Perplexity curve
ppls = [np.exp(min(l, 20)) for l in epoch_losses]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(ppls, color="#9060ff", lw=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Perplexity")
ax.set_title("NOUS Corpus — Perplexity (No BPTT)")
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.bar(range(EPOCHS), epoch_morpho, color="#1D9E75", alpha=0.8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Morphogenesis events")
ax2.set_title("Basin Carving Events per Epoch")
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

# 2. PCA of all tracked-word attractors (colored by word)
all_states, all_labels, all_colors = [], [], []
color_map = plt.cm.tab10(np.linspace(0, 1, len(track_ids)))
for ci, tid in enumerate(track_ids):
    states = word_attractors[tid]
    if len(states) < 2:
        continue
    for s in states:
        all_states.append(s.numpy())
        all_labels.append(track_words[tid])
        all_colors.append(color_map[ci])

if len(all_states) >= 4:
    pca = PCA(n_components=2)
    pts = pca.fit_transform(np.stack(all_states))
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, tid in enumerate(track_ids):
        mask = [i for i, l in enumerate(all_labels) if l == track_words[tid]]
        if not mask:
            continue
        xs = pts[mask, 0]; ys = pts[mask, 1]
        ax.scatter(xs, ys, color=color_map[ci], s=30, alpha=0.55, label=track_words[tid])
        cx, cy = xs.mean(), ys.mean()
        ax.scatter(cx, cy, color=color_map[ci], s=180, marker="*",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(track_words[tid], (cx, cy), xytext=(5, 4),
                    textcoords="offset points", fontsize=9,
                    color=color_map[ci], fontweight="bold")

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("512D Attractor Clusters — Same Word Across Different Sentences\n"
                 "(Stars = centroids. Tight clusters = consistent semantic representation.)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/attractor_clusters.png", dpi=130)
    plt.close()

# 3. Intra-word consistency bar chart
if consistency:
    words_sorted = sorted(consistency.items(), key=lambda x: x[1][0])
    labels  = [track_words[tid] for tid, _ in words_sorted]
    stds    = [v[0] for _, v in words_sorted]
    counts  = [v[1] for _, v in words_sorted]
    colors_ = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, stds, color=colors_)
    ax.set_ylabel("Intra-word attractor std (512D)")
    ax.set_title("Attractor Consistency per Word\n"
                 "(Lower = same word always lands in similar basin regardless of sentence context)")
    ax2b = ax.twinx()
    ax2b.plot(labels, counts, "o--", color="#3060a0", ms=5, lw=1, alpha=0.7)
    ax2b.set_ylabel("Occurrences", color="#3060a0")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/word_consistency.png", dpi=120)
    plt.close()

print(f"Outputs: {OUT_DIR}/")
print("  training_curves.png    — perplexity + morphogenesis events per epoch")
print("  attractor_clusters.png — 512D PCA of same-word attractors across sentences")
print("  word_consistency.png   — intra-word attractor variance (the key metric)")

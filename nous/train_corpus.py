"""
NOUS Phase 3A — Corpus Scaling
WikiText-2, 40 sentences, 5 training epochs.

Scientific question: do attractors for the same word cluster
across different sentence contexts?

Metrics:
  - Perplexity per epoch
  - Intra-word attractor variance (low = consistent representation)
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

EMBED_DIM  = 64
STATE_DIM  = 512
N_RBF      = 32
EPOCHS     = 5
N_SENTS    = 40
MIN_TOKENS = 5
MAX_TOKENS = 9
TRACK_TOPK = 8

print("NOUS Phase 3A — Corpus Scaling")
print("─" * 60)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
VOCAB_SIZE = tokenizer.vocab_size

print("Loading WikiText-2 ...", end=" ", flush=True)
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

print(f"{len(sentences)} sentences  avg_len={np.mean([len(s) for s in sentences]):.1f}")

token_counts = defaultdict(int)
for s in sentences:
    for t in s:
        token_counts[t] += 1

skip = set(tokenizer.encode(" the a of in and is was to"))
track_ids = [tid for tid, cnt in sorted(token_counts.items(), key=lambda x: -x[1])
             if tid not in skip and cnt >= 2][:TRACK_TOPK]
track_words = {tid: tokenizer.decode([tid]).strip() for tid in track_ids}
print(f"Tracking: {list(track_words.values())}")
print()

torch.manual_seed(7)
embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
nn.init.normal_(embedding.weight, std=0.02)
E = EnergyNet(input_dim=EMBED_DIM, state_dim=STATE_DIM, hidden=256, depth=4, n_rbf=N_RBF)
decoder = nn.Linear(STATE_DIM, VOCAB_SIZE)
nn.init.xavier_uniform_(decoder.weight, gain=0.3)
nn.init.zeros_(decoder.bias)

optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(E.parameters()) + list(decoder.parameters()),
    lr=8e-4)

solver   = EulerLagrangeSolver(E, dt=0.02, n_steps=60, delta=1e-3)
eqprop   = EquilibriumProp(E, solver, decoder, optimizer,
                            eps=0.3, phi_distance=0.05, phi_curvature=1.2)
annealer = AnnealingScheduler(beta_0=0.5, lambda_=0.003, beta_max=10.0, alpha_0=8e-4)

epoch_losses, epoch_morpho = [], []
word_attractors = defaultdict(list)

print(f"{'Epoch':>5}  {'PPL':>8}  {'Morpho':>7}  {'β':>5}")
print("─" * 35)

for epoch in range(EPOCHS):
    for pg in optimizer.param_groups:
        pg['lr'] = annealer.alpha()
    losses, morpho_count = [], 0
    order = list(range(len(sentences)))
    np.random.shuffle(order)
    for si in order:
        sent = sentences[si]
        q_state = torch.zeros(STATE_DIM)
        for t in range(len(sent) - 1):
            x_t      = embedding(torch.tensor(sent[t])).detach()
            target_t = torch.tensor(sent[t + 1])
            loss, morpho, q_free, q_nudge = eqprop.step(x_t, target_t, q0_override=q_state)
            losses.append(loss)
            if morpho:
                morpho_count += 1
            q_state = q_free.detach()
            if sent[t] in track_ids:
                word_attractors[sent[t]].append(q_free.detach().clone())
    annealer.tick()
    avg = np.mean(losses)
    ppl = np.exp(min(avg, 20))
    epoch_losses.append(avg)
    epoch_morpho.append(morpho_count)
    print(f"{epoch:5d}  {ppl:8.1f}  {morpho_count:7d}  {annealer.beta():5.2f}", flush=True)

print()
print("── Attractor Consistency ──")
consistency = {}
for tid in track_ids:
    states = word_attractors[tid]
    if len(states) < 2:
        continue
    mat = torch.stack(states).numpy()
    centroid = mat.mean(0)
    dists = np.linalg.norm(mat - centroid, axis=1)
    intra_std = dists.std()
    consistency[tid] = (intra_std, len(states))
    print(f"  {track_words[tid]:12s}  n={len(states):3d}  intra_std={intra_std:.4f}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ppls = [np.exp(min(l,20)) for l in epoch_losses]
axes[0].plot(ppls, color="#9060ff", lw=2, marker="o", ms=5)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Perplexity")
axes[0].set_title("NOUS Corpus — Perplexity (No BPTT)"); axes[0].grid(alpha=0.3)
axes[1].bar(range(EPOCHS), epoch_morpho, color="#1D9E75", alpha=0.8)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Morphogenesis events")
axes[1].set_title("Basin Carving per Epoch"); axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120); plt.close()

all_states, all_labels, all_colors = [], [], []
cmap = plt.cm.tab10(np.linspace(0,1,len(track_ids)))
for ci, tid in enumerate(track_ids):
    for s in word_attractors[tid]:
        all_states.append(s.numpy()); all_labels.append(track_words[tid]); all_colors.append(cmap[ci])

if len(all_states) >= 4:
    pca = PCA(n_components=2)
    pts = pca.fit_transform(np.stack(all_states))
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, tid in enumerate(track_ids):
        mask = [i for i,l in enumerate(all_labels) if l==track_words[tid]]
        if not mask: continue
        xs=pts[mask,0]; ys=pts[mask,1]
        ax.scatter(xs, ys, color=cmap[ci], s=25, alpha=0.5, label=track_words[tid])
        ax.scatter(xs.mean(), ys.mean(), color=cmap[ci], s=160, marker="*",
                   edgecolors="white", lw=0.8, zorder=5)
        ax.annotate(track_words[tid], (xs.mean(),ys.mean()), xytext=(5,4),
                    textcoords="offset points", fontsize=9, color=cmap[ci], fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("512D Attractor Clusters — Same Word Across Different Sentences\n"
                 "(Stars=centroids. Tight clusters = consistent semantic representation.)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/attractor_clusters.png", dpi=130); plt.close()

if consistency:
    ws = sorted(consistency.items(), key=lambda x: x[1][0])
    labels=[track_words[t] for t,_ in ws]; stds=[v[0] for _,v in ws]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(labels, stds, color=plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(labels))))
    ax.set_ylabel("Intra-word attractor std"); ax.grid(axis='y', alpha=0.3)
    ax.set_title("Attractor Consistency — lower = same word lands in similar basin across sentences")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/word_consistency.png", dpi=120); plt.close()

print(f"\nOutputs: {OUT_DIR}/")

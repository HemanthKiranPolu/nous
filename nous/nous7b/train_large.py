"""
NOUS large-scale training: streaming data + MPS/GPU + gradient accumulation.

Supports:
  wikitext103  — 103M tokens, good for overnight runs
  c4           — ~300B tokens, serious pretraining
  wikitext2    — 2M tokens, quick validation

Batching strategy:
  - B sentences run in parallel (independent → safe to batch)
  - Within each sentence, tokens are processed sequentially (stateful carry)
  - Gradient accumulation over `accum_steps` batches before optimizer.step()

Run (local MPS, NOUS-Small, WikiText-103):
    python -m nous.nous7b.train_large --config small --dataset wikitext103 \\
           --batch_size 8 --accum_steps 4 --max_tokens 10_000_000

Run (GPU cluster, NOUS-7B):
    python -m nous.nous7b.train_large --config 7b --dataset c4 \\
           --batch_size 32 --accum_steps 8 --max_tokens 1_000_000_000
"""
import argparse
import math
import os
import time
from pathlib import Path
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset, IterableDataset

from nous.nous7b.config import NOUS_SMALL, NOUS_7B, NOUSConfig
from nous.nous7b.energy_net_7b import NOUSEnergyNet7B
from nous.nous7b.ode_batched import BatchedEulerODE
from nous.annealing import AnnealingScheduler


# ── Device ────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Data pipeline (streaming, no RAM limit) ───────────────────────────────

def sentence_stream(dataset_name: str, tokenizer, min_len=4, max_len=64):
    """Infinite stream of tokenized sentences."""
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = (item["text"] for item in ds)
    elif dataset_name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = (item["text"] for item in ds)
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = (item["text"] for item in ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    for text in texts:
        text = text.strip()
        if not text or text.startswith("="): continue
        ids = tokenizer.encode(text)
        # Chunk long docs into max_len windows
        for start in range(0, max(1, len(ids) - min_len), max_len // 2):
            chunk = ids[start: start + max_len]
            if len(chunk) >= min_len:
                yield chunk


def batch_stream(stream, batch_size: int):
    """Group sentences into batches of size B."""
    buf = []
    for sent in stream:
        buf.append(sent)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


# ── EqProp step (batched, accumulate) ────────────────────────────────────

def eqprop_batch_step(model, ode: BatchedEulerODE, opt, batch_sents,
                       tokenizer, device, cfg: NOUSConfig,
                       accumulate: bool = False):
    """
    Process one batch of sentences.
    Runs tokens sequentially within each sentence (stateful carry).
    Accumulates EqProp grads across the batch.

    Returns: (mean_loss, morpho_count, total_tokens)
    """
    B     = len(batch_sents)
    dtype = torch.float16 if str(device) == "mps" else torch.float32

    q_states = torch.zeros(B, cfg.state_dim, device=device, dtype=dtype)
    embedding = model.embedding

    losses, morpho_count, total_tokens = [], 0, 0

    # Find the max sentence length in this batch
    max_t = max(len(s) for s in batch_sents) - 1

    # Pad sentences to equal length for vectorized ops
    padded   = [s + [0] * (max_t + 1 - len(s)) for s in batch_sents]
    tok_in   = torch.tensor([[s[t] for s in padded] for t in range(max_t)],
                              device=device)   # (T, B)
    tok_out  = torch.tensor([[s[t+1] for s in padded] for t in range(max_t)],
                              device=device)   # (T, B)
    lens     = torch.tensor([len(s) - 1 for s in batch_sents])  # (B,)

    if not accumulate:
        opt.zero_grad()

    for t in range(max_t):
        # Mask out padding
        active = (t < lens)
        if not active.any(): continue
        b_idx = active.nonzero(as_tuple=True)[0]
        B_t   = len(b_idx)

        x_t = torch.stack([
            embedding(tok_in[t, b]).detach()
            for b in b_idx
        ]).to(dtype)                          # (B_t, embed_dim)
        tgt_t = tok_out[t, b_idx]            # (B_t,)
        q0_t  = q_states[b_idx]              # (B_t, state_dim)

        # Free phase
        q_free = ode.solve_batch(x_t, q0_t, n_steps=cfg.n_steps_free)

        # Compute loss at free equilibrium
        with torch.no_grad():
            step_loss = torch.stack([
                F.cross_entropy(model.decode(q_free[i]).unsqueeze(0),
                                tgt_t[i:i+1])
                for i in range(B_t)
            ]).mean().item()
        losses.append(step_loss)
        total_tokens += B_t

        # Nudged phase
        extra_fns = [
            (lambda i: lambda q: cfg.eps * F.cross_entropy(
                model.decode(q).unsqueeze(0), tgt_t[i:i+1]))(i)
            for i in range(B_t)
        ]
        q_nudge = ode.solve_batch(x_t, q0_t, n_steps=cfg.n_steps_nudge,
                                   extra_fns=extra_fns)

        # EqProp gradient (accumulate over active samples)
        scale = 1.0 / (B_t * cfg.eps)
        for i in range(B_t):
            gf = _param_grad(model, x_t[i], q_free[i])
            gn = _param_grad(model, x_t[i], q_nudge[i])
            for name, param in model.named_parameters():
                if param.requires_grad and name in gf:
                    g = scale * (gn[name] - gf[name])
                    if param.grad is None:
                        param.grad = g
                    else:
                        param.grad.add_(g)

        # Decoder CE on nudged equilibrium (standard BP)
        ce_sum = sum(
            F.cross_entropy(model.decode(q_nudge[i]).unsqueeze(0), tgt_t[i:i+1])
            for i in range(B_t)
        ) / B_t
        ce_sum.backward()

        # Carry state
        for j, b in enumerate(b_idx.tolist()):
            q_states[b] = q_free[j].to(dtype)

        # Morphogenesis
        for i in range(B_t):
            dist = (q_nudge[i] - q_free[i]).norm().item()
            lmin = model.stochastic_min_curvature(x_t[i], q_free[i],
                                                   n_probes=4).item()
            if dist > cfg.phi_distance and lmin < cfg.phi_curvature:
                model.add_rbf_center(q_free[i])
                morpho_count += 1

    return np.mean(losses) if losses else float("nan"), morpho_count, total_tokens


def _param_grad(model, x_embed, q_star):
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    E = model(x_embed, q_star.detach())
    E.backward()
    return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for n, p in model.named_parameters()}


# ── Training loop ─────────────────────────────────────────────────────────

def train(cfg: NOUSConfig, args):
    device  = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = NOUSConfig(**{**cfg.__dict__, "vocab_size": tokenizer.vocab_size})

    dtype = torch.float16 if str(device) == "mps" else torch.float32
    model = NOUSEnergyNet7B(cfg).to(device=device, dtype=dtype)
    model.parameter_summary()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    ode = BatchedEulerODE(model, cfg)

    stream = batch_stream(
        sentence_stream(args.dataset, tokenizer),
        batch_size=args.batch_size
    )

    global_step  = 0
    total_tokens = 0
    loss_hist    = []
    morpho_total = 0
    t0           = time.time()

    print(f"\n{'Step':>7}  {'PPL':>9}  {'Morpho':>7}  {'Mtok':>7}  {'tok/s':>7}")
    print("─" * 55)

    for batch_idx, batch_sents in enumerate(stream):
        accumulate = (batch_idx % args.accum_steps != 0)
        if not accumulate:
            opt.zero_grad()

        loss, morpho, ntok = eqprop_batch_step(
            model, ode, opt, batch_sents, tokenizer,
            device, cfg, accumulate=accumulate
        )

        loss_hist.append(loss)
        morpho_total += morpho
        total_tokens += ntok

        if not accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1

        if global_step % 100 == 0 and not accumulate:
            avg   = np.mean(loss_hist[-200:])
            ppl   = math.exp(min(avg, 20))
            tps   = total_tokens / (time.time() - t0)
            mtok  = total_tokens / 1e6
            print(f"{global_step:7d}  {ppl:9.1f}  {morpho_total:7d}"
                  f"  {mtok:7.2f}  {tps:7.0f}", flush=True)

        if global_step % 1000 == 0 and global_step > 0 and not accumulate:
            ckpt = out_dir / f"step_{global_step:07d}.pt"
            torch.save({"step": global_step, "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "loss": float(np.mean(loss_hist[-500:]))}, ckpt)
            print(f"  → {ckpt}")

        if total_tokens >= args.max_tokens:
            print(f"\nReached {args.max_tokens/1e6:.1f}M tokens — done.")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="small", choices=["small", "7b"])
    p.add_argument("--dataset",     default="wikitext103",
                   choices=["wikitext2", "wikitext103", "c4"])
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--max_tokens",  type=int, default=5_000_000)
    p.add_argument("--out_dir",     default="nous_output_large")
    args = p.parse_args()

    cfg = NOUS_SMALL if args.config == "small" else NOUS_7B
    train(cfg, args)


if __name__ == "__main__":
    main()

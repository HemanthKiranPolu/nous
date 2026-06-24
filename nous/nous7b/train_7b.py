"""
NOUS-7B training script.

Target: 8×A100 80GB, bf16, ~7B parameters.
Local dev: use NOUS_SMALL config (125M, fp32, single GPU/CPU).

Run (small, local):
    python -m nous.nous7b.train_7b --config small --dataset wikitext2

Run (7B, cluster):
    torchrun --nproc_per_node=8 -m nous.nous7b.train_7b --config 7b --dataset pile

Design decisions:
  - Gradient accumulation over `grad_accum_steps` tokens before optimizer.step()
  - Stateful carry: q_{t} initializes ODE for token t+1 (no BPTT)
  - AnnealingScheduler coupled to optimizer lr
  - Checkpoint: saves model + optimizer + annealer state every 1000 steps
"""
import argparse
import os
import time
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from transformers import LlamaTokenizer, GPT2Tokenizer
from datasets import load_dataset

from nous.nous7b.config import NOUS_SMALL, NOUS_7B, NOUSConfig
from nous.nous7b.energy_net_7b import NOUSEnergyNet7B
from nous.nous7b.eqprop_7b import EqProp7B
from nous.annealing import AnnealingScheduler


def build_model_and_opt(cfg: NOUSConfig, device: torch.device):
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    model = NOUSEnergyNet7B(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return model, opt


def load_data(dataset_name: str, tokenizer, max_sentences: int, cfg: NOUSConfig):
    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        sentences = []
        for item in raw:
            text = item["text"].strip()
            if not text or text.startswith("="):
                continue
            ids = tokenizer.encode(text)
            if 4 <= len(ids) <= 64:
                sentences.append(ids)
            if len(sentences) >= max_sentences:
                break
    elif dataset_name == "pile":
        raw = load_dataset("EleutherAI/pile", split="train", streaming=True)
        sentences = []
        for item in raw:
            ids = tokenizer.encode(item["text"])
            for start in range(0, max(0, len(ids) - 64), 32):
                sentences.append(ids[start:start + 64])
                if len(sentences) >= max_sentences:
                    break
            if len(sentences) >= max_sentences:
                break
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return sentences


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(cfg: NOUSConfig, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    except Exception:
        print("LLaMA tokenizer unavailable, falling back to GPT-2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        cfg = NOUSConfig(**{**cfg.__dict__, "vocab_size": tokenizer.vocab_size})

    print(f"Vocab size: {cfg.vocab_size}")

    # Model
    model, opt = build_model_and_opt(cfg, device)
    model.parameter_summary()

    eqprop   = EqProp7B(model, opt, cfg)
    annealer = AnnealingScheduler(
        beta_0=cfg.beta_0,
        lambda_=cfg.lambda_annealing,
        beta_max=cfg.beta_max,
        alpha_0=cfg.lr,
    )

    # Data
    print(f"Loading {args.dataset}...")
    sentences = load_data(args.dataset, tokenizer,
                          max_sentences=args.max_sentences, cfg=cfg)
    print(f"  {len(sentences)} sentences")

    # Training loop
    global_step  = 0
    total_tokens = sum(len(s) - 1 for s in sentences) * args.epochs
    morpho_count = 0
    loss_hist    = []
    t0 = time.time()

    print(f"\n{'Step':>7}  {'PPL':>8}  {'Morpho':>7}  {'β':>6}  {'tok/s':>7}")
    print("─" * 50)

    for epoch in range(args.epochs):
        order = np.random.permutation(len(sentences))
        for si in order:
            sent = sentences[si]
            # Stateful carry across tokens — no BPTT
            q_state = torch.zeros(cfg.state_dim, device=device,
                                  dtype=torch.bfloat16 if cfg.dtype == "bfloat16"
                                  else torch.float32)

            for t in range(len(sent) - 1):
                tok_in  = torch.tensor(sent[t],     device=device)
                tok_out = torch.tensor(sent[t + 1], device=device)

                for pg in opt.param_groups:
                    pg["lr"] = cosine_lr(global_step, warmup=2000,
                                          total=total_tokens,
                                          lr_max=cfg.lr, lr_min=cfg.lr / 10)

                loss, morpho, q_free = eqprop.step(tok_in, tok_out, q0=q_state)
                q_state = q_free

                loss_hist.append(loss)
                if morpho:
                    morpho_count += 1
                    model.add_rbf_center(q_free)

                annealer.tick()
                global_step += 1

                if global_step % 500 == 0:
                    avg_loss = np.mean(loss_hist[-200:])
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - t0
                    tps = global_step / elapsed
                    print(f"{global_step:7d}  {ppl:8.1f}  {morpho_count:7d}"
                          f"  {annealer.beta():6.2f}  {tps:7.1f}")

                if global_step % 5000 == 0:
                    ckpt = out_dir / f"step_{global_step:07d}.pt"
                    torch.save({
                        "step":    global_step,
                        "model":   model.state_dict(),
                        "opt":     opt.state_dict(),
                        "beta":    annealer.beta(),
                        "loss":    float(np.mean(loss_hist[-500:])),
                    }, ckpt)
                    print(f"  → checkpoint: {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="small",    choices=["small", "7b"])
    parser.add_argument("--dataset",  default="wikitext2", choices=["wikitext2", "pile"])
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--max_sentences", type=int, default=500)
    parser.add_argument("--out_dir",  default="nous_output_7b")
    args = parser.parse_args()

    cfg = NOUS_SMALL if args.config == "small" else NOUS_7B
    train(cfg, args)


if __name__ == "__main__":
    main()

"""
NOUS ablation runner.

Each ablation varies one axis vs the NOUS-Small baseline.
All runs: 100 WikiText-2 sentences, 3 epochs, same seed.
Metric: final PPL (lower) + morphogenesis events (higher = more active).

Run: python -m nous.nous7b.ablation
"""
import math
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional
from datasets import load_dataset
from transformers import GPT2Tokenizer

from nous.nous7b.config import NOUS_SMALL, NOUSConfig
from nous.nous7b.energy_net_7b import NOUSEnergyNet7B
from nous.nous7b.eqprop_7b import EqProp7B, EulerODE
from nous.annealing import AnnealingScheduler


# ── Shared data (load once) ────────────────────────────────────────────────

_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_VOCAB = _tokenizer.vocab_size

def _load_sentences(n=100):
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for item in raw:
        t = item["text"].strip()
        if not t or t.startswith("="): continue
        ids = _tokenizer.encode(t)
        if 4 <= len(ids) <= 32:
            out.append(ids)
        if len(out) >= n: break
    return out

SENTENCES = _load_sentences(100)


# ── Single training run ────────────────────────────────────────────────────

@dataclass
class RunConfig:
    name: str
    cfg: NOUSConfig
    stateful: bool = True          # carry q across tokens
    morpho_on: bool = True         # enable morphogenesis add_rbf_center
    bowl_only: bool = False        # ablate: remove blocks and RBF
    no_rbf: bool = False           # ablate: remove RBF component


def run_one(rc: RunConfig, epochs: int = 3, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    cfg = replace(rc.cfg, vocab_size=_VOCAB)

    model = NOUSEnergyNet7B(cfg)

    # Component ablations: patch forward methods
    if rc.bowl_only:
        model.V_blocks = lambda q: torch.zeros(())
        model.V_rbf    = lambda q: torch.zeros(())
    elif rc.no_rbf:
        model.V_rbf    = lambda q: torch.zeros(())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1,
                             betas=(0.9, 0.95))
    eqprop = EqProp7B(model, opt, cfg)
    ode    = EulerODE(model, cfg)

    loss_hist, morpho_count = [], 0
    t0 = time.time()

    for epoch in range(epochs):
        order = np.random.permutation(len(SENTENCES))
        for si in order:
            sent = SENTENCES[si]
            q_state = torch.zeros(cfg.state_dim)
            for t in range(len(sent) - 1):
                tok_in  = torch.tensor(sent[t])
                tok_out = torch.tensor(sent[t + 1])
                for pg in opt.param_groups: pg["lr"] = cfg.lr
                q0 = q_state if rc.stateful else torch.zeros(cfg.state_dim)
                loss, morpho, q_free = eqprop.step(tok_in, tok_out, q0=q0)
                loss_hist.append(loss)
                if morpho and rc.morpho_on:
                    morpho_count += 1
                    model.add_rbf_center(q_free)
                q_state = q_free

    final_ppl = math.exp(min(np.mean(loss_hist[-200:]), 20))
    return {
        "name":   rc.name,
        "ppl":    final_ppl,
        "morpho": morpho_count,
        "secs":   time.time() - t0,
    }


# ── Ablation grid ──────────────────────────────────────────────────────────

def make_runs() -> List[RunConfig]:
    base = NOUS_SMALL
    runs = []

    # Baseline
    runs.append(RunConfig("baseline", base))

    # A1: Energy blocks
    for n in [2, 4, 16, 32]:
        runs.append(RunConfig(f"blocks={n}",
            replace(base, n_energy_blocks=n,
                    ffn_hidden=max(256, int(base.ffn_hidden * (n / base.n_energy_blocks) ** 0.5)))))

    # A2: ODE steps
    for s in [10, 20, 40, 160]:
        runs.append(RunConfig(f"ode_steps={s}",
            replace(base, n_steps_free=s, n_steps_nudge=s)))

    # A3: EqProp nudge strength ε
    for eps in [0.05, 0.1, 0.5, 1.0]:
        runs.append(RunConfig(f"eps={eps}",
            replace(base, eps=eps)))

    # A4: Morphogenesis off
    runs.append(RunConfig("morpho=off", base, morpho_on=False))

    # A5: Stateful carry off (re-initialize q=0 each token)
    runs.append(RunConfig("stateful=off", base, stateful=False))

    # A6: V(q) components
    runs.append(RunConfig("V=bowl_only",    base, bowl_only=True))
    runs.append(RunConfig("V=bowl+MLP",     base, no_rbf=True))

    # A7: phi_distance (morphogenesis sensitivity)
    for phi in [0.01, 0.1, 0.5, 1.0]:
        runs.append(RunConfig(f"phi_dist={phi}",
            replace(base, phi_distance=phi)))

    # A8: Learning rate
    for lr in [1e-4, 1e-3, 3e-3]:
        runs.append(RunConfig(f"lr={lr}", replace(base, lr=lr)))

    return runs


# ── Runner ─────────────────────────────────────────────────────────────────

def main():
    runs = make_runs()
    print(f"NOUS Ablation Study  |  {len(runs)} runs  |  100 sents × 3 epochs each")
    print(f"{'Run':25s}  {'PPL':>9}  {'Morpho':>7}  {'Secs':>6}")
    print("─" * 60)

    results = []
    for rc in runs:
        try:
            r = run_one(rc)
            tag = " ← baseline" if rc.name == "baseline" else ""
            print(f"{r['name']:25s}  {r['ppl']:9.1f}  {r['morpho']:7d}  {r['secs']:6.0f}{tag}",
                  flush=True)
            results.append(r)
        except Exception as e:
            print(f"{rc.name:25s}  ERROR: {e}", flush=True)

    # Summary: rank by PPL
    results.sort(key=lambda x: x["ppl"])
    print("\n── Top 5 configurations by PPL ──")
    for r in results[:5]:
        print(f"  {r['name']:25s}  PPL={r['ppl']:.1f}  morpho={r['morpho']}")

    baseline_ppl = next(r["ppl"] for r in results if r["name"] == "baseline")
    print(f"\n── Worst vs baseline (PPL {baseline_ppl:.1f}) ──")
    for r in sorted(results, key=lambda x: -x["ppl"])[:3]:
        delta = r["ppl"] / baseline_ppl
        print(f"  {r['name']:25s}  PPL={r['ppl']:.1f}  ({delta:.1f}× worse)")


if __name__ == "__main__":
    main()

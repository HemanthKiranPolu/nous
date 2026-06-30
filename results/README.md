# Results — Compositional Generalization (SCAN-mini)

Reproduce:

```bash
./scripts/run_compgen_seq.sh          # 32-seed sweep → scan_mini_32s.json
./scripts/run_compgen_seq.sh --quick  # 2-seed CI sanity (no claim)
```

## Headline — `scan_mini_32s.json`

Source: [`nous/train_compgen_seq.py`](../nous/train_compgen_seq.py).
Task: 6 verbs × {once, twice, thrice}; output = the verb's action symbol
emitted `count` times, read off the NOUS equilibrium by a shared,
position-conditioned decoder. Held-out verbs {3,4,5} are trained ONLY at
`once`, so the held-out pairs require emitting a **known symbol in a novel
position** (1, 2) — the SCAN "jump twice" generalization, in miniature.

Evaluation is **train-fit-gated**: only seeds where ALL THREE models reach
train ≥ 0.99 enter the comparison, so a held-out miss is never confounded by
a model that failed to fit the training set. Both baselines use the **identical**
shared decoder, so any gap reflects the representation's binding, not decoder
capacity. The transformer is a 2-layer self-attention encoder over the natural
2-token input (learned embeddings, its best-case encoding).

32 seeds, gate kept 20/32:

| held-out exact-match | per-seed mean ± std | pooled | 95% Wilson CI |
| -------------------- | ------------------- | ------ | ------------- |
| **NOUS**             | 57.5 ± 29.1 %       | 57.5 % | [48.6, 66.0]  |
| **MLP**              | 12.5 ± 12.8 %       | 12.5 % | [7.7, 19.6]   |
| **Transformer**      |  9.2 ± 20.1 %       |  9.2 % | [5.2, 15.7]   |

- Δ = **+45.0 pp** vs MLP, **+48.3 pp** vs Transformer. NOUS's 95% CI clears the
  upper bound of both baselines → a real effect vs both at this scale.
- The transformer's ~9 % is the canonical SCAN behaviour: it fits train but
  collapses on a known symbol in a novel output position.
- Train-fit failures (gated out): NOUS 3, MLP 7, Transformer 5 — the gate
  removes each model's worst seeds, so it is conservative *for* NOUS.

### Honest scope (do not overclaim)

- This is a **toy** (6 verbs, fixed grammar), not the real SCAN benchmark.
- High seed variance (NOUS ±29.1). The mean is solid; individual runs swing.
- Pooled CI treats the 6 held-out points/seed as independent; they are
  correlated within a seed, so the true interval is slightly wider — but the
  per-seed std already bounds the spread and the gap is large regardless.
- This does **not** reproduce any "100% vs 5.9%" figure; no such script exists
  in this repo.

## Negative control — `compgen_toy.json`

Source: [`nous/train_compgen.py`](../nous/train_compgen.py). Same idea but with a
one-hot⊕one-hot **input** and independent per-slot heads. That hands every model
the factorization for free, so even the MLP solves the held-out pair (8/8
seeds). Kept as a documented reminder: **comp-gen difficulty must live in the
output binding** to be informative.

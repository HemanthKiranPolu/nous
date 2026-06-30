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

Evaluation is **train-fit-gated**: only seeds where BOTH models reach
train ≥ 0.99 enter the comparison, so a held-out miss is never confounded by
a model that failed to fit the training set.

| held-out exact-match | per-seed mean ± std | pooled | 95% Wilson CI |
| -------------------- | ------------------- | ------ | ------------- |
| **NOUS**             | 57.6 ± 30.5 %       | 57.6 % | [49.0, 65.7]  |
| **MLP (baseline)**   | 12.1 ± 15.2 %       | 12.1 % | [7.6, 18.8]   |

- Δ = **+45.5 pp**; 95% CIs do not overlap → a real effect at this scale.
- Gate kept 22/32 seeds (MLP failed train-fit 8×, NOUS 3× → the gate is
  conservative *for* NOUS, not flattering).
- The MLP uses the **identical** decoder, so the gap reflects the energy
  manifold's binding, not decoder capacity.

### Honest scope (do not overclaim)

- This is a **toy** (6 verbs, fixed grammar), not the real SCAN benchmark.
- High seed variance (NOUS ±30.5). The mean is solid; individual runs swing.
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

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

---

# Results — Continual Learning (operator stream)

Reproduce:

```bash
python -m nous.train_continual_ops --seeds 5   # → results/continual_ops.json
python -m nous.train_continual_ops --selfcheck # asserts the core invariant + retention
```

## Headline — `continual_ops.json`

Source: [`nous/train_continual_ops.py`](../nous/train_continual_ops.py).
Task: three operations on ℤ₅ — `add`, `mul`, `sub` (output = `(a op b) mod 5`,
5-way). They are **streamed in phases** — learn `add` fully, then `mul`, then
`sub`, with **no retrain from scratch**. After each phase we measure accuracy on
every op seen so far. This tests *retention under a non-stationary stream*, the
frozen-weights / catastrophic-forgetting question — **not** generalization
(SCAN-mini above covers that).

- **NOUS-CLS** learns by surprise-gated, op-local **basin allocation**: a correct
  prediction only deepens the winning attractor (spec §4.2 Hebbian
  consolidation); a wrong one carves a new labeled RBF basin. Crucially, an
  update during op *K* never touches an op≠*K* basin — old skills are untouched
  dimensions of `V(q)`.
- **MLP** is the canonical baseline: a shared-weight net trained on each op *in
  full to ~100%*, then the next — its best case, so any old-op drop is pure
  cross-task interference.

5 seeds, `add → mul → sub`:

| after the full stream        | `add` retention | all-ops final | forgetting (add) |
| ---------------------------- | --------------- | ------------- | ---------------- |
| **NOUS-CLS** (op-local)      | **100.0 %**     | 99.7 %        | **+0.0 pp**      |
| **MLP** (shared weights)     | 20.0 % (chance) | 45.9 %        | **+80.0 pp**     |

- The MLP fits `add` to 100 %, then collapses to chance (20 %) on it after
  learning the later ops — textbook catastrophic forgetting. NOUS-CLS retains
  every op perfectly because its updates are physically local.

### Honest scope (do not overclaim)

- This is a **toy** (mod-5, 25 pairs/op) and tests **retention**, not
  generalization. With a frozen random `W_in`, distinct inputs land at distinct
  positions, so NOUS ends up with ≈ one basin per input region (~72 basins for
  75 inputs): it **memorizes** each op in local structure. The claim is only
  that *local* memory growth does not overwrite old ops — which the
  shared-weight MLP demonstrably does.
- **Unbounded null, reported honestly:** with no budget cap an op-**blind**
  ablation (basins not scoped by op) *also* retains perfectly (+0.0 pp). When
  memory grows freely and inputs are separable, locality is never stressed —
  forgetting then only appears in fixed-capacity models (the MLP). To stress
  locality you must cap the budget — see next.

## Capacity pressure — `continual_ops_capped.json`

Reproduce:

```bash
python -m nous.train_continual_ops --seeds 5 --budget 60   # → results/continual_ops_capped.json
```

Cap the **total** basin budget so surprises must *evict* (LRU) an existing
basin. Both variants get the **same total capacity**; the only difference is
partitioning:

- **op-aware** reserves `budget / n_ops` slots per op — evicts only *within* the
  current op, so old ops are frozen.
- **op-blind** pools all slots — a `mul`/`sub` surprise LRU-evicts the oldest
  basins, which are `add`'s. That is the interference.

5 seeds, `add → mul → sub`, budget 60 (20 slots/op):

| after the full stream        | `add` peak → final | forgetting (add) |
| ---------------------------- | ------------------ | ---------------- |
| **op-aware** (reserved slots)| 80.0 % → 80.8 %    | **−0.8 pp**      |
| **op-blind** (shared pool)   | 100.0 % → 37.6 %   | **+62.4 pp**     |
| **MLP** (shared weights)     | 100.0 % → 20.0 %   | **+80.0 pp**     |

This is the stability–plasticity tradeoff made concrete. Op-blind (and the MLP)
fit each new op to 100 % but overwrite the old ones; op-aware forgets essentially
nothing because its updates are confined to the current op's reserved region of
`V(q)`. Same capacity, opposite retention: **locality, not capacity, is what
prevents catastrophic forgetting.**

### Per-op budget vs peak (op-aware, 5 seeds)

Op-aware peak is bounded by its per-op slots; raise the budget and peak rises to
100 % with forgetting staying ≈ 0. Once a per-op budget covers the op's 25 pairs
(25/op), op-aware is both perfect *and* stable, while op-blind — pooling the same
total — still LRU-evicts old ops:

| budget | per-op | op-aware peak → final (forget) | op-blind forget |
| ------ | ------ | ------------------------------ | --------------- |
| 30     | 10     | 40.8 % → 42.4 %  (−1.6 pp)      | +77.6 pp        |
| 45     | 15     | 60.0 % → 61.6 %  (−1.6 pp)      | +82.4 pp        |
| 60     | 20     | 80.0 % → 80.8 %  (−0.8 pp)      | +62.4 pp        |
| 75     | 25     | 100.0 % → 100.0 % ( 0.0 pp)     | +12.0 pp        |

Reproduce any row with `--budget <N>`. The op-aware column is the knob the
question asked for: **more per-op capacity buys peak accuracy without trading
away retention** — the reserved-slot locality holds at every budget.

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

### Task-free routing — step 1: remove the op label (`taskfree`)

The op-aware result hands locality to the model: the op id names the region and
slots are reserved per op. Step 1 removes that crutch. The `taskfree` learner
gets **no op label at all** — one shared pool (exactly like op-blind), and the
*only* change from op-blind is the eviction rule: when the budget is full it
reuses the **spatially nearest** basin instead of the oldest (LRU). If that alone
restores retention, locality was *discovered* from the input geometry, not told.

5 seeds, `add → mul → sub`:

| budget | op-blind (LRU) forget | **task-free (geom) peak→final, forget** | op-aware forget |
| ------ | --------------------- | --------------------------------------- | --------------- |
| 30     | +77.6 pp              | 100.0 % → 40.8 %,  **+59.2 pp**          | −1.6 pp         |
| 60     | +62.4 pp              | 100.0 % → 90.4 %,  **+9.6 pp**           | −0.8 pp         |

- At budget 60, task-free keeps op-blind's **full 100 % peak** *and* retains
  90 % — forgetting collapses from +62 pp to **+9.6 pp** with the label removed,
  just by evicting geometrically. Locality genuinely emerges from `x`.
- But it does **not** fully match op-aware (≈0 forget), and at the tight
  budget 30 it still forgets +59 pp. Cause: with a **frozen random `W_in`** the
  op is only 3 of 13 input dims, so `add`/`mul`/`sub` regions partly overlap in
  state space (mean cross-op distance 7.1 vs within-op 5.6, but min cross-op 4.5
  ≈ within-op) — geometry can route only as well as the representation separates
  the tasks. That gap is exactly what **step 2 (unfreeze `W_in`)** should close.

### Unfreezing the encoder — step 2: plastic `W_in` (`taskfree_plastic`)

Step 1 left `W_in` a frozen random projection, and its imperfect task separation
was the ceiling on geometric routing. Step 2 makes `W_in` **trainable** and shapes
it with an embedding-space contrastive loss (pull raw `e = W_in·x` toward its
correct-label basin prototype, push from others) — no differentiation through the
relaxation. Prediction: better separation → task-free tightens toward op-aware.

**That is not what happens.** A plastic shared encoder is neutral at best and
harmful as it moves (5 seeds, budget 60):

| encoder LR | task-free (plastic) forget | note |
| ---------- | -------------------------- | ---- |
| frozen     | +9.6 pp                    | step-1 baseline |
| 0.02       | +10.4 pp                   | ≈ neutral |
| 0.1        | +16 pp                     | drift starts |
| 0.3        | +19 pp                     | |
| 0.6        | +35 pp                     | drift dominates |

Two compounding reasons, both instructive:

- **The memory co-adapts, so there is no separation pressure.** Basins are placed
  *at* the current embeddings, so the contrastive loss is already near-zero — the
  encoder gradient is tiny and `W_in` barely moves (why LR 0.02 ≈ frozen).
- **The ops share the output label space.** `label 3` is produced by `add` *and*
  `mul`, so a label-driven encoder objective pulls different ops' inputs
  *together*, not apart — it cannot create task-separated regions, and once the
  encoder does move (higher LR) it only **drifts** old ops' inputs off their
  basins → *more* forgetting.

Takeaway: **you cannot separate tasks by training a shared encoder on a shared
label signal.** Representation plasticity has to be *task-conditioned* — which is
precisely what step 3 (per-region / LoRA-style adapters) supplies: new-task
updates that do not move old tasks' representations.

### Task-conditioned plasticity — step 3: per-region adapters (`adapter`)

Step 2 failed because the plastic encoder was *shared*. Step 3 makes it **local**:
base `W_in` stays frozen, and each discovered **region** owns a small low-rank
adapter `ΔW_r = B_r·A_r`. Regions are found by the *same* label-free geometry as
the memory — route the base embedding to the nearest region centroid, spawn a new
one past `region_radius`. Training updates **only the routed region's adapter**,
so learning a new task can move at most that region's embeddings. This is the
neuroscience recipe: pattern separation (routing) + sparse local updates + old
regions left intact.

Budget 60, at `adapt_lr = 0.3` — a rate where the *shared* encoder drifts:

| method                              | forgetting | note |
| ----------------------------------- | ---------- | ---- |
| step 1 — frozen (`taskfree`)        | +9.6 pp    | baseline |
| step 2 — shared plastic encoder     | +19 pp     | drift |
| **step 3 — per-region adapters**    | **+9–10 pp** | **drift contained** |

- Localizing the plasticity **removes the step-2 regression**: forgetting drops
  back to the frozen baseline even though the encoder is now trainable. And it is
  robust to region granularity — `region_radius` from 2 to 8 gives ~75 down to
  **~3 regions**, all at ≈+10 pp. At the coarse end the ~3 emergent regions line
  up with the 3 ops: the routing *rediscovers the task structure* from geometry,
  with no label.

**Honest — what step 3 does NOT do:**

- **It contains drift; it does not beat frozen.** Retention returns to the
  step-1 level, no better. The memory co-adapts (basins sit at the embeddings),
  so the adapter — like step 2 — gets a near-zero gradient and has little to
  actually learn.
- **No help under capacity pressure.** At budget 30 the adapter forgets +60–66 pp,
  same as frozen: there the forgetting is *capacity*-bound (basin eviction), a
  different axis that a representation adapter cannot touch.
- **Reading:** task-conditioning is what makes representation plasticity *safe*
  (non-forgetting), which shared plasticity was not — a necessary property. The
  *payoff* (plasticity that actually improves retention) should appear where
  representation learning genuinely matters and forgetting isn't just capacity —
  i.e. a real gradient-trained backbone. That is the transformer + per-region
  LoRA step, now motivated by three toy results instead of a hunch.

### Limitations — what this does NOT show

This is a controlled existence proof that *partitioned* memory beats *shared*
parameters for interference, on a 75-point toy where capacity, separability, and
the eviction rule are all hand-set. It is **not** an unfrozen LLM, and several
loads are bearing that will not survive scale:

- **Locality is only partly discovered.** The op-aware result hands locality to
  the model (op id names the region, slots reserved per op). The `taskfree`
  learner (step 1) removes the label and recovers most of the retention from
  geometry alone (+9.6 pp forgetting at budget 60 vs op-blind's +62) — but only
  *most*, and only with enough capacity, because the frozen `W_in` separates the
  tasks imperfectly. Full task-free routing on an under-separated or truly
  unlabelled task is still open.
- **Representation plasticity doesn't help — and can hurt.** Step 2 unfroze
  `W_in` with a contrastive objective; it did not improve routing and drifts into
  *more* forgetting as it moves, because the ops share the label space so a
  label-driven encoder can't separate tasks (see step 2). The frozen random
  projection remains the best encoder here; genuinely useful, *task-conditioned*
  representation learning is still open (step 3).
- **Retention, not generalization.** With ≈ one basin per input region there is
  no compositional transfer (SCAN-mini above is the generalization probe, not
  this). A growing labeled memory trivially avoids forgetting in the limit — the
  informative comparison is only the *equal-capacity, capped* one.
- **No gradient backbone.** Step 3 adds per-region low-rank adapters (a real
  gradient mechanism) but on a linear encoder over a frozen random projection —
  not a learned feature hierarchy. It says nothing yet about interference where
  it actually costs: per-region LoRA on a trained transformer, on a real task.
  That is the next step, and the toy now predicts *task-conditioning is necessary
  for safe representation plasticity* — the hypothesis that step would test.

Claim ladder: *(shown)* structured memory reduces interference at equal capacity
→ *(untested)* it survives learned, moving representations → *(open)* it helps a
large model learn continually. Each rung is a separate experiment; this repo is
on the first.

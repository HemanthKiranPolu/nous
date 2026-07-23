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

### Evidence-based consolidation (`--noisy`)

Reproduce:

```bash
python -m nous.train_continual_ops --noisy --seeds 5   # → results/continual_ops_noisy.json
```

Growing structure on a *single* surprise is fragile: a mislabeled observation
becomes a permanent basin. Evidence-based consolidation applies the
scientific-replication principle to structure growth — a surprise creates a
**provisional** candidate that accumulates evidence (hit frequency + label
agreement) across repeated observations, and is **consolidated** into a permanent
basin only after ≥ k hits with ≥ 60 % label agreement (majority label wins).
Provisional candidates do not drive prediction, so noise cannot corrupt outputs
before it is filtered. (`ConsolidatingLearner`.)

Learn one op from a stream with **per-observation label noise** (each draw's label
is flipped with prob `p`); basin budget 30, 5 seeds:

| noise `p` | immediate: acc / basins | **consolidate: acc / basins** |
| --------- | ----------------------- | ----------------------------- |
| 0.0       | 1.00 / 25               | 1.00 / 25                     |
| 0.1       | 0.90 / 30               | **1.00 / 25**                 |
| 0.2       | 0.74 / 30               | **0.99 / 25**                 |
| 0.3       | 0.74 / 30               | **0.98 / 24**                 |

- **Immediate spawn** carves a basin for every noise surprise → basin count
  balloons to the budget cap → those spurious basins evict good structure →
  clean-label accuracy collapses to ~0.74.
- **Consolidation** never promotes a one-off: transient noise is out-voted by the
  majority of an input's repeated observations, so only the ~25 true concepts
  become basins and clean accuracy stays ~0.98–1.00. **Zero cost when clean**
  (identical at `p = 0`).

The principle: a single observation is provisional; only consistent, repeated
evidence should change structure. It is the frequency-and-consistency half of
consolidation — a held-out prediction-gain gate is the heavier upgrade.

### Semantic-ID store (`--discovery`)

Consolidation decides *when* a concept is real; the natural next move is to give a
consolidated concept a **permanent semantic ID** — a stable, frozen address — and
route by that id rather than re-deriving a region by similarity every time. Full
pipeline: provisional candidate → evidence → consolidate → **assign a frozen id**.

Tested on a **task-free discovery stream**: 25 concepts are introduced in 5 waves
(5 at a time, no task labels), and every concept recurs — with 20 % per-observation
label noise — in all later waves. `id_store` runs the pipeline (frozen id per
consolidated concept, route to nearest id); `similarity` spawns/updates prototypes
immediately and lets them drift. 5 seeds:

| metric                          | **`id_store`** | `similarity` |
| ------------------------------- | -------------- | ------------ |
| structures for 25 concepts      | **25 ids**     | 87 basins    |
| clean accuracy                  | **99 %**       | 94 %         |
| wave-0 retention (after each wave) | 0.96→1.00→1.00→1.00→1.00 | 0.88→0.88→0.92→0.92→0.96 |

- **Addresses are earned.** The id store assigns **exactly one id per true concept**
  (25 for 25) and stays accurate; drifting similarity proliferates **87** spurious
  structures for the same 25 concepts — noise gets its own prototypes because
  nothing gates promotion.
- **Identity persists.** The earliest concepts' accuracy sits flat at 1.00 once
  their ids are frozen. (Honest: similarity's wave-0 curve *rises* rather than
  collapses — repeated correct observations eventually out-vote noise — so the win
  here is structure efficiency + accuracy + stable ids, not a similarity collapse.)

This is where the arc's threads meet: geometric routing (pattern separation) +
evidence-based consolidation (when to promote) + a frozen semantic id (a stable
address) — provisional memory that earns a permanent handle only once the evidence
is in. Note: on a *frozen* representation a learned/id router cannot beat nearest
prototype on raw accuracy (shown earlier); the id's payoff is stability and earned,
noise-free structure, not higher routing accuracy.

### The defer gate — "I don't know" (`--defer`)

The confidence-routing negative (on the pretrained side) showed *softmax*
confidence is miscalibrated: a wrong expert is overconfident, so it can't flag its
own errors. The fix is to gate on **distance**, not confidence, and route
uncertain inputs to the provisional buffer instead of forcing a placement — the
"librarian's *I don't know*". Gate signal = entropy of `softmax(−dist²/T)` over the
consolidated ids: low when one id dominates (confident), high when the input sits
between ids (ambiguous).

Trained on the clean concepts, then queried with the concepts themselves **and**
with ambiguous probes (midpoints between two ids). 5 seeds:

| query          | hard routing | **defer gate** |
| -------------- | ------------ | -------------- |
| clean concepts | routes all (100 % acc) | **abstains 0 %** (routes all) |
| ambiguous mid-points | confidently assigns all | **abstains 86 %** |

The gate abstains **selectively** — never on clean concepts, on 86 % of the
genuinely ambiguous ones — so it *only* says "I don't know" when it should. That
un-routable 86 % is exactly what the provisional/consolidation machinery is built
to hold: park it, wait for corroborating evidence, mint an id only if it recurs.
The reason this works where the pretrained confidence gate (B) failed: **distance
to a prototype is calibrated; a classifier's softmax is not.** Uncertainty
estimation is a real remaining piece of the puzzle — and the cheap distance/entropy
version already does the load-bearing job.

### The whole loop, end to end (`--librarian`)

Each piece above was validated in isolation; `LibrarianLearner` runs them **all at
once** — task-free surprise-spawn + evidence-based consolidation + a frozen semantic
id per concept + the defer gate — on a single mixed stream: clean concepts recurring
with **15 % label noise**, plus a wave of **novel** concepts introduced halfway.
(The learning-time defer adds a *near* guard so a novel input — high entropy but far
from every id — still flows through to provisional memory; only blends *close to*
known ids are parked.) vs `naive` = immediate spawn, no consolidation, no defer.
5 seeds:

| metric                          | **`librarian`** | `naive` |
| ------------------------------- | --------------- | ------- |
| clean-concept accuracy          | 0.97            | 0.95    |
| novel-concept accuracy          | 0.92            | 1.00    |
| **ids for 25 concepts**         | **24**          | 42      |
| **ambiguous-query defer rate**  | **0.83**        | 0.00    |

The librarian keeps **~one clean id per real concept (24)** while `naive` inflates
to **42** — every noise blip becomes permanent structure. It still learns the real
and novel concepts, and it **defers 83 %** of genuinely ambiguous queries where
`naive` confidently misplaces every one. `naive` edges it on raw accuracy (it
memorises everything, noise included), but at 1.75× the structure and zero
uncertainty handling. That is the whole thesis in one run: **modular, evidence-gated,
addressed, calibrated memory — clean structure and honest "I don't know," not just
a bigger pile of parameters.**

### The librarian on a real LLM (`train_llm_librarian.py`)

The same loop, ported off the toy's random projection onto frozen **`gpt2`** (125M)
embeddings on a DBpedia class stream (clean + 15 % noise + novel-halfway). The
librarian is a memory policy over prototypes, so it runs directly on the embeddings
— no LoRA, no relaxation. `naive` = one prototype per observation. 3 seeds:

| metric               | **`librarian`** | `naive` |
| -------------------- | --------------- | ------- |
| clean accuracy       | **0.85**        | 0.74    |
| novel accuracy       | **0.81**        | 0.77    |
| ids                  | **448**         | 2880    |
| ambiguous defer      | **1.00**        | 0.00    |

On real, noisy data the librarian wins on **every** axis — including *accuracy*
(0.85 vs 0.74), unlike the well-separated toy where `naive` matched it. The reason:
`naive`'s noise-prototypes actively *mislead* nearest-neighbour, while evidence
gating throws them out. It uses **6.4× fewer ids** (448 vs 2880). The 448 (~32 per
class) is honest — real classes are **multi-modal**, so a concept consolidates into
several prototypes, not the toy's one; still an order of magnitude leaner than
one-per-observation, and the multi-prototype-per-concept regime is the realistic one.

**Bonus finding (representation, again):** raw `gpt2` mean-pooled embeddings are
badly **anisotropic** — every input collapses into a tiny cone (within-class ≈
cross-class distance ≈ 0.05), which flattens prototype memory (0 consolidations). A
one-line **whitening** (mean-centre + per-dim standardise + renormalise, from train
stats) lifts DBpedia nearest-class-mean routing **0.52 → 0.91** — near a real
sentence embedder. Decoder-LM hidden states need that fix to be usable as concept
addresses; encoders (MiniLM) are already isotropic (`--model minilm`).

**On code models — old vs new (`train_code_librarian.py`).** The same librarian on
a **programming-language** continual stream (CodeSearchNet: Python → Java → Go →
JS → PHP, PHP novel, 15 % noise), frozen code-model embeddings (mean-pool +
whitening). The "diff" = an older code model vs a newer code LLM, identical task:

| code model                         | routing | librarian clean / novel / ids | naive clean / ids |
| ---------------------------------- | ------- | ----------------------------- | ----------------- |
| **CodeBERT** (2020, 125M encoder)  | 0.92    | 0.96 / 0.93 / 230             | 0.85 / 2700       |
| **Qwen2.5-Coder-0.5B** (2024 LLM)  | **1.00**| **0.99 / 1.00** / 273         | 0.81 / 2700       |

- **The diff is real**: the newer code LLM separates languages *perfectly*
  (routing 0.92 → 1.00), which lifts librarian retention (clean 0.96 → 0.99, novel
  0.93 → 1.00) — the representation lever, on code.
- On **both** models the librarian beats `naive` decisively: clean 0.96–0.99 vs
  0.81–0.85 (naive's noise-prototypes mislead), with **~250 ids vs 2700** (≈10×
  leaner) and full ambiguity deferral.
- Bigger coders (Qwen2.5-Coder-7B etc.) need a GPU — the Colab version is
  [`notebooks/librarian_code_colab.ipynb`](../notebooks/librarian_code_colab.ipynb).

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
→ *(shown, small)* it survives a real gradient-trained transformer (below) →
*(open)* it helps a *large* model on a real task. The first two rungs are done;
the third is the pretrained-model step.

---

# Results — Continual Learning on a Transformer (`transformer_cls.json`)

Reproduce:

```bash
python -m nous.train_transformer_cls --seeds 5      # → results/transformer_cls.json
python -m nous.train_transformer_cls --selfcheck
```

Source: [`nous/train_transformer_cls.py`](../nous/train_transformer_cls.py).
Moves the mod-5 result onto a **real gradient-trained net**: a small transformer
(2 layers, d=32) is briefly pretrained on all digits and then **frozen** (the
"use a pretrained backbone" analog). Task stream = **Split-digits** (`sklearn`,
8×8, zero download): 5 binary tasks `{0,1},{2,3},…,{8,9}` in phases; after each
phase, accuracy on all tasks so far.

- **`per_region`** — one low-rank adapter + head per discovered region, routed by
  geometry on the frozen pooled feature (nearest centroid). Only the routed
  region trains; **test-time routing uses no task id**. The step-3 mechanism.
- **`shared`** — one adapter + one growing head trained through the whole stream.
- **`full_ft`** — unfreeze the backbone + one growing head (upper-bound forget).

5 seeds:

| after all 5 tasks           | task 0: peak → final | forgetting | all-tasks final |
| --------------------------- | -------------------- | ---------- | --------------- |
| **`per_region`** (routed)   | 100.0 % → **97.8 %** | **+2.2 pp**  | 89.3 %        |
| **`shared`** adapter        | 100.0 % → 0.0 %      | +100 pp    | 39.3 %          |
| **`full_ft`**               | 100.0 % → 0.0 %      | +100 pp    | 14.4 %          |

- On a real transformer, a shared head/adapter and full fine-tuning **completely
  overwrite** task 0 (→ 0 %) by the end of the stream — textbook catastrophic
  forgetting. Task-conditioned experts forget **+2 pp**.
- Geometric routing spawned **~4.6 regions** for the 5 tasks with no task label —
  the router rediscovers task structure from the frozen features and, at test
  time, sends most inputs to the right expert.
- This is the payoff the mod-5 toy could *not* show (there the co-adaptive memory
  left the adapter nothing to do): with a real backbone whose head and features
  genuinely interfere, localizing the plasticity is the difference between 98 %
  and 0 % retention.

### Honest scope

- **Small and pretrained-then-frozen.** The backbone saw all digit classes during
  its brief pretrain (standard for pretrained backbones, but it means the frozen
  features are already good). Adapters do representation *refinement*, not
  from-scratch feature learning.
- **Adapter is a low-rank residual on the pooled feature**, not LoRA injected into
  the attention matrices — same local-vs-shared plasticity test, less plumbing.
- **Experts spawn at task boundaries** (one per phase); test routing is label-free
  but training still uses the phase boundary. Per-example surprise-spawn (as in
  the toy) is the remaining crutch to remove.
- **Baselines collapse to exactly 0 %** partly because the shared *growing head*
  is class-incremental with no replay — a strong (but standard) forgetting
  setting. The comparison isolates modular vs shared parameters, not replay.
- `all_final` is 89 %, not 100 %: some test inputs route to the wrong expert.
  Routing quality — not memory — is the ceiling here, and it degrades as regions
  crowd. The next step (pretrained model, real task, surprise-spawn) stresses
  exactly that.

---

# Results — Continual Learning on a Pretrained Transformer (`pretrained_cls.json`)

Reproduce (downloads all-MiniLM-L6-v2 ~90MB + 20NG ~14MB once, then cached):

```bash
python -m nous.train_pretrained_cls --seeds 3     # → results/pretrained_cls.json
python -m nous.train_pretrained_cls --smoke        # 1-seed sanity
```

Source: [`nous/train_pretrained_cls.py`](../nous/train_pretrained_cls.py).
A **real pretrained** transformer, frozen, with **real `peft` LoRA**: one LoRA
adapter + head per geometrically-routed region, vs a single shared LoRA and full
fine-tuning. Task stream = **20 Newsgroups**, 5 tasks, in phases. This is the
first rung with a genuinely pretrained model and real, *overlapping* tasks.

The committed script uses **`all-MiniLM-L6-v2`** with **coherent** super-topic
tasks — the *fixed* setup (see the last two subsections for why). The table below
is the **initial `distilbert-base-uncased` run** (commit `c3b80a7`, arbitrary
class grouping) that exposed the routing bottleneck; the MiniLM headline is at the
end of this section.

Initial run — distilbert, 3 seeds:

| after all 5 tasks                     | task 0: peak → final | forgetting | all-tasks final |
| ------------------------------------- | -------------------- | ---------- | --------------- |
| **`per_region`** (geometric routing)  | 70 % → 28 %          | +42 pp     | 43 %            |
| **`per_region`**, *oracle routing*    | 70 % → **70 %**      | **+0 pp**  | **75 %**        |
| **`shared`** LoRA                     | 70 % → 0 %           | +70 pp     | 6 %             |
| **`full_ft`**                         | 71 % → 0 %           | +71 pp     | 14 %            |

The result splits cleanly into a solved half and an open half:

- **Modular memory works — even here.** With *oracle* routing (each doc sent to
  the region that owns its label) the per-region experts retain **perfectly**:
  task 0 stays at 70 %, zero forgetting, on a real pretrained transformer where
  a shared LoRA and full fine-tuning both collapse task 0 to **0 %**. Localizing
  the plasticity is, again, the whole game.
- **Routing is now the bottleneck.** Realized retention (28 %) falls far below the
  oracle (70 %) — the **entire +42 pp gap is routing error**, not memory.
  Unsupervised nearest-centroid routing on frozen distilbert features is only
  ~60 % accurate on 20NG, because real topics overlap in feature space (the
  frozen features *carry* topic info — a trained linear probe gets ~72 % — but
  centroids don't separate them). On the clean-separated digits tasks routing was
  near-perfect; on real overlapping text it is not.

**Reading.** As tasks become real and overlapping, the hard problem *moves*: from
"don't overwrite old parameters" (solved by modular experts — oracle shows +0 pp)
to "send each input to the right expert" — **pattern separation**, the dentate-gyrus
function the neuroscience analogy names. A nearest-centroid is a poor stand-in for
it. That — a router that separates overlapping tasks — is the next real problem,
above per-example surprise-spawn.

### Honest scope

- Small subset (40 train / 20 test per class), `max_len` 64, 3 seeds, LoRA on
  `q_lin`,`v_lin` only — enough to show the gap, not a benchmark number.
- Experts still **spawn at task boundaries**; test-time routing is label-free.
- **Oracle routing uses test labels** — it is an upper bound to attribute the gap,
  never a deployable predictor.
- Baselines' class-incremental head has no replay; they hit 0 % on task 0 by the
  end, which is the standard strong-forgetting setting.

### Router step — a learned router does not help; the representation is the ceiling

The pretrained step blamed routing for the 28→70 % oracle gap, so the obvious next
move is a better router. Built one (`route="disc"`): a **modular discriminative
router** — each region keeps a small **feature-replay buffer** (32 cached pooled
features), and all region rows are refit jointly as a regularized multinomial
logistic on the buffer at each spawn (the buffer, not frozen rows, is what keeps
it from forgetting). It routes on unit-norm features. Compared head-to-head with
the nearest-centroid router and the oracle, 3 seeds:

| per_region, all-tasks final | learned router (`disc`) | nearest centroid | oracle |
| --------------------------- | ----------------------- | ---------------- | ------ |
| accuracy                    | **43 %**                | 43 %             | 76 %   |

**The learned router ties the centroid — zero gain.** And it is not the router's
fault: a logistic trained on *all* region features (not just the 32-vector replay)
reaches only **61 %** task-routing vs the centroid's **58 %**. The ceiling is the
**frozen representation** — 20NG tasks are unions of unrelated newsgroups that
overlap in distilbert's frozen feature space, and *no* linear router on those
features can separate them. You cannot route your way out of a bad representation.

**Reframe (the real open problem).** The series now closes a loop:

- Modular memory removes *overwriting* (oracle → +0 pp forgetting).
- On real overlapping tasks the bottleneck is *pattern separation* (routing).
- But routing is capped by the *representation*, not the router algorithm.
- Separable representations require *learning the features* — and step 2 showed
  that unfreezing a **shared** encoder drifts and forgets.

So the genuine next problem is **task-conditioned representation learning that
yields separable routing features without shared-encoder drift** — the recursion
this whole progression keeps hitting. A smarter router on frozen features is a
dead end; the leverage is in the features.

### Fix: a real sentence embedder + coherent tasks (MiniLM headline)

The router was a dead end, so we changed **the features**, not the router — the
cheapest lever. Two things were wrong with the distilbert setup:

- **`distilbert-base` `[CLS]` is a weak sentence embedding** (no sentence-level
  pretraining). Swapping to **`all-MiniLM-L6-v2`** (mean-pooled, unit-norm) —
  a model *trained* for semantic separation — lifts task-routing from ~0.58 to
  ~0.67.
- **The tasks were arbitrary.** Grouping *consecutive* class indices put unrelated
  newsgroups in one "task" (atheism + graphics + windows), so no embedder could
  route them. Using **coherent** super-topics (comp / rec / sci / talk / misc),
  which is what real continual tasks look like, lifts routing to ~0.71.

Committed setup (MiniLM + coherent tasks), 3 seeds, **plain centroid routing**:

| after all 5 tasks            | task 0: peak → final | forgetting | all-tasks final |
| ---------------------------- | -------------------- | ---------- | --------------- |
| **`per_region`**             | 70 % → **60 %**      | **+10 pp** | **53 %**        |
| **`per_region`**, *oracle*   | 70 % → 70 %          | +0 pp      | 77 %            |
| **`shared`** LoRA            | 70 % → 0 %           | +70 pp     | 7 %             |
| **`full_ft`**                | 59 % → 0 %           | +59 pp     | 14 %            |

- **Task-0 forgetting drops from +42 pp to +10 pp** just by fixing the
  representation and the task definition — the modular mechanism was never the
  problem, the embedding was. Baselines still collapse to 0 %.
- The learned `disc` router **still ties centroid** (both 53 %) — confirming the
  earlier finding a second time: on these features, plain nearest-centroid is the
  right router; sophistication buys nothing.
- A gap to oracle (77 %) remains — routing is ~0.68, dragged down by the one
  incoherent "misc" task and genuine sci/talk overlap. 20NG tops out here; a
  cleaner benchmark (or learned task-separating features) is the next lever.

**Net of the whole arc:** modular per-region experts + a decent frozen embedder +
coherent tasks give **+10 pp** forgetting where shared-LoRA and full fine-tuning
give **+59–70 pp**. Catastrophic forgetting is removed by *modularity*; the
residual is *pattern-separation* quality, set by the representation — exactly the
neuroscience split the series set out to test.

### Does a stronger embedder fix 20NG? Only up to intrinsic separability

If the representation is the ceiling, a *better* embedder should raise it. It
barely does — 20NG per-class-proto routing across embedder quality:

| embedder     | params | 20NG routing |
| ------------ | ------ | ------------ |
| MiniLM-L6    | 22 M   | 0.760        |
| bge-small    | 33 M   | 0.780        |
| mpnet-base   | 110 M  | 0.783        |

A **5× larger, stronger** embedder buys **+2 pp**. So on 20NG the residual is
**intrinsic task overlap, not embedding quality** — the concepts genuinely aren't
separable, and no better "address" fixes an ambiguous destination. Contrast the
*other* regime: a weak embedder on separable tasks (DBpedia: distilbert 0.6 →
MiniLM 0.95) — there, better embeddings lift routing enormously. So **"improve the
embeddings" helps only up to the tasks' intrinsic separability**; past that, the
leverage is task/concept design, not a bigger encoder. (Retention tracks routing
throughout, so 20NG retention is capped the same way regardless of encoder.)

### Sharper routing: per-class prototypes (`route="proto"`)

The residual gap (realized 53 % vs oracle 77 %) is pure routing, so we looked at
*where* it fails. Per-task routing accuracy:

| comp | rec | sci | talk | **misc** |
| ---- | --- | --- | ---- | -------- |
| 0.89 | 0.74 | 0.60 | 0.62 | **0.46** |

The **incoherent "misc" task** (atheism + windows.x + forsale + christian) drags
routing down: its task-centroid is a blur of unrelated classes, so nothing routes
to it. Fix — route to the **nearest per-CLASS prototype**, then to its region,
instead of the task-centroid. Even a blurry task keeps sharp per-class prototypes.
It stays modular: each region caches its class prototypes (frozen), and routing is
still label-free at test.

3 seeds, `all-tasks final` by routing method:

| **per-class proto** | task-centroid | learned `disc` | oracle |
| ------------------- | ------------- | -------------- | ------ |
| **61 %**            | 53 %          | 51 %           | 77 %   |

- Per-class prototype routing lifts realized retention **53 % → 61 %** (+8 pp),
  closing most of the remaining gap to the oracle (77 %) — routing accuracy rises
  from ~0.68 to ~0.79. Finer-grained pattern separation, no new training, no
  drift. (The learned `disc` router still trails, a third confirmation that
  sophistication on these features buys nothing — granularity does.)
- What is left (61 vs 77) is genuine class overlap (sci ↔ comp ↔ talk) that only a
  better representation resolves — the standing open lever.

### Confidence-gated top-k routing — a negative (`route="confident"`)

The 61 vs 77 gap is misrouting, and the correct region is in the **top-3** nearest
prototypes **96 %** of the time (top-1 only 76 %). So the obvious "postal
redundancy" fix: consult the top-3 candidate experts and trust the most confident
one (max softmax over its classes). It **doesn't work** — 3 seeds, 20NG:

| routing            | all-tasks final |
| ------------------ | --------------- |
| **proto** (top-1)  | **61 %**        |
| confident (top-3)  | 49 %            |
| centroid           | 53 %            |
| oracle             | 77 %            |

Consulting top-3 *lowers* accuracy (61 → 49 %). The reason is a well-known
failure: **neural-net confidence is miscalibrated out-of-distribution** — a wrong
expert, shown an input from another task, is *overconfident* on its own classes,
so "trust the most confident" lets the wrong expert out-shout the right one. The
routing information is there (top-3 recall 0.96), but a raw softmax can't extract
it. The signal that *does* work is prototype **distance** — which is exactly why
nearest-prototype (`proto`) already wins. Exploiting the top-k recall would need a
calibrated out-of-distribution / reject score, not raw confidence — an open
problem, and a reminder that uncertainty is its own unsolved piece alongside
representation quality.

### Separable benchmark: DBpedia-14 — modular memory hits oracle

The 20NG residual was the *benchmark's* intrinsic overlap, not a method flaw, so
we tested on a benchmark whose tasks are genuinely separable: **DBpedia-14**
ontology types grouped by super-type — org, people, place, nature, works. These
are cleanly distinguishable (a company vs an athlete vs a river vs a film), so
per-class prototype routing hits **~0.95** (vs 0.75 on 20NG). Same MiniLM +
per-region LoRA + proto routing; only the data changes (`--dataset dbpedia`).

3 seeds:

| after all 5 tasks           | task 0: peak → final | forgetting | all-tasks final |
| --------------------------- | -------------------- | ---------- | --------------- |
| **`per_region`** (proto)    | 98 % → **92 %**      | **+6 pp**  | **94 %**        |
| **`per_region`**, *oracle*  | —                    | —          | 97 %            |
| **`shared`** LoRA           | 98 % → 0 %           | +98 pp     | 7 %             |
| **`full_ft`**               | 82 % → 0 %           | +82 pp     | 22 %            |

- **Realized 94 % vs oracle 97 % — the gap closes to 3 pp.** When routing works,
  modular memory retains *near-perfectly* (+6 pp forgetting) on a real pretrained
  transformer, while a shared LoRA and full fine-tuning are wiped out (→ 0 %).
- This is the whole arc's thesis, end to end: **catastrophic forgetting is a
  parameter-sharing artifact.** Give each task its own frozen expert and route
  correctly, and the model just *keeps* what it learned. The only thing between
  the toy and this is pattern-separation quality — set by the representation and
  the benchmark, not by the memory mechanism, which was never the problem.

**Whole series in one line:** modularity removes the forgetting (every step);
routing quality sets how much you realize of it (≈100 % when tasks separate,
~80 % when they overlap); no learned router beats per-class prototypes on frozen
features — the leverage is separable representations, not a cleverer router.

### Does it scale? Task count 5 → 14 (`--tasks`)

Everything above uses 5 tasks. The open question is whether retention holds as
tasks *multiply* — more experts spawn, more prototypes crowd the routing space.
DBpedia-14 chunked into `n` contiguous tasks (`--dataset dbpedia --tasks n`),
3 seeds:

| tasks | **`per_region` all-final** | oracle | routing (proto) | `shared` | `full_ft` | regions |
| ----- | -------------------------- | ------ | --------------- | -------- | --------- | ------- |
| 5     | **94 %**                   | 97 %   | 0.943           | 7 %      | 22 %      | 5       |
| 7     | **94 %**                   | 99 %   | 0.929           | 7 %      | 7 %       | 7       |
| 14    | **93 %**                   | 100 %  | 0.918           | 7 %      | 7 %       | 14      |

- **Retention is flat and near-oracle — 94 → 94 → 93 % — from 5 to 14 tasks.**
  The modular advantage does not decay as tasks scale; if anything the gap to the
  baselines *widens*, since a shared head collapses harder the more classes it must
  cram (`full_ft` 22 % → 7 % as tasks go 5 → 14, i.e. toward 1/14 chance).
- Routing degrades only **gracefully** (0.94 → 0.92), so the earliest task's
  per-task forgetting rises just +6 → +12 pp while all-tasks retention holds.
- Cost grows exactly linearly — one frozen expert per task (5 / 7 / 14 regions),
  no shared state to interfere.

So within the scale runnable here, **the mechanism holds up**: catastrophic
forgetting stays solved as tasks multiply; the only slowly-moving part is routing,
and it degrades gently on a separable benchmark. The untested frontier remains a
genuinely *large* model and hundreds of tasks — out of reach on this hardware
(MPS, no CUDA), but the curve here points the right way.

---

# Results — Sequential fine-tuning of a small LLM (`llm_cl.json`)

Reproduce (downloads gpt2 ~0.5GB + DBpedia once, cached):

```bash
python -m nous.train_llm_cl --seeds 3      # → results/llm_cl.json
python -m nous.train_llm_cl --smoke        # 1-seed sanity
```

Source: [`nous/train_llm_cl.py`](../nous/train_llm_cl.py).
A **real pretrained decoder LLM** — `gpt2` (125M) — fine-tuned on a stream of
DBpedia super-topic classification tasks (5 tasks, in phases), comparing standard
sequential fine-tuning against the modular per-task-expert mechanism. After each
phase we measure accuracy on every task so far and report **average forgetting**
(mean drop from each task's peak to its final).

- **`seq_lora`** — one shared LoRA adapter (+ growing head) fine-tuned on each task
  in turn: the canonical continual-fine-tuning baseline.
- **`full_ft`** — unfreeze the whole backbone + growing head, sequentially.
- **`modular`** — one frozen-backbone LoRA expert + head per task, routed by
  geometry on the frozen mean-pooled feature (per-class prototype, no task id).

3 seeds:

| method                                | final avg acc | **avg forgetting** | task 0 final |
| ------------------------------------- | ------------- | ------------------ | ------------ |
| **`modular`** (per-task experts)      | **70 %**      | **+16 pp**         | **60 %**     |
| **`seq_lora`** (standard sequential)  | 21 %          | +94 pp             | 0 %          |
| **`full_ft`** (standard sequential)   | 28 %          | +82 pp             | 0 %          |

- On a **real LLM**, standard sequential fine-tuning — LoRA *or* full — forgets
  catastrophically: task 0 → **0 %**, average forgetting **+82–94 pp**. The
  modular mechanism keeps retention high (70 %) with **+16 pp** forgetting.
- Modular's residual +16 pp is again **routing**, not overwriting (its experts are
  frozen): gpt2's features are a weaker router than a trained sentence embedder
  (per-class proto routing ~0.71 vs MiniLM's ~0.95), and gpt2's *last-token* state
  routes even worse (~0.38) — mean-pool is used for that reason. Better routing
  features would lift the 70 % toward the ~100 % the separable-benchmark encoder
  runs reached.

### Honest scope

- Small LLM (gpt2 125M), 5 tasks, 40 train / 20 test per class, LoRA on `c_attn`,
  2–3 seeds — a real-LLM existence proof, not a benchmark number. Large models and
  long task streams remain out of reach on MPS (no CUDA).
- The baselines' collapse to 0 % on task 0 is the standard class-incremental,
  no-replay setting — the point is the *contrast* with modular under identical
  data and budget.

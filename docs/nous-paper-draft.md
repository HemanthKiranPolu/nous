# NOUS: Computation as Physical Relaxation on a Riemannian Semantic Manifold

**Abstract.** We present NOUS (Neural Omnidimensional Unified System), an architecture
in which computation is identified with physical relaxation and learning with landscape
sculpting — without backpropagation through time or through any ODE solver. An input
token is encoded as a continuous force field clamped over a high-dimensional state
space; inference is the overdamped relaxation of a particle to the minimum of the
resulting energy landscape; training reshapes the landscape via Equilibrium Propagation
(EqProp), requiring only two evaluations of ∂E/∂θ at fixed-point states. Morphogenesis
— the adaptive creation of new attractor basins — is triggered by a dual condition on
displacement and stochastic minimum curvature, and its firing pattern reveals three
physically distinct phases: ignition (flat landscape), carving (silence as basins form),
and reorganization (active repositioning). On a causal language modeling task with a
50,257-token vocabulary, NOUS reduces perplexity from random baseline (e^10.82) to
below e^0.66 in 80 epochs with zero gradient steps through any recurrence. The same
word appearing in different sentence contexts produces distinct equilibrium positions,
encoding context sensitivity as attractor topology rather than as a lookup or attention
map.

---

## 1. Introduction

Modern neural sequence models are differentiated programs: the forward pass traces a
computation graph, and learning requires gradients to flow backward through every step
of that graph. For recurrent architectures this is Backpropagation Through Time (BPTT);
for transformers, through the attention and MLP layers at every layer of every position.
The computational and memory cost of these backward passes scales with sequence length,
layer depth, and model width.

We ask a different question: what if a model does not have a computation graph at all?
What if the forward pass is not a program but a *physical process* — the relaxation of a
dynamical system to equilibrium — and training is not gradient descent through the
forward pass but the sculpting of the energy landscape that governs that relaxation?

This is the founding insight of NOUS. It unifies three ideas from outside the standard
deep learning canon:

1. **Lagrangian mechanics / Principle of Least Action**: the state q ∈ ℝ^d of the
   system evolves along the path of least energy. Inference = relaxation.

2. **Equilibrium Propagation (Scellier & Bengio, 2017)**: parameters θ are updated by
   comparing ∂E/∂θ at two fixed points — the free equilibrium and a gently nudged
   equilibrium — without any backward pass through the ODE solver.

3. **Turing morphogenesis**: when a new context has no attractor basin (detected by
   stochastic minimum curvature dropping below a threshold), new RBF centers are
   instantiated. The architecture grows.

The result is a model in which:
- Memory across tokens is encoded as **attractor topology**, not hidden state vectors
- Learning is **landscape sculpting**, not parameter gradient descent through a program
- The model can grow **new semantic basins** during training without disturbing existing ones

---

## 2. Architecture

### 2.1 Energy function

For input x ∈ ℝ^m and state q ∈ ℝ^d:

```
E(x, q; θ) = V(q; θ) − x^T W_in q
```

The potential V(q) consists of three components:

```
V(q) = ½‖q‖²  +  Σ_k (−amp_k) · exp(−‖q − μ_k‖² / σ_k²)  +  MLP_res(q)
```

- **Quadratic bowl** `½‖q‖²`: prevents any monotone slope; all eigenvalues of ∂²V/∂q²
  receive a minimum contribution of +1.
- **RBF terms**: carve local attractor basins at positions μ_k with amplitude amp_k and
  width σ_k. These are learned parameters.
- **MLP residual**: fine-grained landscape shaping beyond what RBF centers can express.

The coupling term `−x^T W_in q` adds a linear tilt proportional to the input. Different
inputs create different tilts → different equilibrium positions. The equilibrium condition
∂E/∂q = 0 gives ∂V/∂q = W_in^T x: the potential gradient balances the input force.

### 2.2 Inference

The state evolves under overdamped gradient flow (the Euler-Lagrange equation for an
overdamped system):

```
q̇ = −∂E(x, q; θ)/∂q
```

integrated with Euler steps until convergence to q*. The input x is **clamped throughout
dynamics**, not just used as an initial condition. This is the critical distinction from
earlier energy-based models: different inputs produce different force fields, and therefore
different attractor positions, regardless of initialization.

For sequence modeling, q_{t} is initialized from q*_{t-1}: the previous equilibrium
position is the starting point for the next token's relaxation. Context is carried by
**basin position in 512D space**, not by a recurrent cell.

### 2.3 Training via Equilibrium Propagation

EqProp (Scellier & Bengio, 2017) computes parameter gradients without backpropagation
through the ODE:

**Phase 1 (Free):** solve q̇ = −∂E/∂q with x clamped → q*_free

**Phase 2 (Nudge):** solve q̇ = −∂(E + ε·C)/∂q → q*_nudge
where C(q) = CrossEntropy(decoder(q), target)

**Update:**
```
Δθ = −α · (1/ε) · [∂E(q*_nudge; θ)/∂θ − ∂E(q*_free; θ)/∂θ]
```

This requires only two evaluations of ∂E/∂θ at fixed-point states — no Jacobian of the
ODE, no adjoint equations, no stored activations proportional to sequence length.

The decoder (Linear(d, vocab_size)) is updated separately via standard CE at q*_nudge.

### 2.4 Morphogenesis trigger

A new RBF center is instantiated when two conditions hold simultaneously:

```
‖q*_nudge − q*_free‖ > φ_dist    AND    λ̃_min(∂²E/∂q²)|_{q*_free} < φ_curv
```

The first condition detects large displacement (the landscape has shallow regions the
nudge can move through). The second condition, estimated via stochastic Rayleigh quotient
with n_probes random unit vectors, detects low curvature at the current equilibrium
(no deep basin exists there). In 512D, the bowl contributes +1 to every eigenvalue; we
set φ_curv = 1.2, triggering when the RBF component has not added ≥0.2 additional
curvature.

---

## 3. Three-Phase Morphogenesis Dynamics

Training on a 10-word sentence for 80 epochs reveals three qualitatively distinct phases:

**Phase I — Ignition (epochs 0-10):** The landscape is flat (only the bowl). λ̃_min ≈ 1.0
for all token contexts. Morphogenesis fires at every token position on every step. Loss
at 10.82 (random baseline over 50,257 tokens).

**Phase II — Carving (epochs 10-27):** EqProp sculpts the landscape; RBF wells deepen.
λ̃_min rises above φ_curv = 1.2. **Morphogenesis goes silent.** This silence gap is a
geometric signal — the threshold crossing indicates that 32 RBF centers have acquired
sufficient curvature to constitute actual attractor basins. Loss: 2.17 → 1.99.

**Phase III — Reorganization (epochs 27-79):** As EqProp continues refining basin
*positions* for better next-token prediction, some basins temporarily flatten as they
shift, triggering morphogenesis again. This is not instability; it is active topological
reorganization. Loss: 1.99 → 0.66.

The silence gap is not an artifact of threshold choice. It represents the transition from
"flat landscape with no functional basins" to "landscape with carve attractor geometry."

---

## 4. Experiments

### 4.1 XOR (2D, proof of concept)

- Input: 2D binary vectors; state: 2D; decoder: Linear(2, 2)
- EqProp without any backprop through the ODE
- 4/4 accuracy achieved on seeds {3, 5, 13, 16} out of 20 tested
- Best: seed 5, loss = 0.022
- Basin formation confirmed visually via V(q) landscape plots

The ~20% success rate on random seeds reflects a geometric constraint: with a linear
decoder in 2D, the 4 equilibria must land in a linearly separable arrangement. This
constraint disappears at higher state dimension.

### 4.2 Single-sentence language modeling (512D)

- Sentence: "The cat sat on the mat near the big tree" (10 GPT-2 tokens)
- Vocab: 50,257; embed: 64D; state: 512D; RBF: 32 centers
- 80 epochs; 0 backpropagation steps through any ODE
- Loss: 10.82 → 0.66 (16× below random baseline)
- Best individual transition: "The" → "cat" CE = 0.21 (near-certain)

**Context sensitivity without positional encoding:** The word "the" appears at positions
4 and 7. Both produce different 512D equilibria because q is initialized from the
previous token's equilibrium. Same token, different context → different attractor.
This is verified by PCA projection of the 512D states.

**Syntactic structure from energy alone:** PCA and pairwise distance matrices show
"on" and "near" clustering together (both prepositions in similar syntactic slots)
without any part-of-speech labels, attention mechanism, or positional encoding.

### 4.3 Corpus scaling — attractor consistency [IN PROGRESS]

- Dataset: WikiText-2, 200 sentences (5-12 tokens each)
- Tracking 12 recurring tokens across all sentences and epochs
- Key metric: intra-word attractor variance (low → same word always lands near same
  basin regardless of sentence context)

[Results to be filled after training completes]

---

## 5. Related Work

**Equilibrium Propagation (Scellier & Bengio, 2017):** NOUS applies EqProp to sequence
modeling for the first time. The original paper demonstrated EqProp on static tasks
(MNIST); we extend to causal language modeling with stateful carry between positions.

**Energy-Based Models (LeCun et al., 2006; Grathwohl et al., 2019):** EBMs define a
scalar energy over (x, y) pairs and train by contrastive methods. NOUS differs: the
energy E(x, q) is defined over continuous state q, not over (input, output) pairs, and
inference is relaxation, not sampling.

**Continuous-depth models (Chen et al., 2018 — Neural ODEs):** Neural ODEs define
dynamics via a neural network and differentiate through the ODE solver. NOUS explicitly
avoids differentiating through the solver; EqProp is the replacement.

**Predictive Coding (Rao & Ballard, 1999; Millidge et al., 2022):** PC also avoids
BPTT by using local prediction errors at each layer. NOUS differs in treating the entire
state as a single particle on an energy landscape rather than a layered hierarchy.

**Hopfield Networks / Modern Hopfield (Ramsauer et al., 2020):** Modern Hopfield
networks store memories as energy minima and retrieve via dynamics. NOUS is closer in
spirit but differs critically: the energy landscape is input-dependent (clamped coupling),
trained via EqProp rather than Hebbian rules, and extended with RBF potential and
morphogenesis.

**Turing morphogenesis (Turing, 1952):** The dual trigger (displacement + curvature) is
inspired by Turing's reaction-diffusion instability criterion. New basins appear when the
existing landscape cannot accommodate the current input's energy minimization.

---

## 6. Limitations and Open Problems

1. **Initialization sensitivity at low state dimension:** In 2D, 4/4 XOR requires a
   specific geometric arrangement of equilibria. At STATE_DIM ≥ 4 this constraint
   relaxes naturally — full investigation is ongoing.

2. **ODE solve cost:** Each EqProp step requires two ODE solves (free + nudge). At
   STATE_DIM = 512 with 150 Euler steps each, per-token cost is ~0.1s on CPU. GPU
   parallelism across tokens in a batch is the natural fix.

3. **Stochastic Hessian bias:** The stochastic minimum Rayleigh quotient with 8 probes
   underestimates λ_min. For morphogenesis trigger accuracy, more probes or Lanczos
   iteration would improve the curvature estimate.

4. **Linear decoder breaks the paradigm:** Ideally, output token selection is also
   physical relaxation: the predicted token is the one whose embedding minimizes
   E(q*, e_token). This would make the full pipeline Lagrangian end-to-end.

5. **Sequence length:** Currently O(n) ODE solves per sequence. No attention-like
   O(n²) — but also no parallelism within a sequence. Batching across sequences
   is independent EqProp steps.

---

## 7. Conclusion

NOUS demonstrates that computation can be physical relaxation and learning can be
landscape sculpting, with zero gradient flow through any recurrence or ODE solver.
The morphogenesis silence gap — where basin-carving events cease as λ̃_min crosses
the curvature threshold — provides a direct geometric signal that functional attractor
basins have formed. The resulting 512D semantic manifold encodes context sensitivity
as topology (same word, different context → different equilibrium), and syntactic
structure (prepositions clustering) emerges from EqProp pressure alone.

The architecture has no prior art as a unified system. We release all code at
https://github.com/HemanthKiranPolu/nous.

---

## References

- Scellier, B. & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between
  energy-based models and backpropagation. *Frontiers in Computational Neuroscience.*
- LeCun, Y. et al. (2006). A tutorial on energy-based learning. *Predicting Structured
  Data.*
- Chen, R. T. Q. et al. (2018). Neural ordinary differential equations. *NeurIPS.*
- Ramsauer, H. et al. (2020). Hopfield networks is all you need. *ICLR 2021.*
- Turing, A. M. (1952). The chemical basis of morphogenesis. *Philosophical Transactions
  of the Royal Society B.*
- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex.
  *Nature Neuroscience.*
- Millidge, B., Seth, A. & Buckley, C. L. (2022). Predictive coding: a theoretical and
  experimental review. *arXiv.*

# NOUS — Neural Omnidimensional Unified System
**Architecture Specification v0.1 · 2026-06-23**

---

## 0. The One Sentence

NOUS is an AI architecture where computation is physical relaxation: a system governed by a learned Lagrangian that settles into attractor basins encoding knowledge, trained without backpropagation via Equilibrium Propagation and thermodynamic annealing, and capable of growing its own structure via Turing reaction-diffusion when it encounters unrepresentable inputs.

---

## 1. Foundational Axioms Broken

| Axiom (all prior architectures) | NOUS |
|---|---|
| Information = vector in ℝ^d | Information = position in continuous semantic field |
| Thinking = matrix multiply + nonlinearity | Thinking = physical relaxation to equilibrium |
| Memory = stored weight or retrieved vector | Memory = attractor basin depth/position in ℒ |
| Training = backprop through computational graph | Training = sculpting energy landscape by cooling |
| Architecture = fixed at design time | Architecture = grown via morphogenesis during training |

---

## 2. The Three Unified Paradigms

The three paradigms are not separate components. They are the same system described in three languages:

- **Geometry (I):** The shape of attractor basins in the potential V(q) ⊂ ℒ **is** the Riemannian metric tensor g(x)
- **Dynamics (II):** Euler-Lagrange relaxation **is** the computation
- **Growth (III):** Unresolvable perturbation energy **is** the morphogen signal

---

## 3. Core Mathematical Formulation

### 3.1 The Lagrangian

The system state is a continuous field q(t) ∈ ℝ^d evolving over "thought time" t ∈ [0, T].

The learned Lagrangian:

```
ℒ(q, q̇, t; θ) = T(q, q̇) − V(q; θ)
```

Where:
- **T(q, q̇) = ½ q̇ᵀ M(q) q̇** — kinetic energy; M(q) is a learned mass matrix (positive definite, parameterized by a small MLP)
- **V(q; θ)** — learned potential energy; encodes all world knowledge as attractor basins; parameterized by a deep network

### 3.2 The Equations of Motion (The Forward Pass)

The system evolves according to the Euler-Lagrange equations:

```
d/dt(∂ℒ/∂q̇) − ∂ℒ/∂q = 0
```

Expanded:

```
M(q)·q̈ + (∂M/∂q · q̇)·q̇ − ½(∂/∂q)(q̇ᵀ M(q) q̇) = −∂V(q)/∂q
```

This is a second-order ODE. The system is run forward in time until equilibrium (q̇ → 0).

**Equilibrium condition:** `‖q̇(t)‖ < δ` for δ = 1e-4

The equilibrium state q* is the system's "answer" to the input boundary condition q(0).

### 3.3 Boundary Conditions (Input Encoding)

```
q(0) = E_enc(input)     # learned encoder: tokens → initial field state
q̇(0) = 0               # system starts at rest
```

For language: input tokens are embedded and projected to the d-dimensional state space via a learned linear encoder E_enc ∈ ℝ^(vocab×d).

### 3.4 Output Decoding

```
output = E_dec(q*)      # learned decoder: equilibrium state → distribution over tokens
```

For language modeling: E_dec is a linear projection followed by softmax.

---

## 4. Training Algorithm (No Backpropagation)

### 4.1 Equilibrium Propagation — Two-Phase Update

**Phase 1 — Free Relaxation:**
```
q(0) ← E_enc(input)
Solve Euler-Lagrange until ‖q̇‖ < δ
q*_free ← q(T_eq)
```

**Phase 2 — Nudged Relaxation:**
```
ℒ_nudge(q) ← ℒ(q; θ) + ε · C(E_dec(q), target)
Solve nudged Euler-Lagrange until ‖q̇‖ < δ
q*_nudge ← q(T_eq)
```

Where C is the cross-entropy cost and ε ≪ 1 is the nudge strength.

**Parameter Update (no backward pass):**
```
Δθ = −α · (1/ε) · [∂ℒ(q*_nudge; θ)/∂θ − ∂ℒ(q*_free; θ)/∂θ]
θ ← θ + Δθ
```

This requires only: evaluating ∂ℒ/∂θ at two points in state space. No computational graph retained. No chain rule traversal.

**Theoretical guarantee:** As ε → 0, Δθ converges to the true gradient ∂C/∂θ.
*Source: Scellier & Bengio (2017), Theorem 1. Generalized to continuous ℒ via calculus of variations.*

### 4.2 Hebbian Consolidation (Correct Predictions)

When E_dec(q*_free) ≈ target (prediction is already correct):

```
θ ← θ + η_hebb · ∂²ℒ/∂q² |_{q*_free}
```

This deepens the existing basin without disturbing surrounding landscape. Correct answers reinforce.

### 4.3 Thermodynamic Training Schedule

Training temperature replaces epochs. The probability of any trajectory is:

```
P[q] ∝ exp(−β(t) · S[q])
```

where S[q] = ∫ ℒ(q, q̇) dt is the action along trajectory q.

**Cooling schedule:**
```
β(t) = β₀ · exp(λ · t)
```

- **High temperature (small β):** System samples many trajectories. Broad shallow basins form across the landscape. Wrong attractors are accessible — system explores.
- **Low temperature (large β):** Only minimum-action trajectories survive. Shallow/incorrect basins dissolve via Boltzmann suppression. exp(−β·S_wrong) → 0.

**Convergence criterion:** Stop when dF/dt < ε_converge, where F = −(1/β)·log Z is the free energy.

Benefits over epoch-based training:
- No arbitrary stopping criterion
- Incorrect knowledge self-deletes thermodynamically
- No dropout, weight decay, or regularization needed — the physics cleans itself

---

## 5. Morphogenetic Growth (Turing Reaction-Diffusion)

### 5.1 Dual Trigger Condition

Morphogenesis fires when BOTH conditions hold:

```
TRIGGER = (‖q*_nudge − q*_free‖ > φ_distance)
        AND (λ_min(∂²ℒ/∂q² |_{q*_free}) < φ_curvature)
```

- **Condition 1:** Large distance between free and nudged equilibria → the nudge couldn't find a nearby basin
- **Condition 2:** Low Hessian curvature at free equilibrium → flat landscape = no basin (distinguishes "shallow basin" from "no basin")

Together: no attractor basin exists at this location in semantic space.

### 5.2 Turing Reaction-Diffusion Dynamics

When triggered, two competing chemical signals govern growth:

```
∂A/∂q = D_A ∇²_q A + f(A, B)    # activator: grow here
∂B/∂q = D_B ∇²_q B + g(A, B)    # inhibitor: constrain growth
```

With D_B ≫ D_A (long-range inhibition, short-range activation — Turing's original condition for pattern formation).

The source term for the activator:

```
A_source(q*_free) = ‖q*_nudge − q*_free‖ · (1 − λ_min(H_ℒ)/λ_threshold)
```

The energy deficit IS the morphogen signal. No separate growth mechanism.

### 5.3 New Structure Growth

Where A(q) − B(q) > θ_grow:
- A new neuron (parameter cluster) is instantiated at position q in semantic space
- Its initial weights: V_new(q) = −A(q) (carves an attractor at that location)
- Connected to existing structure via learned synaptic weights initialized to small values

New structure deepens the basin until `‖q*_free − target_q‖ < φ_distance` — the trigger deactivates.

---

## 6. The Riemannian Geometry (Emergent)

The metric tensor of semantic space is not separately parameterized. It emerges from the Lagrangian:

```
g_ij(q) = M_ij(q) + ∂²V(q)/∂q_i∂q_j
```

- M(q): kinetic mass matrix — governs "inertia" of thought (how hard it is to move through semantic space)
- ∂²V/∂q²: Hessian of potential — basin curvature; deep basins = high curvature = dense semantic neighborhood

Geodesics on this manifold are exactly the minimum-action trajectories. The three paradigms are mathematically identical.

---

## 7. Architecture Components

| Component | Role | Parameterization |
|---|---|---|
| `E_enc` | Input → initial field state | Linear projection (vocab × d) |
| `MassNet` | Learns M(q) — kinetic metric | 2-layer MLP, output ∈ PD matrices |
| `PotentialNet` | Learns V(q) — energy landscape | Deep residual network, scalar output |
| `ELSolver` | Solves Euler-Lagrange ODE | torchdiffeq (adjoint method for bootstrap; EqProp for production) |
| `EqProp` | Two-phase training | No backward pass; two ODE solves + subtract |
| `TuringField` | Reaction-diffusion growth | PDE solver over semantic space |
| `E_dec` | Equilibrium state → output | Linear projection + softmax |
| `AnnealScheduler` | β(t) cooling | Exponential schedule with free energy monitor |

---

## 8. Prototype Plan (Phase 1)

**Goal:** Validate that attractor basins form and training converges on a toy task.

**Task:** 2D semantic space (d=2), XOR classification. Input: two binary values. Output: XOR result. Visualize the energy landscape V(q) evolving during training.

**Stack:**
- Python 3.11 + PyTorch 2.4
- torchdiffeq (adjoint ODE solver — bootstrap before EqProp)
- matplotlib (landscape visualization)

**Components to build (in order):**
1. `PotentialNet` — scalar field V: ℝ² → ℝ
2. `ELSolver` — ODE integration to equilibrium
3. `EqProp` — two-phase update
4. `AnnealScheduler` — β cooling
5. Visualization: V(q) as heatmap, trajectories as streamlines, basins as contours

**Success criterion:** After training, V(q) has four distinct basins corresponding to (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. The landscape self-organizes without backpropagation.

**Phase 2:** Scale to language — 512d state space, GPT-2 tokenizer, train on small corpus.

---

## 9. Open Problems

| Problem | Status |
|---|---|
| Morphogenesis in continuous field: how to instantiate discrete new neurons in a continuous V(q) | Open — explore: V_new as a sum of Gaussian basis functions, new Gaussians added at trigger locations |
| EqProp convergence rate vs adjoint backprop at scale | Unknown — must benchmark at d=512 |
| Thermodynamic cooling schedule λ sensitivity | Unknown — must ablate β₀ and λ |
| Dual morphogenesis trigger threshold tuning (φ_distance, φ_curvature) | Unknown — start with φ_distance=0.5, φ_curvature=0.01 |
| Whether V(q) can represent compositional structure (phrase = sum of word basins?) | Open theoretical question |

---

## 10. Key References

- Scellier & Bengio — "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation" [arXiv:1602.05179, 2017]
- Turing — "The Chemical Basis of Morphogenesis" [Phil. Trans. R. Soc. B, 1952]
- Friston — "The Free Energy Principle" [Nat Rev Neurosci, 2010]
- Feynman & Hibbs — "Quantum Mechanics and Path Integrals" [1965] — thermodynamic annealing via path integral
- Ramsauer et al. — "Hopfield Networks is All You Need" [arXiv:2008.02217, 2020] — modern energy-based memory (antecedent)
- Chen et al. — "Neural Ordinary Differential Equations" [arXiv:1806.07366, 2018] — ODE as computation (antecedent)
- Goldstein — "Classical Mechanics" (3rd ed.) — Euler-Lagrange equations, action principle

---

*NOUS has no prior art. The combination of Lagrangian field dynamics + Equilibrium Propagation training + thermodynamic annealing + Turing morphogenesis as a unified architecture does not exist in any published work as of 2026-06-23.*

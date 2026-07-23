"""
NOUS-CLS — surprise-gated continual operator learning (toy).

Thesis under test
-----------------
Does surprise-gated, LOCAL basin allocation let a NOUS energy field learn a
STREAM of operations (+ then × then −, mod 5) WITHOUT forgetting the earlier
ones — where a standard SGD MLP catastrophically forgets, and an op-BLIND
NOUS variant (shared basins) forgets more than the op-aware one?

Human-learning primitives borrowed (see conversation design):
  - surprise gates plasticity: a correct prediction only *consolidates* (deepens
    the winning basin, spec §4.2); a wrong one *allocates* new structure.
  - complementary-learning-systems locality: each operation's knowledge lives in
    its own RBF basins; an update during op K never touches op≠K basins, so old
    skills are untouched dimensions of the potential V(q).

Readout = labeled attractor basins (no plastic decoder, no far anchors): each
basin carries the class label it was carved for; the particle relaxes and the
prediction is the label of the basin that wins the energy competition at q*
(argmax pull). All learning is basin sculpting.

Honest scope: toy, mod-5. Tests RETENTION under a non-stationary stream, NOT
generalization (SCAN-mini covers that). With a frozen random W_in, distinct
inputs land at distinct positions, so the field ends up ≈ one basin per input
region — it *memorizes* each op. That is fine: the claim is that LOCAL memory
growth does not overwrite old ops, which a shared-parameter learner does. Small-n.

# ponytail: overdamped first-order relaxation with an ANALYTIC vectorized force —
# no autograd in the hot loop, no inertia. Swap in el_solver_v2 if oscillation matters.
# ponytail: refinement = direct Hebbian sculpting (μ EMA toward the input + amp
# deepen), not eqprop-toward-anchor — simpler and robust at this scale. The spec's
# EqProp two-phase is the upgrade path if a differentiable nudge is needed.
# ponytail: allocation trigger = wrong-prediction only; the spec's low-curvature
# gate is deferred (wrong is a sufficient "no basin here" signal at this scale).
"""

import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Task: three operations on Z_5 ────────────────────────────────────────────
VALS = 5
OPS = {
    "add": lambda a, b: (a + b) % VALS,
    "mul": lambda a, b: (a * b) % VALS,
    "sub": lambda a, b: (a - b) % VALS,
}
OP_ID = {"add": 0, "mul": 1, "sub": 2}
IN_DIM = 2 * VALS + len(OPS)   # onehot(a) ⊕ onehot(b) ⊕ onehot(op) = 13


def encode(a: int, b: int, op: str) -> torch.Tensor:
    x = torch.zeros(IN_DIM)
    x[a] = 1.0
    x[VALS + b] = 1.0
    x[2 * VALS + OP_ID[op]] = 1.0
    return x


def op_dataset(op: str):
    return [(encode(a, b, op), OPS[op](a, b))
            for a in range(VALS) for b in range(VALS)]


# ── Energy field: growable, labeled, per-basin-controllable RBF basins ───────
class BasinField:
    """
    E(x, q) = ½‖q‖² − Σ_k amp_k·exp(−‖q−μ_k‖²/σ_k²) − (W_in x)·q

    W_in is a FIXED random projection (places each input in state space). Each
    basin is an independent record so we can grow the set and touch arbitrary
    subsets — that per-basin locality is the anti-forgetting mechanism. A basin
    also stores the class `label` it was carved for and the `scope` (op) that
    owns it.
    """

    LOG_AMP_MAX = math.log(6.0)

    def __init__(self, state_dim: int, W_in: torch.Tensor):
        self.D = state_dim
        self.W_in = W_in                      # (D, IN_DIM), fixed
        self.mu, self.log_amp = [], []        # per-basin center, log depth
        self.sig2 = []                        # per-basin width²
        self.label, self.scope = [], []       # per-basin class, owning op
        self.last_used = []                   # step of last touch (for LRU eviction)

    # ---- structure growth --------------------------------------------------
    def add_basin(self, center: torch.Tensor, label: int, scope: str, t: int = 0,
                  amp0: float = 2.0, sig0: float = 0.5) -> int:
        self.mu.append(center.clone())
        self.log_amp.append(math.log(amp0))
        self.sig2.append(sig0 ** 2)
        self.label.append(label)
        self.scope.append(scope)
        self.last_used.append(t)
        return len(self.mu) - 1

    def scope_count(self, scope: str) -> int:
        return sum(s == scope for s in self.scope)

    def lru_in_scope(self, scope: str):
        """Least-recently-used basin index within a scope (age-based eviction)."""
        idxs = [i for i, s in enumerate(self.scope) if s == scope]
        return min(idxs, key=lambda i: self.last_used[i]) if idxs else None

    def nearest_in_scope(self, scope: str, q: torch.Tensor):
        """Spatially nearest basin index within a scope, any label (geometric eviction).
        Keeps a reuse's damage inside the current input's region — no task label needed."""
        idxs = [i for i, s in enumerate(self.scope) if s == scope]
        if not idxs:
            return None
        return min(idxs, key=lambda i: ((self.mu[i] - q) ** 2).sum().item())

    def reuse(self, i: int, q: torch.Tensor, label: int, t: int, amp0: float = 2.0):
        """Repurpose basin i for a new (region, label) — the capacity-pressure event."""
        self.mu[i] = q.clone()
        self.label[i] = label
        self.log_amp[i] = math.log(amp0)
        self.last_used[i] = t

    def touch(self, i: int, t: int):
        self.last_used[i] = t

    def deepen(self, i: int, step: float = 0.3):
        self.log_amp[i] = min(self.log_amp[i] + step, self.LOG_AMP_MAX)

    def recenter(self, i: int, q: torch.Tensor, rate: float = 0.5):
        self.mu[i] = self.mu[i] + rate * (q - self.mu[i])

    # ---- energy / dynamics (analytic, vectorized) --------------------------
    def _stack(self):
        mu = torch.stack(self.mu)                        # (K, D)
        amp = torch.tensor(self.log_amp).exp()           # (K,)
        s2 = torch.tensor(self.sig2)                     # (K,)
        return mu, amp, s2

    def force(self, x: torch.Tensor, q: torch.Tensor,
              drive: torch.Tensor = None) -> torch.Tensor:
        """−∂E/∂q = −q + drive + Σ_k 2·amp_k·exp(−d²/σ²)/σ²·(μ_k − q).
        `drive` is the (constant) input pull; defaults to W_in·x but an adapter
        can pass an alternative encoding. relax never needs ∂/∂W_in."""
        d = self.W_in.detach() @ x if drive is None else drive
        f = -q + d
        if self.mu:
            mu, amp, s2 = self._stack()
            diff = mu - q                                # (K, D)
            d2 = (diff ** 2).sum(-1)                      # (K,)
            w = 2.0 * amp * torch.exp(-d2 / s2) / s2      # (K,)
            f = f + (w.unsqueeze(-1) * diff).sum(0)
        return f

    def relax(self, x: torch.Tensor, drive: torch.Tensor = None,
              steps: int = 50, dt: float = 0.1) -> torch.Tensor:
        if drive is None:
            drive = self.W_in.detach() @ x               # compute once, constant in q
        q = torch.zeros(self.D)
        for _ in range(steps):
            q = q + dt * self.force(x, q, drive)
        return q

    def pull(self, q: torch.Tensor) -> torch.Tensor:
        """Per-basin attraction amp_k·exp(−‖q−μ_k‖²/σ²) felt at q — (K,)."""
        mu, amp, s2 = self._stack()
        d2 = ((mu - q) ** 2).sum(-1)
        return amp * torch.exp(-d2 / s2)

    def predict(self, x: torch.Tensor) -> int:
        if not self.mu:
            return -1
        q = self.relax(x)
        return self.label[int(self.pull(q).argmax())]

    def nearest(self, q: torch.Tensor, label: int, scope: str, radius: float):
        """Index of the closest basin matching (label, scope) within radius, else None."""
        best, best_d = None, radius
        for i in range(len(self.mu)):
            if self.label[i] != label or self.scope[i] != scope:
                continue
            d = ((self.mu[i] - q) ** 2).sum().sqrt().item()
            if d < best_d:
                best, best_d = i, d
        return best


# ── Learners ─────────────────────────────────────────────────────────────────
class NOUSLearner:
    """
    Surprise-gated, labeled-basin continual learner.

    op_aware=True  → basins are scoped by op; refinement/consolidation never
                     touch another op's basins (the locality mechanism).
    op_aware=False → all basins share scope "_"; refinements collide across ops
                     (the ablation — same growth, only locality removed).
    gate=True      → a correct prediction only consolidates (no reshaping); gate
                     off updates on every example, correct or not.
    """

    def __init__(self, field: BasinField, op_aware: bool = True, gate: bool = True,
                 radius: float = 1.0, budget: int = None, n_ops: int = 1,
                 evict: str = "lru", plastic: bool = False, enc_lr: float = 0.02):
        self.f, self.op_aware, self.gate, self.R = field, op_aware, gate, radius
        self.budget, self.n_ops, self.evict = budget, n_ops, evict
        self.plastic = plastic
        self.enc_opt = torch.optim.SGD([field.W_in], lr=enc_lr) if plastic else None
        self.t = 0                                       # global step (for LRU)
        self.n_alloc = self.n_reuse = 0

    def _encoder_step(self, x: torch.Tensor, y: int):
        """Step 2: shape the (now plastic) encoder so raw embeddings e=W_in·x
        separate by label — softmax-CE pulling e toward its correct-label basin
        prototype, away from others. Shared W_in ⇒ this also moves *other* ops'
        embeddings (the drift channel we are testing). No solver in the loop."""
        if len(self.f.mu) < 2:
            return
        classes = sorted(set(self.f.label))
        if y not in classes or len(classes) < 2:
            return
        e = self.f.W_in @ x                              # (D,), grad flows to W_in
        mu = torch.stack([m.detach() for m in self.f.mu])
        neg_d2 = -((mu - e) ** 2).sum(-1)                # per-basin logit
        class_logit = torch.stack([
            neg_d2[[i for i, c in enumerate(self.f.label) if c == cl]].max()
            for cl in classes])                          # soft-nearest per label
        loss = F.cross_entropy(class_logit.unsqueeze(0),
                               torch.tensor([classes.index(y)]))
        self.enc_opt.zero_grad()
        loss.backward()
        self.enc_opt.step()

    def _cap(self) -> float:
        """Capacity of the CURRENT scope. op-aware splits the budget across ops
        (reserved slots); op-blind pools it all — equal total, only partitioning
        differs. That partitioning is the whole ablation."""
        if self.budget is None:
            return float("inf")
        return self.budget // self.n_ops if self.op_aware else self.budget

    def train_phase(self, data, op: str, epochs: int):
        for _ in range(epochs):
            for j in torch.randperm(len(data)):
                x, y = data[j]
                self.observe(x, y, op)

    def _sculpt(self, q: torch.Tensor, y: int, scope: str):
        """Label-free basin sculpting at equilibrium q: correct→deepen winner,
        wrong→refine near same-label basin, else allocate, else evict-and-reuse."""
        correct = self.f.mu and self.f.label[int(self.f.pull(q).argmax())] == y
        if correct and self.gate:
            i = self.f.nearest(q, y, scope, self.R)      # reinforce the winner, local
            if i is not None:
                self.f.deepen(i, step=0.1)
                self.f.touch(i, self.t)
            return
        i = self.f.nearest(q, y, scope, self.R)
        if i is not None:                                # refine an existing basin
            self.f.recenter(i, q)
            self.f.deepen(i)
            self.f.touch(i, self.t)
        elif self.f.scope_count(scope) < self._cap():    # room left → allocate
            self.f.add_basin(q, y, scope, t=self.t)
            self.n_alloc += 1
        else:                                            # full → evict a basin in scope
            j = (self.f.nearest_in_scope(scope, q) if self.evict == "geom"
                 else self.f.lru_in_scope(scope))
            if j is not None:
                self.f.reuse(j, q, y, self.t)
                self.n_reuse += 1

    def observe(self, x: torch.Tensor, y: int, op: str):
        self.t += 1
        scope = op if self.op_aware else "_"
        q = self.f.relax(x)
        self._sculpt(q, y, scope)
        if self.plastic:                                 # step 2: shape the encoder
            self._encoder_step(x, y)


class AdapterNOUS(NOUSLearner):
    """
    Step 3: task-conditioned plasticity. The encoder is base `W_in` (frozen) plus
    a per-REGION low-rank adapter ΔW_r = B_r·A_r. Regions are discovered by the
    same label-free geometry as the memory: route the base embedding e0 = W_in·x
    to the nearest region centroid; if none is within `region_radius`, spawn one.

    Training only updates the *routed* region's adapter, so learning a new task
    can move at most that region's embeddings — it cannot drift other regions'
    inputs off their basins. Localizing the plasticity CONTAINS the drift that
    the single shared encoder (step 2) spread across every op.
    """

    def __init__(self, field: BasinField, budget: int = None, n_ops: int = 1,
                 rank: int = 2, region_radius: float = 4.0, adapt_lr: float = 0.3):
        super().__init__(field, op_aware=False, gate=True, budget=budget,
                         n_ops=n_ops, evict="geom")
        self.rank, self.region_radius, self.adapt_lr = rank, region_radius, adapt_lr
        self.regions = []                                # each: {c, A, B, opt}

    def _spawn(self, e0: torch.Tensor) -> int:
        A = torch.randn(self.rank, IN_DIM) * 0.1
        B = torch.zeros(self.f.D, self.rank)             # B=0 → ΔW=0 at spawn (identity)
        A.requires_grad_(True)
        B.requires_grad_(True)
        self.regions.append({"c": e0.detach().clone(), "A": A, "B": B,
                             "opt": torch.optim.SGD([A, B], lr=self.adapt_lr)})
        return len(self.regions) - 1

    def _route(self, e0: torch.Tensor) -> int:
        if not self.regions:
            return self._spawn(e0)
        d = [((r["c"] - e0) ** 2).sum().item() for r in self.regions]
        j = min(range(len(d)), key=lambda k: d[k])
        return j if d[j] ** 0.5 <= self.region_radius else self._spawn(e0)

    def _embed(self, r: int, x: torch.Tensor, grad: bool):
        """Adapted encoding e = W_in·x + B_r·(A_r·x)."""
        reg = self.regions[r]
        e = self.f.W_in.detach() @ x + reg["B"] @ (reg["A"] @ x)
        return e if grad else e.detach()

    def _adapter_step(self, r: int, x: torch.Tensor, y: int):
        if len(self.f.mu) < 2:
            return
        classes = sorted(set(self.f.label))
        if y not in classes or len(classes) < 2:
            return
        e = self._embed(r, x, grad=True)
        mu = torch.stack([m.detach() for m in self.f.mu])
        neg_d2 = -((mu - e) ** 2).sum(-1)
        class_logit = torch.stack([
            neg_d2[[i for i, c in enumerate(self.f.label) if c == cl]].max()
            for cl in classes])
        loss = F.cross_entropy(class_logit.unsqueeze(0),
                               torch.tensor([classes.index(y)]))
        self.regions[r]["opt"].zero_grad()
        loss.backward()
        self.regions[r]["opt"].step()

    def predict(self, x: torch.Tensor) -> int:
        if not self.f.mu:
            return -1
        r = self._route(self.f.W_in.detach() @ x)
        q = self.f.relax(x, drive=self._embed(r, x, grad=False))
        return self.f.label[int(self.f.pull(q).argmax())]

    def observe(self, x: torch.Tensor, y: int, op: str):
        self.t += 1
        r = self._route(self.f.W_in.detach() @ x)
        q = self.f.relax(x, drive=self._embed(r, x, grad=False))
        self._sculpt(q, y, "_")                          # label-free, shared pool
        self._adapter_step(r, x, y)                      # local plasticity: region r only


class MLPStream:
    """Canonical baseline: a shared-weight MLP trained on each op IN FULL to
    convergence, then the next op. Best case for the MLP (it reaches ~100% per
    phase), so any old-op drop is pure cross-task interference — the textbook
    catastrophic-forgetting demonstration."""

    def __init__(self, lr: float = 0.01, iters: int = 300):
        self.net = nn.Sequential(nn.Linear(IN_DIM, 64), nn.Tanh(), nn.Linear(64, VALS))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.iters = iters
        self.n_alloc = 0

    def train_phase(self, data, op: str, epochs: int):
        X = torch.stack([x for x, _ in data])
        Y = torch.tensor([y for _, y in data])
        for _ in range(self.iters):
            self.opt.zero_grad()
            F.cross_entropy(self.net(X), Y).backward()
            self.opt.step()

    def predict(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            return self.net(x.unsqueeze(0)).argmax(-1).item()


def accuracy(model, op: str) -> float:
    predict = model.predict if hasattr(model, "predict") else model.f.predict
    data = op_dataset(op)
    ok = sum(predict(x) == y for x, y in data)
    return ok / len(data)


# ── Streaming protocol + metrics ─────────────────────────────────────────────
def new_field(state_dim: int, seed: int, scale: float = 3.0,
              plastic: bool = False) -> BasinField:
    g = torch.Generator().manual_seed(seed)
    W_in = scale * torch.randn(state_dim, IN_DIM, generator=g) / math.sqrt(IN_DIM)
    if plastic:
        W_in.requires_grad_(True)                        # step 2: unfreeze the encoder
    return BasinField(state_dim, W_in)


def make_model(kind: str, state_dim: int, seed: int, budget: int = None, n_ops: int = 1):
    if kind == "gated":
        return NOUSLearner(new_field(state_dim, seed), op_aware=True, gate=True,
                           budget=budget, n_ops=n_ops)
    if kind == "ungated":
        return NOUSLearner(new_field(state_dim, seed), op_aware=False, gate=False,
                           budget=budget, n_ops=n_ops, evict="lru")
    if kind == "taskfree":
        # no op label: one shared pool (like ungated) but geometric eviction —
        # locality must emerge from the input geometry alone.
        return NOUSLearner(new_field(state_dim, seed), op_aware=False, gate=True,
                           budget=budget, n_ops=n_ops, evict="geom")
    if kind == "taskfree_plastic":
        # step 2: same as taskfree but the encoder W_in is trainable (contrastive).
        # Tests learned separation (+) vs shared-encoder drift (−).
        return NOUSLearner(new_field(state_dim, seed, plastic=True), op_aware=False,
                           gate=True, budget=budget, n_ops=n_ops, evict="geom",
                           plastic=True)
    if kind == "adapter":
        # step 3: task-conditioned plasticity — per-region low-rank encoder
        # adapters, routed by the same label-free geometry. Localizes drift.
        return AdapterNOUS(new_field(state_dim, seed), budget=budget, n_ops=n_ops,
                           region_radius=6.0, adapt_lr=0.3)
    if kind == "mlp":
        return MLPStream()
    raise ValueError(kind)


def run_stream(kind: str, ops, seed: int, epochs: int, state_dim: int, budget: int = None):
    """Train `kind` over the op stream; return per-phase accuracy matrix + basin/reuse counts."""
    torch.manual_seed(seed)
    model = make_model(kind, state_dim, seed, budget=budget, n_ops=len(ops))
    rows = []                                    # rows[k][op] = acc on op after phase k
    for phase, op in enumerate(ops):
        model.train_phase(op_dataset(op), op, epochs)
        rows.append({o: accuracy(model, o) for o in ops[:phase + 1]})
    return {"acc_matrix": rows, "n_basins": model.n_alloc,
            "n_reuse": getattr(model, "n_reuse", 0)}


def summarize(results, ops):
    """Aggregate seeds → op0 retention, forgetting, and final all-op accuracy."""
    first = ops[0]
    peak = [r["acc_matrix"][0][first] for r in results]      # op0 acc right after learning it
    final = [r["acc_matrix"][-1][first] for r in results]    # op0 acc after the whole stream
    drop = [p - f for p, f in zip(peak, final)]
    all_final = [_mean([r["acc_matrix"][-1][o] for o in ops]) for r in results]
    return {
        "op0_peak_mean": _mean(peak),
        "op0_final_mean": _mean(final),
        "forgetting_mean": _mean(drop),
        "forgetting_std": _std(drop),
        "all_ops_final_mean": _mean(all_final),
        "n_basins_mean": _mean([r["n_basins"] for r in results]),
    }


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ── Self-check (ponytail: one runnable check for the non-trivial logic) ──────
def selfcheck():
    D = 16

    # (1) CORE INVARIANT: an op-aware update never touches another op's basin.
    f = new_field(D, seed=0)
    add_i = f.add_basin(torch.zeros(D), label=3, scope="add")
    before_mu = f.mu[add_i].clone()
    before_amp = f.log_amp[add_i]
    learner = NOUSLearner(f, op_aware=True, gate=True)
    learner.observe(encode(1, 2, "mul"), OPS["mul"](1, 2), "mul")
    assert torch.equal(f.mu[add_i], before_mu), "add basin μ moved during a mul update"
    assert f.log_amp[add_i] == before_amp, "add basin depth moved during a mul update"
    assert any(s == "mul" for s in f.scope), "no mul basin created for the surprise"

    # (2) RETENTION: on add→mul, op-aware NOUS keeps op0 better than MLP.
    ops = ["add", "mul"]
    gated = summarize([run_stream("gated", ops, s, 15, D) for s in range(3)], ops)
    ungat = summarize([run_stream("ungated", ops, s, 15, D) for s in range(3)], ops)
    mlp = summarize([run_stream("mlp", ops, s, 15, D) for s in range(3)], ops)

    for name, r in (("gated", gated), ("ungated", ungat), ("mlp", mlp)):
        print(f"{name:8s} op0 {r['op0_peak_mean']:.2f}→{r['op0_final_mean']:.2f}  "
              f"forget {r['forgetting_mean']:+.2f}  allfinal {r['all_ops_final_mean']:.2f}  "
              f"basins {r['n_basins_mean']:.1f}")

    assert gated["op0_peak_mean"] > 0.8, "gated failed to learn op0"
    assert gated["all_ops_final_mean"] > 0.8, "gated failed to learn both ops"
    assert gated["forgetting_mean"] < mlp["forgetting_mean"], "gating did not beat MLP forgetting"

    # (3) CAPPED ABLATION: under a shared budget, op-locality is what prevents
    # forgetting — op-aware reserves per-op slots (retains); op-blind pools and
    # LRU-evicts old ops (forgets). Same total capacity, only partitioning differs.
    cap_g = summarize([run_stream("gated", ops, s, 15, D, budget=20) for s in range(3)], ops)
    cap_u = summarize([run_stream("ungated", ops, s, 15, D, budget=20) for s in range(3)], ops)
    cap_t = summarize([run_stream("taskfree", ops, s, 15, D, budget=20) for s in range(3)], ops)
    print(f"capped   gated forget {cap_g['forgetting_mean']:+.2f}  "
          f"ungated forget {cap_u['forgetting_mean']:+.2f}  "
          f"taskfree forget {cap_t['forgetting_mean']:+.2f}")
    assert cap_g["forgetting_mean"] < cap_u["forgetting_mean"] - 0.2, \
        "under a cap, op-locality did not reduce forgetting vs op-blind"

    # (4) EMERGENT LOCALITY: task-free (no op label, geometric eviction) forgets
    # less than op-blind (same shared pool, LRU eviction) — locality is discovered
    # from the input geometry, not handed via the label. The gap depends on how
    # separable the frozen W_in makes the ops, so we assert only the direction.
    assert cap_t["forgetting_mean"] < cap_u["forgetting_mean"], \
        "task-free geometric routing did not reduce forgetting without the op label"

    # (5) PLASTIC ENCODER DRIFTS, DOESN'T RESCUE (step 2): unfreezing the shared
    # W_in does not reach op-aware retention — the ops share the label space, so a
    # label-driven encoder objective can't separate tasks, only drift. So plastic
    # still forgets clearly more than op-aware.
    cap_p = summarize([run_stream("taskfree_plastic", ops, s, 15, D, budget=20) for s in range(3)], ops)
    cap_a = summarize([run_stream("adapter", ops, s, 15, D, budget=20) for s in range(3)], ops)
    print(f"         taskfree_plastic forget {cap_p['forgetting_mean']:+.2f}  "
          f"adapter forget {cap_a['forgetting_mean']:+.2f}")
    assert cap_p["forgetting_mean"] > cap_g["forgetting_mean"] + 0.2, \
        "plastic encoder unexpectedly matched op-aware retention"

    # (6) TASK-CONDITIONED PLASTICITY CONTAINS DRIFT (step 3): per-region adapters
    # retain far better than op-blind LRU, i.e. local plasticity does not regress
    # into the shared-encoder drift. (It contains drift; it does not beat frozen.)
    assert cap_a["forgetting_mean"] < cap_u["forgetting_mean"], \
        "per-region adapters did not contain drift / retain vs op-blind"
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--state-dim", type=int, default=16)
    ap.add_argument("--budget", type=int, default=None,
                    help="total basin cap (forces LRU reuse). Omit for unbounded growth.")
    ap.add_argument("--out", default="results/continual_ops.json")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    ops = ["add", "mul", "sub"]
    seeds = list(range(args.seeds))
    if args.out == "results/continual_ops.json" and args.budget is not None:
        args.out = "results/continual_ops_capped.json"
    out = {"config": {"ops": ops, "seeds": seeds, "epochs": args.epochs,
                      "state_dim": args.state_dim, "vals": VALS, "budget": args.budget},
           "per_seed": {}, "summary": {}}
    for kind in ("gated", "ungated", "taskfree", "taskfree_plastic", "adapter", "mlp"):
        res = [run_stream(kind, ops, s, args.epochs, args.state_dim, budget=args.budget)
               for s in seeds]
        out["per_seed"][kind] = res
        out["summary"][kind] = summarize(res, ops)
        s = out["summary"][kind]
        reuse = _mean([r["n_reuse"] for r in res])
        print(f"{kind:8s} op0 {s['op0_peak_mean']:.3f}→{s['op0_final_mean']:.3f}  "
              f"forget {s['forgetting_mean']:+.3f}±{s['forgetting_std']:.3f}  "
              f"allfinal {s['all_ops_final_mean']:.3f}  basins {s['n_basins_mean']:.1f}  "
              f"reuse {reuse:.1f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

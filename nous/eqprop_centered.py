"""
Centered Equilibrium Propagation (C-EP)
Scellier, Ernoult, Kendall, Kumar — NeurIPS 2023
https://arxiv.org/abs/2312.15103

Standard EP (P-EP) uses one nudge direction (+β), giving a first-order
Taylor approximation of the gradient — the O(β) bias causes overfitting.

C-EP uses TWO nudge phases: +β (toward correct answer) and -β (away from it).
The symmetric difference cancels the O(β) bias, giving a SECOND-ORDER
estimate that directly optimizes the cost (not a lower bound).

C-EP update:
  q_pos = argmin_q [E(x,q) + β·C(q, y)]    ← positive nudge
  q_neg = argmin_q [E(x,q) - β·C(q, y)]    ← negative nudge (β appears with - sign)
  Δθ = (η/2β) [∂E(q_pos;θ)/∂θ − ∂E(q_neg;θ)/∂θ]

Why this is correct:
  E[∂E(q_pos)/∂θ] = ∂E(q_free)/∂θ + β·(∂²E/∂θ∂q·∂C/∂q) + O(β²)
  E[∂E(q_neg)/∂θ] = ∂E(q_free)/∂θ - β·(∂²E/∂θ∂q·∂C/∂q) + O(β²)
  Difference/2β = (∂²E/∂θ∂q·∂C/∂q) + O(β)  ← the true gradient at q_free

Decoder update: standard CE at q_pos (gradient flows from positive attractor).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenteredEqProp:
    """
    C-EP with optional heterogeneous time constants (Kubo et al. 2026).

    tau: per-neuron time constants drawn from log-Normal distribution.
         Prevents resonance and collective oscillations that destabilize free phase.
    """

    def __init__(
        self,
        energy_net,
        solver,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.5,
        tau: torch.Tensor = None,
    ):
        self.E = energy_net
        self.solver = solver
        self.decoder = decoder
        self.optimizer = optimizer
        self.beta = beta
        self.tau = tau  # (state_dim,) time constants, or None for uniform

    def _solve(self, x, q0, cost_sign=0.0, target=None):
        """
        Solve the EqProp fixed-point equation with optional nudge.
        cost_sign: 0 = free phase, +1 = positive nudge, -1 = negative nudge
        """
        if cost_sign == 0.0 or target is None:
            return self.solver.solve(x, q0)

        def nudge_energy(q):
            ce = F.cross_entropy(
                self.decoder(q).unsqueeze(0), target.unsqueeze(0)
            )
            return cost_sign * self.beta * ce

        return self.solver.solve(x, q0, extra_energy_fn=nudge_energy)

    def _param_grads(self, x, q_star):
        for p in self.E.parameters():
            if p.grad is not None:
                p.grad.zero_()
        E_val = self.E.forward(x, q_star.detach())
        E_val.sum().backward()
        return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in self.E.named_parameters()}

    def step(self, x: torch.Tensor, target: torch.Tensor,
             q0_override: torch.Tensor = None,
             x_with_grad: torch.Tensor = None):
        """
        One C-EP step on example (x, target).

        x            — detached input embedding (for ODE, no grad tracking needed)
        x_with_grad  — same embedding WITH gradient tape open, for backpropping
                       C-EP signal through the projector. Pass projector(img)
                       before detaching.

        Nudge phases start from q_free (not q0): they're already near equilibrium
        so they need far fewer ODE steps to converge.

        Projector gradient via EqProp: coupling = −xᵀ W_in q, so ∂E/∂x = −W_in^T q.
        C-EP signal on x: Δx = (∂E(q_pos)/∂x − ∂E(q_neg)/∂x) / 2β
                               = W_in^T (q_neg − q_pos) / 2β

        Returns: (loss, q_free, q_pos, q_neg)
        """
        q0 = q0_override if q0_override is not None else torch.zeros(self.E.state_dim)

        # ── Free phase ────────────────────────────────────────────────────────
        q_free = self._solve(x, q0, cost_sign=0.0)
        with torch.no_grad():
            loss = F.cross_entropy(
                self.decoder(q_free).unsqueeze(0), target.unsqueeze(0)
            ).item()

        # ── Nudge phases start from q_free (already near equilibrium) ─────────
        q_pos = self._solve(x, q_free, cost_sign=+1.0, target=target)
        grads_pos = self._param_grads(x, q_pos)

        q_neg = self._solve(x, q_free, cost_sign=-1.0, target=target)
        grads_neg = self._param_grads(x, q_neg)

        # ── C-EP update: symmetric difference cancels O(β) bias ──────────────
        self.optimizer.zero_grad()
        for name, param in self.E.named_parameters():
            if param.requires_grad:
                param.grad = (1.0 / (2.0 * self.beta)) * (
                    grads_pos[name] - grads_neg[name]
                )

        # ── Projector gradient via C-EP signal on x ───────────────────────────
        # coupling = −xᵀ W_in q  →  ∂coupling/∂x = −W_in q
        # ∂E(q_pos)/∂x − ∂E(q_neg)/∂x = W_in^T(q_neg − q_pos) / 2β
        if x_with_grad is not None and x_with_grad.requires_grad:
            with torch.no_grad():
                W = self.E.W_in.weight   # (state_dim, input_dim)
                dx = W.t() @ (q_neg - q_pos) / (2.0 * self.beta)
            x_with_grad.backward(dx.detach())

        # ── Decoder: standard CE at positive equilibrium ──────────────────────
        logits_pos = self.decoder(q_pos.detach())
        ce = F.cross_entropy(logits_pos.unsqueeze(0), target.unsqueeze(0))
        ce.backward()

        self.optimizer.step()

        return loss, q_free.detach(), q_pos.detach(), q_neg.detach()

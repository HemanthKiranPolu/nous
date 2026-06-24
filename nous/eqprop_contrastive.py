"""
Contrastive Basin EqProp.

Standard EqProp: one nudge phase toward correct class.
  Δθ = (1/ε)[∂E(q+)/∂θ − ∂E(q_free)/∂θ]

Contrastive Basin EqProp: positive nudge + negative nudge simultaneously.
  Δθ = (1/ε)[∂E(q+)/∂θ − ∂E(q_free)/∂θ]   ← deepen correct basin
       − (γ/ε)[∂E(q−)/∂θ − ∂E(q_free)/∂θ]  ← flatten hardest wrong basin

q+: nudge toward correct class (standard EqProp nudge)
q−: nudge toward hardest negative (argmax of wrong-class logits at free equilibrium)

Physical interpretation:
  The positive update carves a deeper basin for the correct answer.
  The negative update raises the floor of the most-confused competitor,
  increasing the free-energy difference ΔF = F_wrong − F_correct.
  Together they implement contrastive Hebbian learning in continuous state space.

This is distinct from contrastive learning (SimCLR, etc.):
  - No data augmentation needed
  - Operates on energy landscape, not embedding cosine similarity
  - Negative is chosen by the model's own confusion, not random pairing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEqProp:
    def __init__(
        self,
        energy_net,
        solver,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        eps: float = 0.3,
        gamma: float = 0.5,
        fast_dim: int = 16,
        use_dual_timescale: bool = True,
    ):
        self.E = energy_net
        self.solver = solver
        self.decoder = decoder
        self.optimizer = optimizer
        self.eps = eps
        self.gamma = gamma
        self.fast_dim = fast_dim
        self.use_dual_timescale = use_dual_timescale

    def _solve(self, x, q0, extra_energy_fn=None):
        if self.use_dual_timescale and hasattr(self.solver, 'solve_dual_timescale'):
            return self.solver.solve_dual_timescale(
                x, q0, fast_dim=self.fast_dim,
                extra_energy_fn=extra_energy_fn
            )
        return self.solver.solve(x, q0, extra_energy_fn=extra_energy_fn)

    def step(self, x: torch.Tensor, target: torch.Tensor,
             q0_override: torch.Tensor = None):
        """
        One contrastive EqProp step.
        Returns: (loss, q_free, q_pos, q_neg)
        """
        q0 = q0_override if q0_override is not None else torch.zeros(self.E.state_dim)

        # ── Free phase ────────────────────────────────────────────────────────
        q_free = self._solve(x, q0)
        logits_free = self.decoder(q_free)
        loss = F.cross_entropy(logits_free.unsqueeze(0), target.unsqueeze(0))
        grads_free = self._param_grads(x, q_free)

        # ── Positive nudge (toward correct class) ─────────────────────────────
        def pos_energy(q):
            return self.eps * F.cross_entropy(
                self.decoder(q).unsqueeze(0), target.unsqueeze(0)
            )

        q_pos = self._solve(x, q0, extra_energy_fn=pos_energy)
        grads_pos = self._param_grads(x, q_pos)

        # ── Negative nudge (toward hardest wrong class) ───────────────────────
        with torch.no_grad():
            wrong_logits = logits_free.clone()
            wrong_logits[target.item()] = float('-inf')
            hard_neg_class = wrong_logits.argmax()

        def neg_energy(q):
            return self.eps * F.cross_entropy(
                self.decoder(q).unsqueeze(0), hard_neg_class.unsqueeze(0)
            )

        q_neg = self._solve(x, q0, extra_energy_fn=neg_energy)
        grads_neg = self._param_grads(x, q_neg)

        # ── Contrastive EqProp update ─────────────────────────────────────────
        self.optimizer.zero_grad()

        for name, param in self.E.named_parameters():
            if param.requires_grad:
                pos_signal = grads_pos[name] - grads_free[name]
                neg_signal = grads_neg[name] - grads_free[name]
                param.grad = (1.0 / self.eps) * (pos_signal - self.gamma * neg_signal)

        # Decoder: supervised CE at positive equilibrium
        logits_pos = self.decoder(q_pos)
        ce_pos = F.cross_entropy(logits_pos.unsqueeze(0), target.unsqueeze(0))
        ce_pos.backward()

        self.optimizer.step()

        return loss.item(), q_free.detach(), q_pos.detach(), q_neg.detach()

    def _param_grads(self, x, q_star):
        for p in self.E.parameters():
            if p.grad is not None:
                p.grad.zero_()
        E_val = self.E.forward(x, q_star.detach())
        E_val.sum().backward()
        return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in self.E.named_parameters()}

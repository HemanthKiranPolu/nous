"""
Langevin EqProp ODE solver.

Standard EqProp: dq = −∂E/∂q · dt  (deterministic)
Langevin EqProp: dq = −∂E/∂q · dt + √(2dt/β) · dW  (stochastic during training)

Physical basis: Einstein-Smoluchowski relation.
At inverse temperature β, thermal fluctuations have amplitude √(2kT) = √(2/β).
Narrow attractor basins (overfitting attractors) are thermodynamically unstable —
they dissolve under noise. Only wide, well-separated basins survive, which
correspond to genuinely learned representations rather than memorized examples.

Training mode: stochastic (β from annealer, noise decays as system cools)
Eval mode:     deterministic (noise=0, converges to deepest basin)
"""

import torch
import math


class LangevinSolver:
    def __init__(self, energy_net, dt: float = 0.1, n_steps: int = 50,
                 delta: float = 1e-3, training: bool = True):
        self.E = energy_net
        self.dt = dt
        self.n_steps = n_steps
        self.delta = delta
        self.training = training
        self._beta = 1.0

    def set_beta(self, beta: float):
        self._beta = max(beta, 0.1)

    def set_training(self, mode: bool):
        self.training = mode

    def _noise_scale(self):
        if not self.training:
            return 0.0
        return math.sqrt(2.0 * self.dt / self._beta)

    def solve(self, x: torch.Tensor, q0: torch.Tensor,
              extra_energy_fn=None, n_steps_override: int = None) -> torch.Tensor:
        q = q0.clone().detach()
        n = n_steps_override or self.n_steps
        sigma = self._noise_scale()

        for _ in range(n):
            with torch.enable_grad():
                q_g = q.detach().requires_grad_(True)
                E_val = self.E.forward(x, q_g)
                if extra_energy_fn is not None:
                    E_val = E_val + extra_energy_fn(q_g)
                dE_dq = torch.autograd.grad(E_val.sum(), q_g)[0]

            force = -dE_dq.detach()
            q = q + force * self.dt
            if sigma > 0:
                q = q + sigma * torch.randn_like(q)

            if force.norm() < self.delta:
                break

        return q.detach()

    def solve_dual_timescale(self, x: torch.Tensor, q0: torch.Tensor,
                              fast_dim: int = 16, n_fast: int = 10,
                              extra_energy_fn=None) -> torch.Tensor:
        """
        Dual-timescale relaxation: adiabatic separation of fast/slow modes.

        q = [q_fast | q_slow]
        Phase 1: relax q_fast only (n_fast steps) — low-level features settle
        Phase 2: relax full q jointly (n_steps total) — global structure emerges

        Physically motivated by adiabatic theorem: fast degrees of freedom
        equilibrate before slow ones when timescales are well-separated.
        """
        q = q0.clone().detach()
        sigma = self._noise_scale()

        # Phase 1: fast subspace only
        for _ in range(n_fast):
            with torch.enable_grad():
                q_g = q.detach().requires_grad_(True)
                E_val = self.E.forward(x, q_g)
                if extra_energy_fn is not None:
                    E_val = E_val + extra_energy_fn(q_g)
                dE_dq = torch.autograd.grad(E_val.sum(), q_g)[0]

            force = -dE_dq.detach()
            # Update only fast dims
            q = q.clone()
            q[:fast_dim] = q[:fast_dim] + force[:fast_dim] * self.dt
            if sigma > 0:
                q[:fast_dim] = q[:fast_dim] + sigma * torch.randn(fast_dim)

        # Phase 2: full joint relaxation
        for _ in range(self.n_steps):
            with torch.enable_grad():
                q_g = q.detach().requires_grad_(True)
                E_val = self.E.forward(x, q_g)
                if extra_energy_fn is not None:
                    E_val = E_val + extra_energy_fn(q_g)
                dE_dq = torch.autograd.grad(E_val.sum(), q_g)[0]

            force = -dE_dq.detach()
            q = q + force * self.dt
            if sigma > 0:
                q = q + sigma * torch.randn_like(q)

            if force.norm() < self.delta:
                break

        return q.detach()

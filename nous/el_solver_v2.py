"""
True second-order Euler-Lagrange solver with inertia and damping.

The NOUS spec promised:  M(q)q̈ + γq̇ = −∂E/∂q
All prior implementations used first-order gradient descent (overdamped limit: γ→∞, M→0).

This solver implements the FULL second-order dynamics:
  dq/dt = p / m(q)              (position update via momentum)
  dp/dt = −∂E/∂q − γ·p/m(q)   (momentum update via force − damping)

where m(q) = scalar inertia field (position-dependent mass), learned as exp(MLP(q)).

Physical consequence: the system OSCILLATES before settling.
The oscillation frequency near q* is:
  ω_k(q*) ≈ √(λ_k(∂²E/∂q²|_{q*}) / m(q*))

These frequencies are the SPECTRAL FINGERPRINT of the attractor basin —
unique to each class, richer than position alone.

Key APIs:
  solve(x, q0)                 → q* (standard convergence, records trajectory)
  get_spectral_fingerprint(x, q0, n_eigs) → top eigenvalues of Hessian at q*
  trajectory                   → full (T, d) trajectory for analysis
"""

import torch
import torch.nn as nn
import math


class InertiaNet(nn.Module):
    """Learned scalar inertia field m(q) = exp(MLP(q)).
    Positive by construction. Small MLP to keep compute low."""
    def __init__(self, state_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # Initialize near m=1 (unit mass)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.net(q).squeeze(-1))


class EulerLagrangeSolverV2:
    """
    True second-order EL solver with position-dependent inertia.

    Reduces to first-order (original solver) when gamma is very large,
    but operates in the oscillatory underdamped regime by default.
    """

    def __init__(self, energy_net, inertia_net: InertiaNet = None,
                 dt: float = 0.05, n_steps: int = 200,
                 delta: float = 1e-3, gamma: float = 0.5,
                 record_trajectory: bool = False):
        self.E = energy_net
        self.M = inertia_net
        self.dt = dt
        self.n_steps = n_steps
        self.delta = delta
        self.gamma = gamma
        self.record_trajectory = record_trajectory
        self.trajectory = None

    def _inertia(self, q: torch.Tensor) -> torch.Tensor:
        if self.M is None:
            return torch.ones(1)
        with torch.no_grad():
            return self.M(q).clamp(min=0.1, max=10.0)

    def _force(self, x: torch.Tensor, q: torch.Tensor,
               extra_energy_fn=None) -> torch.Tensor:
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E_val = self.E.forward(x, q_g)
            if extra_energy_fn is not None:
                E_val = E_val + extra_energy_fn(q_g)
            dE_dq = torch.autograd.grad(E_val.sum(), q_g)[0]
        return -dE_dq.detach()

    def solve(self, x: torch.Tensor, q0: torch.Tensor,
              extra_energy_fn=None) -> torch.Tensor:
        """
        Run second-order EL ODE: M(q)q̈ + γq̇ = −∂E/∂q
        Returns equilibrium q* and optionally stores trajectory.
        """
        q = q0.clone().detach()
        p = torch.zeros_like(q)       # momentum starts at rest
        traj = [q.clone()] if self.record_trajectory else None

        for step in range(self.n_steps):
            m = self._inertia(q)
            force = self._force(x, q, extra_energy_fn)

            # Symplectic Euler (momentum-first for better energy conservation)
            damping = self.gamma * p / m
            p = p + (force - damping) * self.dt
            q = q + p / m * self.dt

            if traj is not None:
                traj.append(q.clone())

            # Convergence: kinetic energy small
            if (p / m).norm() < self.delta:
                break

        self.trajectory = torch.stack(traj) if traj is not None else None
        return q.detach()

    def get_spectral_fingerprint(self, x: torch.Tensor, q_star: torch.Tensor,
                                  n_eigs: int = 8) -> torch.Tensor:
        """
        Compute top-n Hessian eigenvalues at attractor q*.
        These are the squared natural frequencies of small oscillations:
            ω_k² = λ_k(∂²E/∂q²) / m(q*)

        Returns: tensor of shape (n_eigs,) — the spectral fingerprint.
        Uses Lanczos (random probes) for efficiency in high dimensions.
        """
        d = q_star.shape[0]
        m = self._inertia(q_star).item()

        if d <= 16:
            # Exact Hessian for small state dims
            with torch.enable_grad():
                q_g = q_star.detach().requires_grad_(True)
                E_val = self.E.forward(x, q_g)
                grad = torch.autograd.grad(E_val.sum(), q_g, create_graph=True)[0]
                H = torch.zeros(d, d)
                for i in range(d):
                    g2 = torch.autograd.grad(grad[i], q_g, retain_graph=True,
                                              allow_unused=True)[0]
                    if g2 is not None:
                        H[i] = g2.detach()
            eigs = torch.linalg.eigvalsh(H)
            top_eigs = eigs[-n_eigs:].flip(0)
        else:
            # Stochastic Lanczos for large dims
            eig_estimates = []
            with torch.enable_grad():
                q_g = q_star.detach().requires_grad_(True)
                E_val = self.E.forward(x, q_g)
                grad = torch.autograd.grad(E_val.sum(), q_g, create_graph=True)[0]
                for _ in range(n_eigs * 4):
                    v = torch.randn_like(q_g)
                    v = v / (v.norm() + 1e-8)
                    Hv = torch.autograd.grad((grad * v.detach()).sum(), q_g,
                                             retain_graph=True)[0]
                    rayleigh = (v.detach() * Hv.detach()).sum().item()
                    eig_estimates.append(rayleigh)
            eig_estimates.sort(reverse=True)
            top_eigs = torch.tensor(eig_estimates[:n_eigs])

        # Natural frequencies: ω² = λ / m
        natural_freqs = top_eigs / m
        return natural_freqs.float()

import torch


class EulerLagrangeSolver:
    """
    Solves overdamped gradient flow with clamped input x:
        q̇ = −∂E(x, q; θ)/∂q = force(x, q)

    The input x is fixed (clamped) throughout. Different x → different equilibria.
    Uses simple Euler integration — robust, no torchdiffeq dependency issues.
    """

    def __init__(self, energy_net, dt: float = 0.1, n_steps: int = 200, delta: float = 1e-3):
        self.E = energy_net
        self.dt = dt
        self.n_steps = n_steps
        self.delta = delta

    def solve(self, x: torch.Tensor, q0: torch.Tensor, extra_energy_fn=None) -> torch.Tensor:
        """
        Run dynamics to equilibrium with input x clamped.
        extra_energy_fn: optional ε·C(decode(q), target) added for nudged phase.
        Returns q* (equilibrium state).
        """
        q = q0.clone().float()
        x = x.float()

        for _ in range(self.n_steps):
            force = self.E.force(x, q)
            if extra_energy_fn is not None:
                # Add force from extra energy via finite difference
                eps_fd = 1e-3
                q_extra = q.detach().requires_grad_(True)
                extra_E = extra_energy_fn(q_extra)
                extra_force = -torch.autograd.grad(extra_E, q_extra)[0].detach()
                force = force + extra_force

            q = q + self.dt * force

            if force.norm().item() < self.delta:
                break

        return q.detach()

    def solve_trajectory(self, x: torch.Tensor, q0: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """Return full trajectory as tensor (T, d) for visualization."""
        q = q0.clone().float()
        x = x.float()
        traj = [q.clone()]
        steps = n_steps or self.n_steps
        for _ in range(steps):
            with torch.no_grad():
                force = self.E.force(x, q)
                q = q + self.dt * force
                traj.append(q.clone())
        return torch.stack(traj)

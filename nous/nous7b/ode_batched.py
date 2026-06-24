"""
Batched ODE solver: process B independent sentences in parallel.

Within a sentence, tokens are sequential (stateful carry).
Across sentences, computation is fully independent → batch them.

q shape: (B, state_dim)
x shape: (B, embed_dim)
"""
import torch
import torch.nn.functional as F


class BatchedEulerODE:
    """
    Vectorized Euler integration for a batch of independent rollouts.
    Each sample b has its own q_b, x_b, and target_b.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.dt    = cfg.dt
        self.tol   = cfg.ode_tol

    def _force_batch(self, x_batch: torch.Tensor, q_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute −∂E/∂q for each sample in the batch.
        x_batch: (B, embed_dim)
        q_batch: (B, state_dim)
        Returns force: (B, state_dim)
        """
        with torch.enable_grad():
            q_g = q_batch.detach().requires_grad_(True)
            # Vectorize energy over batch via vmap-style loop (MPS-safe)
            E_sum = sum(
                self.model(x_batch[b], q_g[b])
                for b in range(q_batch.shape[0])
            )
            grad = torch.autograd.grad(E_sum, q_g)[0]
        return -grad.detach()

    def solve_batch(self, x_batch: torch.Tensor, q0_batch: torch.Tensor,
                    n_steps: int, extra_fns=None) -> torch.Tensor:
        """
        x_batch:  (B, embed_dim)
        q0_batch: (B, state_dim)
        extra_fns: list of B callables (q_b → scalar) for nudged phase, or None
        Returns q*: (B, state_dim)
        """
        B = q0_batch.shape[0]
        q = q0_batch.detach().clone()

        for _ in range(n_steps):
            force = self._force_batch(x_batch, q)

            if extra_fns is not None:
                for b in range(B):
                    q_b = q[b].detach().requires_grad_(True)
                    ef  = extra_fns[b](q_b)
                    ef_force = -torch.autograd.grad(ef, q_b)[0].detach()
                    force[b] = force[b] + ef_force

            q = q + self.dt * force

            if force.norm(dim=-1).max().item() < self.tol:
                break

        return q.detach()

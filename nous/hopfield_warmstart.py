"""
Modern Hopfield Network warm-start for EqProp free-phase initialization.
Ramsauer et al. 2020 — "Hopfield Networks is All You Need"
https://arxiv.org/abs/2008.02217

Problem: EqProp free phase starts from zeros (or class EMA), wasting
~200 ODE steps converging from a random initial state.

Solution: Use a Modern Hopfield Network as an episodic memory.
- Store seen training equilibria q* indexed by input projections x
- At inference: one Hopfield retrieval step finds the nearest attractor
- Use retrieved state as q0 → free phase converges in ~30 steps instead of ~200

Modern Hopfield update (one step, exact retrieval for β→∞):
  q_init = X · softmax(β_hop · X^T · ξ)

Where:
  X = stored patterns (state_dim × n_stored)
  ξ = query (projected input embedding, embed_dim → state_dim)
  β_hop = inverse temperature (higher = sharper retrieval)

This acts as an associative memory: the Hopfield network finds which
training example the new input is most similar to, then uses that
example's equilibrium state as a warm start.
"""

import torch
import torch.nn as nn


class HopfieldWarmStart:
    """
    Episodic warm-start memory using Modern Hopfield dynamics.

    Maintains a rolling buffer of recent (projection, equilibrium) pairs.
    At query time, uses one-step Hopfield retrieval to get warm q0.
    """

    def __init__(self, embed_dim: int, state_dim: int,
                 capacity: int = 500, beta_hop: float = 4.0):
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.capacity = capacity
        self.beta_hop = beta_hop

        # Circular buffers
        self._keys   = torch.zeros(capacity, embed_dim)   # input projections
        self._values = torch.zeros(capacity, state_dim)   # equilibrium states
        self._ptr    = 0
        self._full   = False

    @property
    def n_stored(self):
        return self.capacity if self._full else self._ptr

    def store(self, x_proj: torch.Tensor, q_star: torch.Tensor):
        """Store a (projection, equilibrium) pair."""
        idx = self._ptr % self.capacity
        self._keys[idx]   = x_proj.detach()
        self._values[idx] = q_star.detach()
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True
            self._ptr  = self._ptr % self.capacity

    def retrieve(self, x_proj: torch.Tensor) -> torch.Tensor:
        """
        One-step Hopfield retrieval: X · softmax(β · X^T · ξ)
        Returns warm-start q0, or zeros if memory is empty.
        """
        n = self.n_stored
        if n == 0:
            return torch.zeros(self.state_dim)

        keys   = self._keys[:n]     # (n, embed_dim)
        values = self._values[:n]   # (n, state_dim)

        # Similarity: (n,)
        sim = keys @ x_proj / (x_proj.norm() + 1e-8)
        weights = torch.softmax(self.beta_hop * sim, dim=0)  # (n,)

        # Weighted combination of stored equilibria
        q_init = (weights.unsqueeze(1) * values).sum(0)      # (state_dim,)
        return q_init.detach()

    def retrieve_topk(self, x_proj: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Retrieve using only the top-k most similar stored patterns.
        More robust than full softmax when memory is large and noisy.
        """
        n = self.n_stored
        if n == 0:
            return torch.zeros(self.state_dim)

        keys   = self._keys[:n]
        values = self._values[:n]

        sim = keys @ x_proj / (x_proj.norm() + 1e-8)

        actual_k = min(k, n)
        topk_idx = sim.topk(actual_k).indices
        topk_sim = sim[topk_idx]
        weights  = torch.softmax(self.beta_hop * topk_sim, dim=0)

        q_init = (weights.unsqueeze(1) * values[topk_idx]).sum(0)
        return q_init.detach()

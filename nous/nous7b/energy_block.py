"""
Single energy block: SwiGLU FFN that contributes one scalar to V(q).

V_l(q) = scalar_proj( RMSNorm(q) · SwiGLU(RMSNorm(q)) )

Stacking 48 of these gives NOUS-7B's deep energy function without
sacrificing the scalar-valued guarantee needed for ODE force computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class EnergyBlock(nn.Module):
    """
    One SwiGLU energy block. Contributes scalar V_l(q) to the total potential.

    SwiGLU: out = (W_gate(x) · σ(W_gate(x))) ⊙ W_up(x), then W_down
    Final projection to R^1 gives the scalar energy contribution.

    Parameters per block at 7B config (state_dim=4096, hidden=11008):
      gate_proj:   4096 × 11008 = 45,088,768
      up_proj:     4096 × 11008 = 45,088,768
      down_proj:  11008 × 4096  = 45,088,768
      scalar_proj:  4096 × 1    =      4,096
      norm:              4096   =      4,096
      ─────────────────────────────────────
      Total:                    ≈ 135,274,496
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.norm = RMSNorm(state_dim)
        self.gate_proj = nn.Linear(state_dim, hidden, bias=False)
        self.up_proj   = nn.Linear(state_dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, state_dim, bias=False)
        self.scalar_proj = nn.Linear(state_dim, 1, bias=True)

        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight,   gain=0.1)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.zeros_(self.scalar_proj.weight)
        nn.init.zeros_(self.scalar_proj.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (..., state_dim) → scalar energy contribution (...,)"""
        h = self.norm(q)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        return self.scalar_proj(h).squeeze(-1)

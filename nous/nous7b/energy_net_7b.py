"""
NOUS-7B energy function.

E(x, q; θ) = V(q; θ) − x^T W_in q

V(q) = ½‖q‖²                                    ← global bowl (no params)
      + Σ_{l=1}^{n_blocks} V_l(q; θ_l)          ← 48 SwiGLU blocks
      − Σ_k amp_k · exp(−‖q−μ_k‖²/σ_k²)         ← RBF basins (morphogenesis)

Force:  f(x,q) = −∂E/∂q  (computed by autograd; never backprop through ODE)
Update: ∂E/∂θ at two fixed points (EqProp)

Parameter count (7B config):
  48 blocks × 135,274,496   = 6,493,175,808
  W_in: embed_dim×state_dim =    16,777,216
  Embedding: vocab×embed    =   131,072,000
  RBF mu: n_rbf×state_dim   =    33,554,432
  RBF amp/sigma: 2×n_rbf    =        16,384
  ─────────────────────────────────────────
  Total                     ≈ 6,674,595,840  (~6.67B)
"""
import torch
import torch.nn as nn
from nous.nous7b.config import NOUSConfig
from nous.nous7b.energy_block import EnergyBlock


class NOUSEnergyNet7B(nn.Module):

    def __init__(self, cfg: NOUSConfig):
        super().__init__()
        self.cfg = cfg
        self.state_dim = cfg.state_dim

        # Token embedding (shared with decoder via tied weights)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Input coupling: −x^T W_in q
        self.W_in = nn.Linear(cfg.embed_dim, cfg.state_dim, bias=False)
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)

        # Deep potential: 48 SwiGLU energy blocks
        self.blocks = nn.ModuleList([
            EnergyBlock(cfg.state_dim, cfg.ffn_hidden)
            for _ in range(cfg.n_energy_blocks)
        ])

        # RBF basin carving (morphogenesis expands these)
        self.mu       = nn.Parameter(torch.randn(cfg.n_rbf, cfg.state_dim) * cfg.rbf_init_scale)
        self.log_amp  = nn.Parameter(torch.ones(cfg.n_rbf) * 0.3)
        self.log_sigma = nn.Parameter(torch.zeros(cfg.n_rbf))

        # Output norm for decoder logits
        self.out_norm = nn.LayerNorm(cfg.state_dim)

    # ── Energy components ───────────────────────────────────────────────────

    def V_bowl(self, q: torch.Tensor) -> torch.Tensor:
        return 0.5 * q.pow(2).sum(-1)

    def V_blocks(self, q: torch.Tensor) -> torch.Tensor:
        return sum(block(q) for block in self.blocks)

    def V_rbf(self, q: torch.Tensor) -> torch.Tensor:
        diff = q.unsqueeze(-2) - self.mu              # (..., n_rbf, d)
        sq   = diff.pow(2).sum(-1)                    # (..., n_rbf)
        sig2 = self.log_sigma.exp().pow(2) + 1e-4
        amp  = self.log_amp.exp()
        return -(amp * (-sq / sig2).exp()).sum(-1)

    def V(self, q: torch.Tensor) -> torch.Tensor:
        return self.V_bowl(q) + self.V_blocks(q) + self.V_rbf(q)

    def coupling(self, x_embed: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return -(self.W_in(x_embed) * q).sum(-1)

    def forward(self, x_embed: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Total energy E(x,q). x_embed is the embedded token vector."""
        return self.V(q) + self.coupling(x_embed, q)

    # ── Force ───────────────────────────────────────────────────────────────

    def force(self, x_embed: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """−∂E/∂q, computed by autograd. Never called during backward through ODE."""
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E = self.forward(x_embed, q_g)
            dE = torch.autograd.grad(E.sum(), q_g)[0]
        return -dE.detach()

    # ── Morphogenesis ────────────────────────────────────────────────────────

    def stochastic_min_curvature(self, x_embed: torch.Tensor, q: torch.Tensor,
                                  n_probes: int = None) -> torch.Tensor:
        n_probes = n_probes or self.cfg.n_curvature_probes
        min_r = torch.tensor(float('inf'), device=q.device)
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E   = self.forward(x_embed, q_g)
            grad = torch.autograd.grad(E.sum(), q_g, create_graph=True)[0]
            for _ in range(n_probes):
                v  = torch.randn_like(q_g)
                v  = v / (v.norm() + 1e-8)
                Hv = torch.autograd.grad((grad * v.detach()).sum(), q_g,
                                          retain_graph=True)[0]
                r  = (v.detach() * Hv.detach()).sum()
                if r < min_r:
                    min_r = r
        return min_r

    def add_rbf_center(self, q: torch.Tensor):
        """Morphogenesis: append a new RBF center at current equilibrium."""
        with torch.no_grad():
            new_mu    = q.detach().unsqueeze(0).to(self.mu.dtype)
            new_amp   = torch.zeros(1, device=self.mu.device)
            new_sigma = torch.zeros(1, device=self.mu.device)
            self.mu        = nn.Parameter(torch.cat([self.mu,        new_mu],    dim=0))
            self.log_amp   = nn.Parameter(torch.cat([self.log_amp,   new_amp],   dim=0))
            self.log_sigma = nn.Parameter(torch.cat([self.log_sigma, new_sigma], dim=0))

    # ── EqProp gradient ─────────────────────────────────────────────────────

    def param_grad_at(self, x_embed: torch.Tensor, q_star: torch.Tensor) -> dict:
        """∂E(x, q*; θ)/∂θ — two-point EqProp update, never through ODE."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()
        E = self.forward(x_embed, q_star.detach())
        E.sum().backward()
        return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in self.named_parameters()}

    # ── Decoder (tied embedding) ─────────────────────────────────────────────

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        """Logits over vocabulary via tied embedding transpose."""
        return self.out_norm(q) @ self.embedding.weight.T

    # ── Utility ─────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self):
        total = self.num_parameters()
        groups = {
            "embedding":  self.embedding.weight.numel(),
            "W_in":       self.W_in.weight.numel(),
            "blocks":     sum(p.numel() for p in self.blocks.parameters()),
            "rbf":        self.mu.numel() + self.log_amp.numel() + self.log_sigma.numel(),
            "out_norm":   sum(p.numel() for p in self.out_norm.parameters()),
        }
        print(f"NOUS-7B parameter summary:")
        for name, n in groups.items():
            print(f"  {name:12s}: {n/1e9:.3f}B  ({100*n/total:.1f}%)")
        print(f"  {'TOTAL':12s}: {total/1e9:.3f}B")

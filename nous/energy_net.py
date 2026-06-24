import torch
import torch.nn as nn


class EnergyNet(nn.Module):
    """
    Total energy: E(x, q; θ) = V(q; θ) − xᵀ W_in q

    The input x is clamped throughout dynamics (not just initial condition).
    Different inputs create different force fields → different equilibria.

    Equilibrium condition: ∂V/∂q = W_in^T x
    (the potential gradient balances the input force)

    This is the correct EqProp formulation (Scellier & Bengio 2017, Eq. 1).
    """

    def __init__(self, input_dim: int, state_dim: int, hidden: int = 128,
                 depth: int = 4, n_rbf: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # V(q; θ) = ½‖q‖² + Σ_k (−amp_k)·exp(−‖q−μ_k‖²/σ_k²)
        # RBF potential: guaranteed local basins, no monotone slope problem.
        self.n_rbf = n_rbf
        self.mu = nn.Parameter(torch.randn(self.n_rbf, state_dim) * 1.5)
        self.log_amp = nn.Parameter(torch.ones(self.n_rbf) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(self.n_rbf))

        # Residual MLP for fine-grained landscape shaping
        layers = [nn.Linear(state_dim, hidden), nn.Tanh(),
                  nn.Linear(hidden, hidden), nn.Tanh(),
                  nn.Linear(hidden, 1)]
        self.V_mlp = nn.Sequential(*layers)

        # Input-state coupling: −xᵀ W_in q
        self.W_in = nn.Linear(input_dim, state_dim, bias=False)

        for m in self.V_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.W_in.weight)

    def V(self, q: torch.Tensor) -> torch.Tensor:
        """
        V(q) = ½‖q‖² − Σ_k amp_k·exp(−‖q−μ_k‖²/σ_k²) + MLP(q)
        Quadratic bowl prevents slope. RBF terms carve local basins.
        """
        # Quadratic restoring force
        bowl = 0.5 * (q ** 2).sum(-1)

        # RBF basin terms: shape (..., n_rbf)
        diff = q.unsqueeze(-2) - self.mu          # (..., n_rbf, d)
        sq_dist = (diff ** 2).sum(-1)             # (..., n_rbf)
        sigma2 = torch.exp(self.log_sigma) ** 2 + 1e-4
        amp = torch.exp(self.log_amp)
        rbf = -(amp * torch.exp(-sq_dist / sigma2)).sum(-1)

        # MLP residual for fine shaping
        mlp = self.V_mlp(q).squeeze(-1)

        return bowl + rbf + mlp

    def coupling(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Input coupling energy −xᵀ W_in q"""
        return -(self.W_in(x) * q).sum(-1)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Total energy E(x, q) = V(q) + coupling(x, q)"""
        return self.V(q) + self.coupling(x, q)

    def force(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        −∂E/∂q = −∂V/∂q + W_in^T x
        The net force driving the state toward equilibrium.
        """
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E = self.forward(x, q_g)
            dE_dq = torch.autograd.grad(E.sum(), q_g)[0]
        return -dE_dq.detach()

    def hessian_min_eigenvalue(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """λ_min(∂²E/∂q²) — curvature at equilibrium. Low = flat = no basin.
        Uses exact Hessian for d≤32, stochastic Lanczos for d>32."""
        d = q.shape[-1]
        if d > 32:
            return self.stochastic_min_curvature(x, q)
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E = self.forward(x, q_g)
            grad = torch.autograd.grad(E.sum(), q_g, create_graph=True)[0]
            H = torch.zeros(d, d)
            for i in range(d):
                g2 = torch.autograd.grad(grad[i], q_g, retain_graph=True, allow_unused=True)[0]
                if g2 is not None:
                    H[i] = g2.detach()
        return torch.linalg.eigvalsh(H).min()

    def stochastic_min_curvature(self, x: torch.Tensor, q: torch.Tensor, n_probes: int = 8) -> torch.Tensor:
        """Stochastic minimum Rayleigh quotient via random Hessian-vector products.
        Each probe: r = v^T H v for a random unit vector v. Min over probes ≈ λ_min.
        Cost: O(n_probes) backward passes instead of O(d)."""
        min_r = torch.tensor(float('inf'))
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            E = self.forward(x, q_g)
            grad = torch.autograd.grad(E.sum(), q_g, create_graph=True)[0]
            for _ in range(n_probes):
                v = torch.randn_like(q_g)
                v = v / (v.norm() + 1e-8)
                Hv = torch.autograd.grad((grad * v.detach()).sum(), q_g,
                                         retain_graph=True)[0]
                r = (v.detach() * Hv.detach()).sum()
                if r < min_r:
                    min_r = r
        return min_r

    def param_grad_at(self, x: torch.Tensor, q_star: torch.Tensor) -> dict:
        """∂E(x, q*; θ)/∂θ for all parameters — used in EqProp update."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()
        E = self.forward(x, q_star.detach())
        E.sum().backward()
        return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for n, p in self.named_parameters()}

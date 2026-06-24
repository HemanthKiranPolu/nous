import torch
import torch.nn as nn


class PotentialNet(nn.Module):
    """
    Learned potential energy V(q; θ): ℝ^d → ℝ
    Scalar field whose minima are attractor basins (memories).
    Architecture: residual MLP with spectral normalization for Lipschitz stability.
    """

    def __init__(self, d: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.d = d

        layers = [nn.utils.spectral_norm(nn.Linear(d, hidden)), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.utils.spectral_norm(nn.Linear(hidden, hidden)), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]

        self.net = nn.Sequential(*layers)

        # Initialize: shallow flat landscape (small weights → shallow basins initially)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.3)
                nn.init.normal_(m.bias, std=0.1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (..., d) → scalar energy (...,)"""
        return self.net(q).squeeze(-1)

    def gradient(self, q: torch.Tensor) -> torch.Tensor:
        """∂V/∂q — force field pointing downhill"""
        q = q.detach().requires_grad_(True)
        V = self.forward(q)
        grad = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        return grad

    def hessian_min_eigenvalue(self, q: torch.Tensor) -> torch.Tensor:
        """
        λ_min(∂²V/∂q²) — minimum curvature at q.
        Low value = flat landscape = no basin = morphogenesis trigger.
        """
        q = q.detach().requires_grad_(True)
        V_val = self.forward(q)
        grad = torch.autograd.grad(V_val.sum(), q, create_graph=True)[0]
        d = q.shape[-1]
        H = torch.zeros(d, d, device=q.device)
        for i in range(d):
            g2 = torch.autograd.grad(
                grad[i], q, retain_graph=True, allow_unused=True
            )[0]
            if g2 is not None:
                H[i] = g2.detach()
        eigenvalues = torch.linalg.eigvalsh(H)
        return eigenvalues.min()

import torch
import torch.nn as nn


class EquilibriumProp:
    """
    Two-phase Equilibrium Propagation (Scellier & Bengio 2017).

    Phase 1 (Free):   solve q̇ = -∂E(x,q)/∂q          → q*_free
    Phase 2 (Nudge):  solve q̇ = -∂(E + ε·C)/∂q        → q*_nudge

    Update:  Δθ = −α·(1/ε)·[∂E(q*_nudge;θ)/∂θ − ∂E(q*_free;θ)/∂θ]

    No backward pass through the ODE. Only ∂E/∂θ at two fixed points.
    """

    def __init__(
        self,
        energy_net,
        solver,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        eps: float = 0.5,
        phi_distance: float = 0.3,
        phi_curvature: float = 0.01,
    ):
        self.E = energy_net
        self.solver = solver
        self.decoder = decoder
        self.optimizer = optimizer
        self.eps = eps
        self.phi_distance = phi_distance
        self.phi_curvature = phi_curvature

    def step(self, x: torch.Tensor, target: torch.Tensor,
             q0_override: torch.Tensor = None):
        """
        One EqProp step on example (x, target).
        q0_override: initial state (default zeros). Pass previous q* for stateful processing.
        Returns: (loss, morphogenesis_triggered, q*_free, q*_nudge)
        """
        # -- PHASE 1: Free relaxation (no supervision) --
        q0 = q0_override if q0_override is not None else torch.zeros(self.E.state_dim)
        q_free = self.solver.solve(x, q0)

        pred_free = self.decoder(q_free)
        loss = nn.functional.cross_entropy(pred_free.unsqueeze(0), target.unsqueeze(0))

        grads_free = self.E.param_grad_at(x, q_free)

        # -- PHASE 2: Nudged relaxation (gentle output clamping) --
        def extra_energy(q: torch.Tensor) -> torch.Tensor:
            logits = self.decoder(q)
            return self.eps * nn.functional.cross_entropy(
                logits.unsqueeze(0), target.unsqueeze(0)
            )

        q_nudge = self.solver.solve(x, q0, extra_energy_fn=extra_energy)
        grads_nudge = self.E.param_grad_at(x, q_nudge)

        # -- UPDATE: EqProp gradient for energy parameters --
        self.optimizer.zero_grad()
        for name, param in self.E.named_parameters():
            if param.requires_grad:
                param.grad = (1.0 / self.eps) * (grads_nudge[name] - grads_free[name])

        # -- Decoder update: standard CE at nudged equilibrium --
        pred_nudge = self.decoder(q_nudge)
        ce_nudge = nn.functional.cross_entropy(pred_nudge.unsqueeze(0), target.unsqueeze(0))
        ce_nudge.backward()

        self.optimizer.step()

        # -- DUAL MORPHOGENESIS TRIGGER --
        dist = (q_nudge - q_free).norm().item()
        lambda_min = self.E.hessian_min_eigenvalue(x, q_free).item()
        morpho = (dist > self.phi_distance) and (lambda_min < self.phi_curvature)

        return loss.item(), morpho, q_free.detach(), q_nudge.detach()

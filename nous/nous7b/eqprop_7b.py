"""
EqProp for NOUS-7B: truncated ODE + bf16 + chunked gradient accumulation.

Key differences from small-scale EqProp:
  1. Truncated fixed-point: n_steps=80 (not 200) — approximate equilibrium
     is sufficient when landscape is smooth (DeepEqProp result: K-step truncation
     with K≥1/λ_min converges to exact EqProp gradient)
  2. bf16 dynamics: force computed in bf16, accumulated in fp32 for stability
  3. Block-wise gradient accumulation: param_grad_at computed per-block to
     avoid holding the full 7B gradient tensor in memory
  4. Gradient scaling: EqProp grads scaled by (1/ε) are O(1/ε) larger than
     CE grads — separate scaling for E params vs decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nous.nous7b.config import NOUSConfig
from nous.nous7b.energy_net_7b import NOUSEnergyNet7B


class EulerODE:
    """Euler integrator for overdamped ODE q̇ = −∂E/∂q."""

    def __init__(self, model: NOUSEnergyNet7B, cfg: NOUSConfig):
        self.model = model
        self.dt    = cfg.dt
        self.tol   = cfg.ode_tol

    def solve(self, x_embed: torch.Tensor, q0: torch.Tensor,
              n_steps: int = None, extra_fn=None) -> torch.Tensor:
        """
        Euler integration to approximate fixed point.
        x_embed: embedded token (frozen during ODE)
        extra_fn: optional ε·C(decode(q), target) for nudged phase
        Returns q* detached from compute graph.
        """
        q = q0.detach().clone()
        dt = self.dt
        for _ in range(n_steps or 80):
            force = self.model.force(x_embed, q)
            if extra_fn is not None:
                q_g = q.detach().requires_grad_(True)
                extra_E = extra_fn(q_g)
                extra_f = -torch.autograd.grad(extra_E, q_g)[0].detach()
                force = force + extra_f
            q = q + dt * force
            if force.norm().item() < self.tol:
                break
        return q.detach()


class EqProp7B:
    """
    Two-phase EqProp for NOUS-7B.

    Phase 1: q*_free  = ODE(E,           n_steps_free)
    Phase 2: q*_nudge = ODE(E + ε·CE,   n_steps_nudge)

    Update (E params):   Δθ ∝ (1/ε)[∂E(q*_nudge)/∂θ − ∂E(q*_free)/∂θ]
    Update (tied embed): Δθ ∝ CE(decode(q*_nudge), target)  [standard BP]

    Memory: grads computed one block at a time to fit 7B in 80GB.
    """

    def __init__(self, model: NOUSEnergyNet7B, optimizer: torch.optim.Optimizer,
                 cfg: NOUSConfig):
        self.model = model
        self.opt   = optimizer
        self.cfg   = cfg
        self.ode   = EulerODE(model, cfg)
        self.eps   = cfg.eps
        self.phi_d = cfg.phi_distance
        self.phi_c = cfg.phi_curvature

    @torch.no_grad()
    def _embed(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.model.embedding(token_id).detach()

    def step(self, token_id: torch.Tensor, target_id: torch.Tensor,
             q0: torch.Tensor = None) -> tuple:
        """
        One EqProp step on (token → next_token).

        Returns: (loss, morpho_triggered, q*_free)
        """
        cfg = self.cfg
        x_embed = self._embed(token_id)
        if q0 is None:
            q0 = torch.zeros(cfg.state_dim, device=x_embed.device,
                             dtype=x_embed.dtype)

        # ── Phase 1: free ──────────────────────────────────────────────────
        q_free = self.ode.solve(x_embed, q0, n_steps=cfg.n_steps_free)

        with torch.enable_grad():
            logits_free = self.model.decode(q_free)
            loss = F.cross_entropy(logits_free.unsqueeze(0),
                                   target_id.unsqueeze(0))

        grads_free = self.model.param_grad_at(x_embed, q_free)

        # ── Phase 2: nudged ────────────────────────────────────────────────
        def extra_energy(q):
            return self.eps * F.cross_entropy(
                self.model.decode(q).unsqueeze(0), target_id.unsqueeze(0))

        q_nudge = self.ode.solve(x_embed, q0, n_steps=cfg.n_steps_nudge,
                                  extra_fn=extra_energy)
        grads_nudge = self.model.param_grad_at(x_embed, q_nudge)

        # ── EqProp update ──────────────────────────────────────────────────
        self.opt.zero_grad()

        # Energy params: EqProp contrastive gradient
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in grads_free:
                param.grad = ((1.0 / self.eps) *
                              (grads_nudge[name] - grads_free[name]))

        # Decoder (tied embedding): standard CE on nudged equilibrium
        with torch.enable_grad():
            logits_nudge = self.model.decode(q_nudge.requires_grad_(False))
            ce_nudge = F.cross_entropy(logits_nudge.unsqueeze(0),
                                        target_id.unsqueeze(0))
        ce_nudge.backward()

        # Gradient clipping: separate scales for EqProp vs CE grads
        e_params  = [p for n, p in self.model.named_parameters()
                     if not n.startswith("embedding") and p.grad is not None]
        emb_params = [self.model.embedding.weight]
        nn.utils.clip_grad_norm_(e_params,   max_norm=1.0)
        nn.utils.clip_grad_norm_(emb_params, max_norm=1.0)

        self.opt.step()

        # ── Morphogenesis trigger ──────────────────────────────────────────
        dist      = (q_nudge - q_free).norm().item()
        lambda_min = self.model.stochastic_min_curvature(x_embed, q_free).item()
        morpho    = (dist > self.phi_d) and (lambda_min < self.phi_c)

        return loss.item(), morpho, q_free.detach()

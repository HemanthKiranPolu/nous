"""NOUS model configs — small (125M) for local dev, 7B for cluster training."""
from dataclasses import dataclass


@dataclass
class NOUSConfig:
    # Manifold
    state_dim: int = 4096
    # Energy network
    n_energy_blocks: int = 48
    ffn_hidden: int = 11008       # SwiGLU hidden (≈ 2.7× state_dim like LLaMA)
    n_rbf: int = 8192
    rbf_init_scale: float = 2.0
    # Input
    vocab_size: int = 32000       # LLaMA tokenizer
    embed_dim: int = 4096
    # ODE solver
    dt: float = 0.01
    n_steps_free: int = 80        # truncated fixed-point (full = 200, expensive at 7B)
    n_steps_nudge: int = 80
    ode_tol: float = 1e-3
    # EqProp
    eps: float = 0.1              # smaller eps for large models (gentler nudge)
    phi_distance: float = 0.02
    phi_curvature: float = 0.5    # stochastic Hessian, 512D+
    n_curvature_probes: int = 16
    # Training
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta_0: float = 1.0
    lambda_annealing: float = 5e-5
    beta_max: float = 10.0
    # Precision
    dtype: str = "bfloat16"       # bf16 for 7B
    grad_accum_steps: int = 8


# Validated locally on CPU/single GPU
NOUS_SMALL = NOUSConfig(
    state_dim=512,
    n_energy_blocks=8,
    ffn_hidden=1408,
    n_rbf=512,
    vocab_size=32000,
    embed_dim=512,
    dt=0.02,
    n_steps_free=60,
    n_steps_nudge=60,
    eps=0.3,
    phi_distance=0.05,
    phi_curvature=1.2,
    dtype="float32",
    grad_accum_steps=1,
)

# Cluster target: ~7B params, bf16, 8×A100 80GB
NOUS_7B = NOUSConfig()

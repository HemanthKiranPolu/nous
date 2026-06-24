import math


class AnnealingScheduler:
    """
    Thermodynamic training schedule. Replaces epochs.

    β(t) = β₀ · exp(λ · t)

    High temperature (low β): system explores, shallow basins form.
    Low temperature (high β): only minimum-action trajectories survive.
    Incorrect attractors dissolve via Boltzmann suppression: exp(−β·S_wrong) → 0.

    The learning rate α is coupled to temperature: α(t) = α₀ / β(t)
    so that each update is appropriately scaled for the current thermal regime.
    """

    def __init__(self, beta_0: float = 0.1, lambda_: float = 0.01, beta_max: float = 50.0, alpha_0: float = 1e-3):
        self.beta_0 = beta_0
        self.lambda_ = lambda_
        self.beta_max = beta_max
        self.alpha_0 = alpha_0
        self.step_count = 0

    def beta(self) -> float:
        return min(self.beta_0 * math.exp(self.lambda_ * self.step_count), self.beta_max)

    def alpha(self) -> float:
        """Learning rate coupled inversely to temperature, floored at alpha_0/10."""
        return max(self.alpha_0 / max(self.beta(), 1.0), self.alpha_0 / 10)

    def converged(self, free_energy_delta: float, threshold: float = 1e-5) -> bool:
        """Stop when free energy F = −(1/β)·log Z stops decreasing."""
        return abs(free_energy_delta) < threshold and self.beta() >= self.beta_max * 0.9

    def tick(self):
        self.step_count += 1

    def status(self) -> str:
        return f"step={self.step_count}  β={self.beta():.3f}  α={self.alpha():.6f}  T={1/self.beta():.3f}"

"""SIGReg: Sketched Isotropic Gaussian Regularization.

Self-contained port of the Cramer-Wold characteristic-function matching loss
used in LeJEPA (Balestriero 2025). For each random 1D projection of an
embedding batch, compares the empirical CF to the N(0,1) target exp(-omega^2/2)
and accumulates the squared distance.

For LeWorldModel-JEPA: applied to predicted next-frame latents to drive
their marginal distribution toward isotropic Gaussian, providing the same
EDMD orthonormality condition (G ~ I) the bevscout / koopman_jepa stacks use.
"""
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_projections: int = 1024,
        n_frequencies: int = 17,
        freq_min: float = 0.1,
        freq_max: float = 5.0,
    ):
        super().__init__()
        # axis-aligned + random unit projections
        axis_dirs = torch.eye(latent_dim)
        random_dirs = torch.randn(n_projections, latent_dim)
        random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True)
        self.register_buffer("directions", torch.cat([axis_dirs, random_dirs], dim=0))

        freqs = torch.linspace(freq_min, freq_max, n_frequencies)
        self.register_buffer("freqs", freqs)
        self.register_buffer("target_cos", torch.exp(-0.5 * freqs ** 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., latent_dim) -> scalar SIGReg loss."""
        z = z.reshape(-1, z.shape[-1])
        projections = self.directions @ z.T                 # (P, B)
        tz = projections.unsqueeze(2) * self.freqs.view(1, 1, -1)
        emp_cos = tz.cos().mean(dim=1)                      # (P, F)
        emp_sin = tz.sin().mean(dim=1)
        target_cos = self.target_cos.unsqueeze(0)
        return ((emp_cos - target_cos) ** 2 + emp_sin ** 2).mean()

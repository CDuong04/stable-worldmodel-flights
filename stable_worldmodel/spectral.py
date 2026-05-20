"""Spectral utilities for stable-manifold-constrained MPC.

Given a discrete-time linear map A (the Jacobian of the learned latent
predictor at a touchdown equilibrium), this module decomposes the dynamics
into stable and unstable invariant subspaces and builds a Lyapunov function
on the stable part.

All functions are CPU-only numpy/scipy. Torch helpers convert results to
buffers for use inside StableManifoldConstrainedCost.

Math summary:
  Real Schur:      A = Z @ T @ Z.T          (T quasi-upper-triangular)
  ordschur:        permute so |eigvals| < 1 fill the leading T_ss block
  V_s_basis :=     Z_ord[:, :n_s]           orthonormal basis of E^s
  Sylvester:       T_ss Y - Y T_uu = -T_su  yields V_u_basis = Z_ord[:, n_s:] + V_s_basis @ Y
                                            (orthogonal Z_ord cols don't span the invariant
                                             unstable subspace; the Sylvester adjustment
                                             produces a basis of E^u that A leaves invariant)
  Spectral proj.:  V = [V_s_basis | V_u_basis], V_inv = inv(V)
                   P_s = V_s_basis @ V_inv[:n_s, :], P_u = V_u_basis @ V_inv[n_s:, :]
  Lyapunov:        solve T_ss^T P_basis T_ss - P_basis = -Q_basis    (discrete Lyap on E^s)
                   P_lyap = V_inv[:n_s, :].T @ P_basis @ V_inv[:n_s, :]
                                            (lifts to ambient R^d; PSD; kernel = E^u)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch


@dataclass
class SpectralDecomposition:
    """Result of decompose_stable_unstable.

    All numpy arrays are float64 (computed in real arithmetic).
    Shapes use d := state dim, n_s := stable dim, n_u := unstable dim.
    """
    A: np.ndarray                # (d, d) original matrix
    eigvals: np.ndarray          # (d,) complex eigenvalues
    n_s: int
    n_u: int
    P_s: np.ndarray              # (d, d) oblique projector onto E^s along E^u
    P_u: np.ndarray              # (d, d) oblique projector onto E^u along E^s
    V_s_basis: np.ndarray        # (d, n_s) orthonormal basis of E^s
    V_u_basis: np.ndarray        # (d, n_u) basis of E^u (not orthonormal in general)
    A_s_basis: np.ndarray        # (n_s, n_s) restriction of A to E^s in V_s_basis coords
    V_inv_top: np.ndarray        # (n_s, d) rows of inv([V_s|V_u]) projecting v -> a in V_s coords
    gap: float                   # hyperbolicity gap = min |λ_unstable| - max |λ_stable|
    hyperbolic: bool
    spectral_radius_stable: float  # max_{|λ|<1} |λ|  -- bounds Lyapunov decay rate

    def as_torch(self, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """Buffer-friendly tensors for use inside torch cost models."""
        def t(arr):
            return torch.as_tensor(np.ascontiguousarray(arr), device=device, dtype=dtype)
        return {
            "P_s": t(self.P_s),
            "P_u": t(self.P_u),
            "V_s_basis": t(self.V_s_basis),
            "V_u_basis": t(self.V_u_basis),
            "A_s_basis": t(self.A_s_basis),
            "V_inv_top": t(self.V_inv_top),
        }


def decompose_stable_unstable(
    A: np.ndarray,
    gap_tol: float = 0.05,
) -> SpectralDecomposition:
    """Split discrete-time A into stable (|λ|<1) and unstable (|λ|≥1) parts.

    Args:
        A: (d, d) real matrix. Discrete-time Jacobian of f_θ at equilibrium.
        gap_tol: minimum hyperbolicity gap to flag the decomposition as hyperbolic.
            If gap < gap_tol, the center manifold theorem is needed (handled outside).

    Returns:
        SpectralDecomposition with all projectors and bases.

    Raises:
        ValueError: if A is not square or not finite.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains non-finite entries")
    n = A.shape[0]

    eigvals = np.linalg.eigvals(A)
    abs_eigvals = np.abs(eigvals)
    stable_mask = abs_eigvals < 1.0

    # Hyperbolicity gap
    n_stable_eig = int(stable_mask.sum())
    n_unstable_eig = n - n_stable_eig
    if n_stable_eig > 0 and n_unstable_eig > 0:
        gap = float(np.min(abs_eigvals[~stable_mask]) - np.max(abs_eigvals[stable_mask]))
    elif n_unstable_eig == 0:
        gap = float(1.0 - np.max(abs_eigvals))
    else:
        gap = float(np.min(abs_eigvals) - 1.0)
    hyperbolic = gap > gap_tol

    # Real Schur with reorder.  scipy.linalg.schur(..., sort='iuc') places
    # eigenvalues with |λ| ≤ 1 in the leading block.  Conjugate pairs share
    # magnitude so they're never split.
    #
    # 'iuc' uses |λ| ≤ 1 (closed disk).  Strictness vs. our |λ| < 1 stable
    # criterion only matters on the unit circle, which gap_tol catches as
    # non-hyperbolic separately.
    T_ord, Z_ord, sdim = scipy.linalg.schur(A, output="real", sort="iuc")
    n_s = int(sdim)
    n_u = n - n_s

    V_s_basis = np.ascontiguousarray(Z_ord[:, :n_s])
    V_u_naive = np.ascontiguousarray(Z_ord[:, n_s:])
    T_ss = np.ascontiguousarray(T_ord[:n_s, :n_s])
    T_su = np.ascontiguousarray(T_ord[:n_s, n_s:]) if n_u > 0 else np.zeros((n_s, 0))
    T_uu = np.ascontiguousarray(T_ord[n_s:, n_s:]) if n_u > 0 else np.zeros((0, 0))

    # Sylvester adjustment so V_u_basis spans the A-invariant unstable subspace E^u.
    # We seek Y solving T_ss Y - Y T_uu = -T_su; then U_u = V_u_naive + V_s_basis @ Y
    # satisfies A U_u = U_u T_uu (invariant).
    if n_s > 0 and n_u > 0:
        Y = scipy.linalg.solve_sylvester(T_ss, -T_uu, -T_su)
        V_u_basis = V_u_naive + V_s_basis @ Y
    else:
        V_u_basis = V_u_naive

    # Spectral projectors.  V = [V_s | V_u] is a basis of R^d; V_inv splits any v
    # into its E^s and E^u components.
    if n_s > 0 and n_u > 0:
        V_full = np.concatenate([V_s_basis, V_u_basis], axis=1)
        V_inv = np.linalg.inv(V_full)
        V_inv_top = np.ascontiguousarray(V_inv[:n_s, :])
        V_inv_bot = np.ascontiguousarray(V_inv[n_s:, :])
        P_s = V_s_basis @ V_inv_top
        P_u = V_u_basis @ V_inv_bot
    elif n_s == n:
        V_inv_top = V_s_basis.T  # orthonormal
        P_s = np.eye(n)
        P_u = np.zeros((n, n))
    else:
        V_inv_top = np.zeros((0, n))
        P_s = np.zeros((n, n))
        P_u = np.eye(n)

    return SpectralDecomposition(
        A=A,
        eigvals=eigvals,
        n_s=n_s,
        n_u=n_u,
        P_s=P_s.astype(np.float64),
        P_u=P_u.astype(np.float64),
        V_s_basis=V_s_basis.astype(np.float64),
        V_u_basis=V_u_basis.astype(np.float64),
        A_s_basis=T_ss.astype(np.float64),
        V_inv_top=V_inv_top.astype(np.float64),
        gap=gap,
        hyperbolic=hyperbolic,
        spectral_radius_stable=(
            float(np.max(abs_eigvals[stable_mask])) if n_s > 0 else 0.0
        ),
    )


def solve_discrete_lyapunov_on_stable_subspace(
    decomp: SpectralDecomposition,
    Q: np.ndarray | None = None,
) -> np.ndarray:
    """Build P_lyap ∈ R^{d×d} so V_L(v) = v^T P_lyap v decays on E^s under A.

    Solves T_ss^T P_basis T_ss - P_basis = -Q_basis in the orthonormal basis of E^s,
    then lifts so V_L(v) = (V_inv_top @ v)^T P_basis (V_inv_top @ v).

    Properties of the lifted P_lyap:
      * Symmetric PSD on R^d.
      * Kernel ⊇ E^u  (only the E^s component of v contributes to V_L).
      * Strict descent rate: V_L(A v) ≤ ρ(A_s)^2 · V_L(v) for v ∈ E^s.

    Args:
        decomp: result of decompose_stable_unstable.
        Q: optional positive-definite weighting matrix.
            - If None, use Q_basis = I_{n_s}.
            - If shape (d, d), restrict to E^s via V_s_basis^T Q V_s_basis.
            - If shape (n_s, n_s), use directly as Q_basis.

    Returns:
        P_lyap: (d, d) numpy array.
    """
    n = decomp.A.shape[0]
    n_s = decomp.n_s
    if n_s == 0:
        return np.zeros((n, n), dtype=np.float64)

    if Q is None:
        Q_basis = np.eye(n_s, dtype=np.float64)
    else:
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape == (n, n):
            Q_basis = decomp.V_s_basis.T @ Q @ decomp.V_s_basis
        elif Q.shape == (n_s, n_s):
            Q_basis = Q
        else:
            raise ValueError(
                f"Q shape {Q.shape} must be ({n}, {n}) or ({n_s}, {n_s})"
            )

    # scipy.linalg.solve_discrete_lyapunov(A, Q) solves  A X A^H - X + Q = 0
    # i.e.,                                              A X A^H - X = -Q
    # We want T_ss^T P_basis T_ss - P_basis = -Q_basis, so set A_arg = T_ss^T:
    P_basis = scipy.linalg.solve_discrete_lyapunov(decomp.A_s_basis.T, Q_basis)
    P_basis = (P_basis + P_basis.T) / 2  # numerical symmetrization

    V_inv_top = decomp.V_inv_top              # (n_s, d)
    P_lyap = V_inv_top.T @ P_basis @ V_inv_top
    P_lyap = (P_lyap + P_lyap.T) / 2
    return P_lyap.astype(np.float64)


def project_to_stable_subspace(
    v: torch.Tensor,
    P_s: torch.Tensor,
    P_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch of perturbations v ∈ R^d into (P^s v, P^u v).

    Args:
        v: (..., d) batch of perturbations.
        P_s, P_u: (d, d) projectors from SpectralDecomposition.as_torch().

    Returns:
        (v_s, v_u): each of shape (..., d), with v = v_s + v_u.
    """
    v_s = v @ P_s.T
    v_u = v @ P_u.T
    return v_s, v_u


def unstable_energy(v: torch.Tensor, P_u: torch.Tensor) -> torch.Tensor:
    """‖P^u v‖² for each row of v — used as constraint C1.

    Args:
        v: (..., d)
        P_u: (d, d)

    Returns:
        (...,) tensor of squared norms.
    """
    v_u = v @ P_u.T
    return (v_u * v_u).sum(dim=-1)


def lyapunov_value(v: torch.Tensor, P_lyap: torch.Tensor) -> torch.Tensor:
    """V_L(v) = v^T P_lyap v  — quadratic Lyapunov function for constraint C2.

    Args:
        v: (..., d)
        P_lyap: (d, d) symmetric PSD

    Returns:
        (...,) tensor.
    """
    return (v @ P_lyap * v).sum(dim=-1)

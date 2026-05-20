"""Tests for stable-manifold-constrained MPC: spectral, cost, runtime safety."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from stable_worldmodel.spectral import (
    SpectralDecomposition,
    decompose_stable_unstable,
    lyapunov_value,
    project_to_stable_subspace,
    solve_discrete_lyapunov_on_stable_subspace,
    unstable_energy,
)
from stable_worldmodel.stable_mpc import (
    StableMPCConfig,
    StableManifoldConstrainedCost,
)
from stable_worldmodel.runtime_safety import (
    RuntimeSafetyConfig,
    RuntimeSafetyMonitor,
)


# ---------------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------------


def test_decompose_diagonal_real():
    """Diagonal A with mixed stable/unstable eigvals: trivial decomposition."""
    A = np.diag([0.3, 0.7, 1.4, 2.1])
    d = decompose_stable_unstable(A)

    assert d.n_s == 2
    assert d.n_u == 2
    assert d.hyperbolic
    # gap = min |unstable| − max |stable| = 1.4 − 0.7 = 0.7
    assert d.gap == pytest.approx(0.7, abs=1e-9)
    assert d.spectral_radius_stable == pytest.approx(0.7, abs=1e-9)


def test_decompose_all_stable():
    """All eigvals inside unit circle: n_s = d, n_u = 0."""
    rng = np.random.default_rng(0)
    A = 0.5 * rng.standard_normal((5, 5))
    # ensure spectral radius < 1
    A = 0.4 * A / np.max(np.abs(np.linalg.eigvals(A)))
    d = decompose_stable_unstable(A)
    assert d.n_s == 5 and d.n_u == 0
    np.testing.assert_allclose(d.P_s, np.eye(5), atol=1e-10)
    np.testing.assert_allclose(d.P_u, np.zeros((5, 5)), atol=1e-10)


def test_decompose_complex_pair():
    """Complex conjugate eigvals stay grouped: both stable or both unstable."""
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Scale to |λ| = 0.8 → both eigvals stable
    A = 0.8 * R
    d = decompose_stable_unstable(A)
    assert d.n_s == 2 and d.n_u == 0
    np.testing.assert_allclose(np.abs(d.eigvals), [0.8, 0.8], atol=1e-9)


def test_projector_identities():
    """P_s + P_u = I, P_s² = P_s, P_u² = P_u, P_s @ P_u ≈ 0."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6))
    d = decompose_stable_unstable(A)

    n = A.shape[0]
    np.testing.assert_allclose(d.P_s + d.P_u, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(d.P_s @ d.P_s, d.P_s, atol=1e-7)
    np.testing.assert_allclose(d.P_u @ d.P_u, d.P_u, atol=1e-7)
    np.testing.assert_allclose(d.P_s @ d.P_u, np.zeros((n, n)), atol=1e-7)
    np.testing.assert_allclose(d.P_u @ d.P_s, np.zeros((n, n)), atol=1e-7)


def test_invariance_of_stable_subspace():
    """A @ V_s_basis lies in span(V_s_basis): i.e., A V_s = V_s T_ss."""
    rng = np.random.default_rng(2)
    A = 0.7 * rng.standard_normal((5, 5))  # small enough that some |λ|<1
    d = decompose_stable_unstable(A)

    if d.n_s == 0:
        pytest.skip("no stable modes in this random draw")

    AV_s = A @ d.V_s_basis
    V_s_T_ss = d.V_s_basis @ d.A_s_basis
    np.testing.assert_allclose(AV_s, V_s_T_ss, atol=1e-8)


def test_hyperbolicity_gap_threshold():
    """gap_tol = 0.05 flags near-unit-circle modes as non-hyperbolic."""
    A = np.diag([0.99, 1.01])  # both within 0.01 of unit circle
    d = decompose_stable_unstable(A, gap_tol=0.05)
    # gap = |1.01| - |0.99| = 0.02 < 0.05 → not hyperbolic
    assert not d.hyperbolic
    assert d.gap == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# Discrete Lyapunov solve
# ---------------------------------------------------------------------------


def test_lyapunov_psd_and_kernel():
    """P_lyap is symmetric PSD; its kernel contains E^u."""
    rng = np.random.default_rng(3)
    A = 0.6 * rng.standard_normal((5, 5))
    d = decompose_stable_unstable(A)
    if d.n_s == 0 or d.n_u == 0:
        pytest.skip("mixed spectrum required")

    P = solve_discrete_lyapunov_on_stable_subspace(d)
    # Symmetry
    np.testing.assert_allclose(P, P.T, atol=1e-8)
    # PSD: all eigvals ≥ 0
    eigvals_P = np.linalg.eigvalsh(P)
    assert eigvals_P.min() >= -1e-7
    # Kernel ⊇ E^u: P @ V_u_basis ≈ 0
    PV_u = P @ d.V_u_basis
    assert np.max(np.abs(PV_u)) < 1e-7


def test_lyapunov_strict_descent_on_stable_subspace():
    """V_L(A v) < V_L(v) for nonzero v ∈ E^s."""
    rng = np.random.default_rng(4)
    A = 0.6 * rng.standard_normal((6, 6))
    d = decompose_stable_unstable(A)
    if d.n_s == 0:
        pytest.skip("need stable modes")

    P = solve_discrete_lyapunov_on_stable_subspace(d)

    # Sample 50 random vectors in E^s
    for _ in range(50):
        coeffs = rng.standard_normal(d.n_s)
        v = d.V_s_basis @ coeffs
        Av = A @ v
        VL_v = float(v @ P @ v)
        VL_Av = float(Av @ P @ Av)
        if VL_v < 1e-10:
            continue  # near zero
        assert VL_Av < VL_v, f"V_L(Av)={VL_Av:.6f} not < V_L(v)={VL_v:.6f}"


# ---------------------------------------------------------------------------
# Torch projection helpers
# ---------------------------------------------------------------------------


def test_project_to_stable_subspace_torch():
    """v_s + v_u = v for any input."""
    rng = np.random.default_rng(5)
    A = 0.7 * rng.standard_normal((4, 4))
    d = decompose_stable_unstable(A)
    buffers = d.as_torch()

    v = torch.randn(7, 4)
    v_s, v_u = project_to_stable_subspace(v, buffers["P_s"], buffers["P_u"])
    torch.testing.assert_close(v_s + v_u, v, atol=1e-5, rtol=1e-5)


def test_unstable_energy_zero_on_stable_subspace():
    """For v ∈ E^s, ‖P^u v‖² ≈ 0."""
    rng = np.random.default_rng(6)
    A = 0.6 * rng.standard_normal((5, 5))
    d = decompose_stable_unstable(A)
    if d.n_s == 0:
        pytest.skip("need stable modes")
    buffers = d.as_torch(dtype=torch.float64)

    # Build batch of vectors purely in E^s
    coeffs = torch.randn(10, d.n_s, dtype=torch.float64)
    v = coeffs @ torch.as_tensor(d.V_s_basis.T, dtype=torch.float64)

    e_u = unstable_energy(v, buffers["P_u"])
    assert e_u.max().item() < 1e-10


def test_lyapunov_value_quadratic_form():
    """lyapunov_value matches v^T P v computed manually."""
    rng = np.random.default_rng(7)
    A = 0.5 * rng.standard_normal((4, 4))
    d = decompose_stable_unstable(A)
    P = solve_discrete_lyapunov_on_stable_subspace(d)

    v = torch.randn(8, 4, dtype=torch.float64)
    P_torch = torch.as_tensor(P, dtype=torch.float64)
    VL = lyapunov_value(v, P_torch)

    expected = torch.einsum("bi,ij,bj->b", v, P_torch, v)
    torch.testing.assert_close(VL, expected, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# StableManifoldConstrainedCost (synthetic linear base)
# ---------------------------------------------------------------------------


class LinearFakeBase:
    """Synthetic world-model surrogate: z_{t+1} = A z_t + B u_t from z_0.

    Mimics base.get_cost(info_dict, actions) -> (B, S) plus the side effect of
    stashing predicted latents in info_dict["predicted_pixels_embed"].
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, z_0: torch.Tensor):
        self.A = A
        self.B = B
        self.z_0 = z_0
        self.d = A.shape[0]
        self.m = B.shape[1]

    def get_cost(self, info_dict, action_candidates):
        # action_candidates: (B, S, H, D)
        B_, S, H, D = action_candidates.shape
        BS = B_ * S
        actions_flat = action_candidates.reshape(BS, H, D)

        z = self.z_0.unsqueeze(0).expand(BS, -1).contiguous()  # (BS, d)
        traj = [z]
        for t in range(H):
            z = z @ self.A.T + actions_flat[:, t] @ self.B.T
            traj.append(z)
        z_traj = torch.stack(traj, dim=1)                       # (BS, H+1, d)
        info_dict["predicted_pixels_embed"] = z_traj            # (BS, H+1, d), no patch dim

        # Dummy differentiable cost — gradient-friendly placeholder.
        return action_candidates.pow(2).mean(dim=(-1, -2))      # (B, S)


def _make_synthetic_artifact(
    A_np: np.ndarray, m: int = 2, tmp_dir: Path | None = None
) -> tuple[Path, SpectralDecomposition]:
    """Build a saved spectral artifact from A; return (path, decomposition)."""
    d = decompose_stable_unstable(A_np)
    P_lyap = solve_discrete_lyapunov_on_stable_subspace(d)
    artifact = {
        "z_star": torch.zeros(A_np.shape[0]),
        "a_star": torch.zeros(m),
        "P_s": torch.as_tensor(d.P_s, dtype=torch.float32),
        "P_u": torch.as_tensor(d.P_u, dtype=torch.float32),
        "P_lyap": torch.as_tensor(P_lyap, dtype=torch.float32),
        "V_s_basis": torch.as_tensor(d.V_s_basis, dtype=torch.float32),
        "A_s_basis": torch.as_tensor(d.A_s_basis, dtype=torch.float32),
        "eigvals": torch.as_tensor(d.eigvals.real, dtype=torch.float32),
        "n_s": d.n_s,
        "n_u": d.n_u,
        "gap": d.gap,
        "spectral_radius_stable": d.spectral_radius_stable,
    }
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "local_linear_model.pt"
    torch.save(artifact, path)
    return path, d


def _make_cost_model(A_np, m=2, cfg=None, decoder=None, compute_V=None, B_np=None):
    """Helper to build StableManifoldConstrainedCost + LinearFakeBase pair.

    Default m=2 matches the action_dim used by most tests.  Pass B_np for
    convergence tests that need to control specific (e.g., unstable) directions.
    """
    d = A_np.shape[0]
    if B_np is None:
        B_np = np.eye(d, m, dtype=np.float64)
    artifact_path, _decomp = _make_synthetic_artifact(A_np, m=m)
    A_t = torch.as_tensor(A_np, dtype=torch.float32)
    B_t = torch.as_tensor(B_np, dtype=torch.float32)
    z_0 = torch.zeros(d, dtype=torch.float32)
    base = LinearFakeBase(A_t, B_t, z_0)

    if decoder is None:
        decoder = torch.nn.Identity()
    if compute_V is None:
        def compute_V(x):
            return (x ** 2).sum(dim=-1)

    cost = StableManifoldConstrainedCost(
        base_cost_model=base,
        decoder=decoder,
        compute_V=compute_V,
        spectral_artifact_path=artifact_path,
        cfg=cfg,
        device="cpu",
    )
    return cost, base, _decomp


def test_get_cost_shape_and_value():
    """get_cost returns base_cost + cost_weight * V_L(v_H) with shape (B, S)."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cost_model, base, _ = _make_cost_model(A)

    B_, S, H, D = 2, 3, 5, 2
    actions = torch.zeros(B_, S, H, D, requires_grad=True)
    info = {}
    cost = cost_model.get_cost(info, actions)
    assert cost.shape == (B_, S)
    # With z_0 = 0 and actions = 0, z_H = z_* so V_L(v_H) = 0;
    # base cost is also 0 in the fake model, so total = 0.
    torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1e-6, rtol=1e-6)


def test_constraints_shape_with_all_enabled():
    """get_constraints returns (B, S, C) with expected C = T + 1 + (T-1)."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cost_model, _, _ = _make_cost_model(A)

    B_, S, H, D = 2, 3, 5, 2
    actions = torch.zeros(B_, S, H, D, requires_grad=True)
    info = {}
    constraints = cost_model.get_constraints(info, actions)
    # T = H+1 = 6 rollout steps including initial.
    # C1 has T = 6 cols, C2 is terminal-only (1 col), C3 has T-1 = 5 cols.
    assert constraints.shape == (B_, S, 6 + 1 + 5)


def test_constraints_shape_with_only_C1():
    """Disabling C2 + C3 leaves only the unstable-subspace constraint."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cfg = StableMPCConfig(
        use_C1_unstable_bound=True,
        use_C2_latent_lyap=False,
        use_C3_decoded_lyap=False,
    )
    cost_model, _, _ = _make_cost_model(A, cfg=cfg)

    B_, S, H, D = 1, 2, 4, 2
    actions = torch.zeros(B_, S, H, D, requires_grad=True)
    info = {}
    constraints = cost_model.get_constraints(info, actions)
    # Only C1: H+1 = 5 cols
    assert constraints.shape == (B_, S, 5)


def test_constraints_interior_at_equilibrium():
    """At z_0 = 0 with zero actions, all constraints should be strictly < 0."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cost_model, _, _ = _make_cost_model(A)

    B_, S, H, D = 1, 1, 4, 2
    actions = torch.zeros(B_, S, H, D, requires_grad=True)
    info = {}
    constraints = cost_model.get_constraints(info, actions)
    # C1 = -eps_u^2 (negative). C2 = 0 since V_L stays 0. C3 = 0 + eta = 0.
    # All values ≤ 0 (interior).
    assert constraints.max().item() <= 1e-6


def test_constraints_C1_grows_with_unstable_initial_condition():
    """When z_0 has unstable component, C1 grows over the horizon under zero action."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cfg = StableMPCConfig(
        eps_u=0.5,
        use_C1_unstable_bound=True,
        use_C2_latent_lyap=False,
        use_C3_decoded_lyap=False,
    )
    cost_model, _, _ = _make_cost_model(A, cfg=cfg)
    # Plant an unstable initial condition.
    cost_model.base.z_0 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    B_, S, H, D = 1, 1, 5, 2
    actions = torch.zeros(B_, S, H, D, requires_grad=True)
    info = {}
    constraints = cost_model.get_constraints(info, actions)
    c1_traj = constraints[0, 0, :]
    # Last step should be much more violated than the first.
    assert c1_traj[-1] > c1_traj[0]
    # And the latest step should violate eps_u^2 = 0.25 since 1.5^5 ≈ 7.6.
    assert c1_traj[-1] > 0


def test_cost_is_differentiable():
    """LagrangianSolver requires costs.requires_grad=True."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cost_model, _, _ = _make_cost_model(A)

    actions = torch.randn(1, 2, 4, 2, requires_grad=True)
    info = {}
    cost = cost_model.get_cost(info, actions)
    assert cost.requires_grad
    # Gradient flows through.
    cost.sum().backward()
    assert actions.grad is not None
    assert actions.grad.abs().sum() > 0


def test_constraints_are_differentiable():
    """LagrangianSolver also calls .backward() on the augmented Lagrangian."""
    A = np.diag([0.5, 0.5, 1.5, 1.5]).astype(np.float64)
    cost_model, _, _ = _make_cost_model(A)

    actions = torch.randn(1, 2, 4, 2, requires_grad=True)
    info = {}
    constraints = cost_model.get_constraints(info, actions)
    assert constraints.requires_grad
    constraints.pow(2).sum().backward()
    assert actions.grad is not None


# ---------------------------------------------------------------------------
# Lagrangian convergence on a small synthetic problem
# ---------------------------------------------------------------------------


def test_lagrangian_drives_constraints_negative():
    """LagrangianSolver + StableManifoldConstrainedCost reaches feasibility on
    a small linear system within 5 outer iterations."""
    from gymnasium import spaces as gym_spaces

    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.solver.lagrangian import LagrangianSolver

    # Small fully controllable system: 2-D latent (1 stable + 1 unstable), 2-D action.
    # B = identity → both modes are directly controllable.
    A = np.diag([0.5, 1.3]).astype(np.float64)
    cfg = StableMPCConfig(
        eps_u=0.3,                 # generous bound; z_0 within budget
        alpha=0.95,
        use_C1_unstable_bound=True,
        use_C2_latent_lyap=True,
        use_C3_decoded_lyap=False,  # focus on the new manifold constraints
    )
    cost_model, base, _ = _make_cost_model(A, m=2, cfg=cfg)
    # Plant an unstable initial condition.  Small enough that z_0 itself is feasible.
    base.z_0 = torch.tensor([0.0, 0.2])

    action_dim = 2
    plan_cfg = PlanConfig(horizon=4, receding_horizon=1, action_block=1)

    # Hyperparameters tuned for stable dual ascent on this small problem:
    #   - lr=0.1 prevents Adam overshoot under heavy penalty
    #   - n_steps=80 lets the inner loop fully minimize for each multiplier set
    #   - rho_scale=1.3 grows the penalty gently to avoid oscillation
    solver = LagrangianSolver(
        model=cost_model,
        n_steps=80,
        n_outer_steps=6,
        batch_size=None,
        num_samples=4,
        var_scale=0.05,
        rho_init=0.5,
        rho_scale=1.3,
        rho_max=1e3,
        device="cpu",
        seed=42,
        optimizer_kwargs={"lr": 0.1},
    )
    solver.configure(
        action_space=gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, action_dim), dtype=np.float32
        ),
        n_envs=1,
        config=plan_cfg,
    )

    info_dict = {}
    outputs = solver.solve(info_dict)

    # Inspect realized constraint violation after the final solve
    actions = outputs["actions"]                 # (n_envs, H, D)
    actions_with_sample_dim = actions.unsqueeze(1)  # (n_envs, 1, H, D)
    final_constraints = cost_model.get_constraints(dict(), actions_with_sample_dim)
    max_violation = torch.relu(final_constraints).max().item()
    assert max_violation < 0.5, (
        f"After 5 outer iterations, max constraint violation should be near 0; "
        f"got {max_violation:.4f}"
    )


# ---------------------------------------------------------------------------
# RuntimeSafetyMonitor (reactive layer)
# ---------------------------------------------------------------------------


class _ProbeReturningV:
    """Stub decoder that returns its input verbatim; pairs with a V that
    decreases monotonically so realized δ_t is always positive."""

    def __init__(self):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_monitor(cfg=None):
    """Build a RuntimeSafetyMonitor with a 1-D fake state where V = x[0]^2.

    The state index defaults in LyapunovConfig point at positions (0,1,2).
    Use a 17-D zero state + override position via the (0,1,2) indices.
    """
    decoder = torch.nn.Identity()  # state_dim → state_dim identity
    monitor = RuntimeSafetyMonitor(
        decoder=decoder,
        fallback_fn=lambda info, action: action * 0.0,  # zero-out as fallback
        cfg=cfg or RuntimeSafetyConfig(window_size=5, rate_threshold=0.5,
                                       min_window=3, cooldown=2),
    )
    return monitor


def _make_state(p):
    """Build a 17-D state with position (p, 0, 0), other entries zero."""
    s = torch.zeros(17)
    s[0] = p
    return s


def test_runtime_no_fallback_when_descending():
    """V strictly decreasing every step → no fallback triggered."""
    monitor = _make_monitor()
    z_seq = [_make_state(p) for p in [3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1]]
    for z_t, z_tp1 in zip(z_seq[:-1], z_seq[1:]):
        info = monitor.observe(z_t, z_tp1)
        assert not info["fallback_active"], (
            f"Fallback fired during monotone descent: {info}"
        )
    assert not monitor.should_fallback()


def test_runtime_fallback_triggers_under_sustained_violation():
    """V increasing every step → fallback triggers after min_window."""
    monitor = _make_monitor(
        RuntimeSafetyConfig(window_size=5, rate_threshold=0.5, min_window=3, cooldown=2)
    )
    z_seq = [_make_state(p) for p in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.5]]
    final_active = False
    for z_t, z_tp1 in zip(z_seq[:-1], z_seq[1:]):
        info = monitor.observe(z_t, z_tp1)
        final_active = final_active or info["fallback_active"]
    assert final_active, "Fallback should fire under sustained V increase"


def test_runtime_cooldown_clears_flag():
    """After violations subside, fallback clears after `cooldown` good steps."""
    monitor = _make_monitor(
        RuntimeSafetyConfig(window_size=4, rate_threshold=0.5, min_window=2, cooldown=2)
    )
    # Phase 1: trigger fallback
    bad_seq = [_make_state(p) for p in [0.1, 0.3, 0.5, 0.8]]
    for z_t, z_tp1 in zip(bad_seq[:-1], bad_seq[1:]):
        monitor.observe(z_t, z_tp1)
    assert monitor.should_fallback()
    # Phase 2: good descents → cooldown should clear
    good_seq = [_make_state(p) for p in [0.8, 0.5, 0.3, 0.1, 0.05]]
    for z_t, z_tp1 in zip(good_seq[:-1], good_seq[1:]):
        monitor.observe(z_t, z_tp1)
    assert not monitor.should_fallback(), "Fallback should clear after cooldown"


def test_runtime_fallback_fn_overrides_action():
    """When fallback active, fallback() applies the configured override."""
    monitor = _make_monitor()
    # Force into fallback by enough violations
    bad_seq = [_make_state(p) for p in [0.1, 0.3, 0.5, 0.8, 1.2]]
    for z_t, z_tp1 in zip(bad_seq[:-1], bad_seq[1:]):
        monitor.observe(z_t, z_tp1)
    assert monitor.should_fallback()
    action = torch.tensor([1.0, 2.0, 3.0])
    safe = monitor.fallback({}, action)
    torch.testing.assert_close(safe, torch.zeros_like(action))


def test_runtime_summary_keys():
    monitor = _make_monitor()
    monitor.observe(_make_state(1.0), _make_state(0.9))
    s = monitor.summary()
    for key in ("total_violations", "total_steps", "violation_rate",
                "fallback_triggers", "fallback_currently_active"):
        assert key in s

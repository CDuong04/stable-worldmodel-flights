"""Ranking-correlation benchmark: does the predicted safety signal rank
candidate actions in the same order as what actually happens in the simulator?

The handover documented ranking correlation Spearman rho=0.056 for the original
decoded-V monitor: the planner's "safe" actions were essentially uncorrelated
with what produced realized descent.  This script measures the same quantity
for three candidate signals and writes a comparison JSON:

  1. base_cost        -- DINOWM pixel-MSE goal cost (the standard CEM signal)
  2. decoded_V_descent -- the original monitor: V(d_phi(z_1)) - V(d_phi(z_0))
  3. V_L_terminal     -- v_H^T P_lyap v_H  (NEW probe-free Lyapunov signal)
  4. C3_constraint    -- decoded-V descent as a hard constraint penalty
  5. V_L_change       -- V_L(v_1) - V_L(v_0)  (one-step latent Lyapunov change)

For each of N calibration states, we:
  - Sample K candidate actions (Gaussian around a baseline action)
  - Score each with all 5 signals
  - Save the env state, execute each action one step from the saved state,
    record the realized V change V(x_1) - V(x_0)
  - Compute Spearman rho per state between (predicted ranking, realized ranking)
  - Report distribution of rho across the N states

Outputs JSON with per-state rhos and aggregate statistics for paper Figure 5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Pickle-compat injection (mirrors evaluate_rocket_stable_mpc.py L46-59).
_train_dir = str(REPO_ROOT / "scripts" / "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)
try:
    import importlib as _importlib
    _lejepa_mod = _importlib.import_module("lejepa_wm")
    for _name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
        if hasattr(_lejepa_mod, _name):
            setattr(sys.modules["__main__"], _name, getattr(_lejepa_mod, _name))
except Exception as _e:
    print(f"[warn] could not preload ObsProbe symbol: {_e}")

from scipy.stats import spearmanr

from evaluate_rocket_clf import (
    ProbeStateAdapter,
    _extract_probe_from_wm,
    _first_image,
    _packed_action_history,
    _to_chw_time_batch,
    _to_time_batch,
    compute_proprio_stats,
    encode_pixels,
    load_decoder,
    reset_options_for_level,
)

import stable_worldmodel as swm
from stable_worldmodel.clf_cost import make_rocket_V
from stable_worldmodel.policy import AutoCostModel
from stable_worldmodel.solver.action_adapter import (
    DatasetColumnNormalizer,
    RocketActionNormalizer,
)
from stable_worldmodel.stable_mpc import StableMPCConfig, StableManifoldConstrainedCost


def _score_candidates(
    base_model,
    cost_model: StableManifoldConstrainedCost,
    info_dict: dict,
    action_candidates: torch.Tensor,
    V_fn,
    decoder,
    horizon: int,
):
    """Score K candidate actions with all 5 signals.

    z_traj from base.get_cost has shape (K, T, d) where T = history + horizon.
    History frames are the SAME across candidates (deterministic from current
    observation), so signals computed from t=0 or t=1 see zero variance.  We
    therefore index from the END of the trajectory: the last `horizon` entries
    are the predicted future, and z_traj[:, -horizon] is the first future step.

    Returns dict: {signal_name: (K,) numpy array of scores (lower is better)}.
    """
    K = action_candidates.shape[1]
    info_local = dict(info_dict)
    base_cost = base_model.get_cost(info_local, action_candidates)  # (1, K)
    predicted = info_local["predicted_pixels_embed"]                # (K, T, P, d)
    z_traj = predicted.mean(dim=-2) if predicted.dim() == 4 else predicted  # (K, T, d)

    z_star = cost_model.z_star
    P_lyap = cost_model.P_lyap

    # Index from the END: first predicted future step and last future step.
    # If horizon <= T-1 we have history + future; else fall back to first/last.
    T_total = z_traj.shape[1]
    h = min(horizon, T_total - 1) if T_total > 1 else 0
    v_first_pred = z_traj[:, -h, :] - z_star if h > 0 else z_traj[:, 0, :] - z_star
    v_last_pred  = z_traj[:, -1, :] - z_star

    VL_first = (v_first_pred @ P_lyap * v_first_pred).sum(dim=-1)        # (K,)
    VL_last  = (v_last_pred  @ P_lyap * v_last_pred ).sum(dim=-1)        # (K,)

    # Decoded V at first and last predicted latents.
    x_first_hat = decoder(z_traj[:, -h, :] if h > 0 else z_traj[:, 0, :])
    x_last_hat  = decoder(z_traj[:, -1, :])
    V_x_first = V_fn(x_first_hat).reshape(-1)                            # (K,)
    V_x_last  = V_fn(x_last_hat).reshape(-1)

    return {
        "base_cost":         base_cost.reshape(-1).detach().cpu().numpy(),
        "decoded_V_descent": (V_x_last - V_x_first).detach().cpu().numpy(),
        "V_L_terminal":      VL_last.detach().cpu().numpy(),
        "V_L_change":        (VL_last - VL_first).detach().cpu().numpy(),
        "C3_value":          (V_x_last - V_x_first).detach().cpu().numpy(),
    }


def _realized_descent(world, V_fn_state, info_after_step):
    """Compute V(x_0) - V(x_1) from observed proprio/state after one env step."""
    # The eval harness stores proprio under "proprio" or "observation".
    proprio = info_after_step.get("proprio", info_after_step.get("observation"))
    if proprio is None:
        return None
    x = np.asarray(proprio).reshape(-1)
    return float(V_fn_state(x))


def _make_state_value_fn(pos_idx=(14, 15, 16), vel_idx=(3, 4, 5), omega_idx=(10, 11, 12),
                          alpha=0.5, beta=0.1, pad=(0., 0., 0.)):
    """V(x) = ||p - p_pad||^2 + alpha*||v||^2 + beta*||omega||^2 on raw proprio.

    PyFlyt proprio layout: [pos(3), vel(3), quat_wxyz(4), ang_vel(3), fuel(1), target_rel(3)].
    pos_idx points at target_rel for direct pad-relative position.
    """
    pad_arr = np.asarray(pad, dtype=np.float32)
    p_list, v_list, w_list = list(pos_idx), list(vel_idx), list(omega_idx)
    def V(x):
        p = x[p_list].astype(np.float32)
        v = x[v_list].astype(np.float32)
        w = x[w_list].astype(np.float32)
        return float(np.sum((p - pad_arr) ** 2) + alpha * np.sum(v * v) + beta * np.sum(w * w))
    return V


def main() -> int:
    ap = argparse.ArgumentParser(description="Ranking correlation between predicted and realized descent")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spectral-artifact", required=True)
    ap.add_argument("--norm-dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--decoder-arch", default="mlp_2")
    ap.add_argument("--probe-output-space", default="normalized")
    ap.add_argument("--num-states", type=int, default=100)
    ap.add_argument("--num-candidates", type=int, default=32)
    ap.add_argument("--level", default="easy")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidate-sigma", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[ranking] device={device}, num_states={args.num_states}, num_candidates={args.num_candidates}")

    # ---- Build model ----
    base_model = AutoCostModel(args.checkpoint).to(device).eval()
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size

    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, args.decoder_arch, device)
    if decoder is None:
        raise RuntimeError("Checkpoint has no .obs_probe")
    p_mean, p_std = compute_proprio_stats(args.norm_dataset)
    decoder = ProbeStateAdapter(
        decoder, mean=p_mean, std=p_std, output_space=args.probe_output_space,
    ).to(device).eval()

    V_fn = make_rocket_V(
        alpha=0.5, beta=0.1, pad_pos=(0.0, 0.0, 0.0), pos_idx=(14, 15, 16),
    )
    V_fn_state = _make_state_value_fn()

    cost_cfg = StableMPCConfig(
        eps_u=1.0, alpha=0.99, eta=0.0,
        cost_weight=0.01,
        use_C1_unstable_bound=True,
        use_C2_latent_lyap=True,
        use_C3_decoded_lyap=True,
    )
    cost_model = StableManifoldConstrainedCost(
        base_cost_model=base_model,
        decoder=decoder,
        compute_V=V_fn,
        spectral_artifact_path=Path(args.spectral_artifact),
        cfg=cost_cfg,
        device=device,
    )

    # ---- World setup ----
    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=args.history_size, frame_skip=args.frame_skip,
        max_episode_steps=1000, render_mode="rgb_array",
    )
    action_normalizer = RocketActionNormalizer.from_dataset(
        args.norm_dataset, action_block=args.frame_skip,
    )
    action_low = world.envs.action_space.low
    action_high = world.envs.action_space.high
    primitive_action_dim = int(np.prod(world.envs.action_space.shape[1:]))
    # The world model uses action_block (frame_skip) packed actions.
    action_dim_packed = primitive_action_dim * args.frame_skip

    rng = np.random.default_rng(args.seed)
    per_state_rhos: dict[str, list] = {
        sig: [] for sig in ["base_cost", "decoded_V_descent", "V_L_terminal", "V_L_change", "C3_value"]
    }
    n_candidates_collected = 0
    n_failed_states = 0
    state_records = []

    t0 = time.time()
    state_idx = 0
    while state_idx < args.num_states:
        # Reset to a fresh episode and step a random number of times to get
        # a calibration state mid-flight.
        seed = args.seed + state_idx
        obs, info = world.envs.reset(seed=seed, options=reset_options_for_level(args.level))
        warmup_steps = int(rng.integers(5, 40))
        for _ in range(warmup_steps):
            a = rng.uniform(action_low, action_high)
            obs, reward, terminated, truncated, info = world.envs.step(a)
            if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
               (hasattr(truncated, "__len__") and bool(truncated[0])):
                break
        if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
           (hasattr(truncated, "__len__") and bool(truncated[0])):
            n_failed_states += 1
            continue

        # Save env state so we can roll back per candidate.  PyFlyt rocket
        # env is built on PyBullet — use saveState/restoreState directly.
        try:
            import pybullet as pb
            unwrapped = world.envs.envs[0].unwrapped
            client = unwrapped.env._client
            saved_state = pb.saveState(physicsClientId=client)
        except Exception as _e:
            saved_state = None
            client = None

        # Build info_dict for the cost model.  Helpers return numpy; convert to tensors.
        def _to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            if torch.is_tensor(x):
                return x.float().to(device)
            return torch.tensor(x).float().to(device)

        px_now = _to_t(_to_chw_time_batch(info["pixels"]))
        gl_now = _to_t(_to_chw_time_batch(info["goal"]))
        action_hist = _to_t(_packed_action_history(
            [], history_size=args.history_size,
            action_block=args.frame_skip,
            primitive_dim=primitive_action_dim, batch_size=1,
        ))
        proprio_arr = None
        if "proprio" in info:
            proprio_arr = info["proprio"]
        elif "observation" in info:
            proprio_arr = info["observation"]
        if proprio_arr is not None:
            proprio_arr_t = _to_t(_to_time_batch(proprio_arr, batch_size=1))
        else:
            proprio_arr_t = None
        # Sample K candidate actions: zero-centered Gaussian in NORMALIZED space.
        K = args.num_candidates
        # Action shape expected by world model: (B=1, K, H, action_block * primitive_dim).
        cand = torch.randn(
            1, K, args.horizon, action_dim_packed, device=device,
        ) * args.candidate_sigma
        cand.requires_grad_(False)

        # Expand info_dict tensors to (B=1, K, ...) so that DINOWM's get_cost
        # flatten-on-entry path takes them on the (B, N) branch (dinowm.py L231).
        def _expand_to_K(t):
            if t is None:
                return None
            return t.unsqueeze(1).expand(t.shape[0], K, *t.shape[1:]).contiguous()

        info_dict = {
            "pixels": _expand_to_K(px_now),
            "goal":   _expand_to_K(gl_now),
            "action": _expand_to_K(action_hist),
        }
        if proprio_arr_t is not None:
            info_dict["proprio"] = _expand_to_K(proprio_arr_t)

        # Score with all signals (no grad needed).
        with torch.no_grad():
            scores = _score_candidates(
                base_model, cost_model, info_dict, cand, V_fn, decoder,
                horizon=args.horizon,
            )

        # Execute each candidate from the saved state and record realized V change.
        V_before = V_fn_state(np.asarray(proprio_arr).reshape(-1)) if proprio_arr is not None else 0.0
        realized_delta = np.zeros(K, dtype=np.float64)
        for k in range(K):
            # Restore env state via PyBullet.
            if saved_state is not None and client is not None:
                try:
                    pb.restoreState(stateId=saved_state, physicsClientId=client)
                except Exception:
                    pass
            # First primitive action from candidate.
            # cand[0, k, 0] is the first horizon step, packed (action_block, primitive_dim).
            raw = cand[0, k, 0].detach().cpu().numpy().reshape(args.frame_skip, primitive_action_dim)
            primitive_action = raw[0]  # first sub-step in the action block
            primitive_action = np.clip(primitive_action, action_low.reshape(-1), action_high.reshape(-1))
            try:
                obs_k, _, _, _, info_k = world.envs.step(primitive_action.reshape(1, -1))
            except Exception as e:
                realized_delta[k] = 0.0
                continue
            proprio_after = info_k.get("proprio", info_k.get("observation"))
            if proprio_after is None:
                realized_delta[k] = 0.0
                continue
            V_after = V_fn_state(np.asarray(proprio_after).reshape(-1))
            realized_delta[k] = V_after - V_before  # positive = V increased = bad

        # Free the saved PyBullet state.
        if saved_state is not None and client is not None:
            try:
                pb.removeState(stateUniqueId=saved_state, physicsClientId=client)
            except Exception:
                pass

        # Per-state Spearman rho between predicted score and realized delta.
        # Higher predicted score should correspond to higher realized delta.
        for sig, scores_arr in scores.items():
            if len(np.unique(scores_arr)) < 2 or len(np.unique(realized_delta)) < 2:
                continue
            rho, _ = spearmanr(scores_arr, realized_delta)
            if np.isfinite(rho):
                per_state_rhos[sig].append(float(rho))

        state_records.append({
            "state_idx": state_idx,
            "warmup_steps": warmup_steps,
            "V_before": V_before,
            "realized_delta_mean": float(realized_delta.mean()),
            "realized_delta_std": float(realized_delta.std()),
        })

        state_idx += 1
        n_candidates_collected += K
        if state_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"[ranking] {state_idx}/{args.num_states} states done "
                  f"({n_candidates_collected} candidates, {elapsed:.1f}s)")

    # ---- Aggregate ----
    summary = {
        "n_states": args.num_states,
        "n_candidates_per_state": args.num_candidates,
        "level": args.level,
        "candidate_sigma": args.candidate_sigma,
        "failed_states": n_failed_states,
        "wall_time_s": time.time() - t0,
        "per_state_rhos": per_state_rhos,
        "state_records": state_records,
        "aggregate": {},
    }
    for sig, rhos in per_state_rhos.items():
        if not rhos:
            summary["aggregate"][sig] = {"n": 0}
            continue
        arr = np.asarray(rhos, dtype=np.float64)
        summary["aggregate"][sig] = {
            "n": len(rhos),
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "sign_accuracy": float(np.mean(arr > 0)),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n========== Ranking-correlation summary ==========")
    print(f"  signal                  mean rho   median   sign_acc")
    print("  " + "-" * 55)
    for sig, agg in summary["aggregate"].items():
        if "mean" not in agg:
            continue
        print(f"  {sig:<22}  {agg['mean']:>8.3f}   {agg['median']:>6.3f}   {agg['sign_accuracy']:>7.2f}")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

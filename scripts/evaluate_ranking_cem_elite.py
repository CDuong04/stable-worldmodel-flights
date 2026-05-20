"""Closed-loop CEM-elite ranking correlation.

The handover's documented monitor ρ=0.056 was measured on closed-loop CEM
elite candidates, NOT random Gaussian.  Our first random-candidate ranking
script gave V_L_terminal ρ ≈ 0.016 because random candidates produce model
predictions that bear little relationship to realistic future states.

This script samples candidates from CEM's elite distribution at each
calibration state, then ranks them with all five signals against realized
descent.  Expected: V_L_terminal ρ ≫ 0.056 (the right comparison).

Compute: ~15 min on GPU.

Outputs:
  results/ranking_cem_elite.json  - per-state Spearman rhos + aggregates
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

try:
    import importlib as _importlib
    _lejepa_mod = _importlib.import_module("lejepa_wm")
    for _name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
        if hasattr(_lejepa_mod, _name):
            setattr(sys.modules["__main__"], _name, getattr(_lejepa_mod, _name))
except Exception as e:
    print(f"[warn] {e}")

from scipy.stats import spearmanr

from evaluate_rocket_clf import (
    ProbeStateAdapter, _extract_probe_from_wm, _first_image,
    _packed_action_history, _to_chw_time_batch, _to_time_batch,
    compute_proprio_stats, encode_pixels, load_decoder, reset_options_for_level,
)

import stable_worldmodel as swm
from stable_worldmodel.clf_cost import make_rocket_V
from stable_worldmodel.policy import AutoCostModel
from stable_worldmodel.stable_mpc import StableMPCConfig, StableManifoldConstrainedCost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rocket_lejepa_wm_union_vhead_rolloutprobe")
    ap.add_argument("--spectral-artifact", default="models/local_linear_model.pt")
    ap.add_argument("--norm-dataset", default="data/expert_trajectories_union/rocket_expert_union")
    ap.add_argument("--output", default="results/ranking_cem_elite.json")
    ap.add_argument("--num-states", type=int, default=80)
    ap.add_argument("--num-candidates", type=int, default=32)
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--cem-elite-frac", type=float, default=0.25)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[ranking-elite] device={device}, states={args.num_states}, candidates={args.num_candidates}, cem_iters={args.cem_iters}")

    base_model = AutoCostModel(args.checkpoint).to(device).eval()
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, "mlp_2", device)
    p_mean, p_std = compute_proprio_stats(args.norm_dataset)
    decoder = ProbeStateAdapter(decoder, mean=p_mean, std=p_std, output_space="normalized").to(device).eval()

    V_fn = make_rocket_V(alpha=0.5, beta=0.1, pad_pos=(0., 0., 0.), pos_idx=(14, 15, 16))
    pad_arr = np.zeros(3, dtype=np.float32)
    def V_state(x):
        p = x[[14, 15, 16]].astype(np.float32)
        v = x[[3, 4, 5]].astype(np.float32)
        w = x[[10, 11, 12]].astype(np.float32)
        return float(np.sum((p - pad_arr)**2) + 0.5 * np.sum(v*v) + 0.1 * np.sum(w*w))

    cost_cfg = StableMPCConfig(eps_u=1.0, alpha=0.99, eta=0.0, cost_weight=0.01)
    cost_model = StableManifoldConstrainedCost(
        base_cost_model=base_model, decoder=decoder, compute_V=V_fn,
        spectral_artifact_path=Path(args.spectral_artifact),
        cfg=cost_cfg, device=device,
    )

    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=args.history_size, frame_skip=args.frame_skip,
        max_episode_steps=1000, render_mode="rgb_array",
    )
    primitive_dim = int(np.prod(world.envs.action_space.shape[1:]))
    action_dim_packed = primitive_dim * args.frame_skip
    a_low = world.envs.action_space.low.reshape(-1)
    a_high = world.envs.action_space.high.reshape(-1)

    rng = np.random.default_rng(args.seed)
    per_state_rhos: dict[str, list] = {sig: [] for sig in
        ["base_cost", "decoded_V_descent", "V_L_terminal", "V_L_change", "C3_value"]}
    n_elite = max(8, int(args.num_candidates * args.cem_elite_frac))

    t0 = time.time()
    state_idx = 0
    while state_idx < args.num_states:
        seed = args.seed + state_idx
        obs, info = world.envs.reset(seed=seed, options=reset_options_for_level(args.level))
        warmup = int(rng.integers(5, 40))
        for _ in range(warmup):
            a = rng.uniform(a_low, a_high)
            obs, _, terminated, truncated, info = world.envs.step(a)
            if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
               (hasattr(truncated, "__len__") and bool(truncated[0])):
                break
        if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
           (hasattr(truncated, "__len__") and bool(truncated[0])):
            continue

        # Save env state for per-candidate rollouts.
        try:
            import pybullet as pb
            unwrapped = world.envs.envs[0].unwrapped
            client = unwrapped.env._client
            saved = pb.saveState(physicsClientId=client)
        except Exception:
            saved = None
            continue

        # Build cost model inputs.
        def _to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            return x.float().to(device)
        px = _to_t(_to_chw_time_batch(info["pixels"]))
        gl = _to_t(_to_chw_time_batch(info["goal"]))
        ah = _to_t(_packed_action_history([], history_size=args.history_size,
                                          action_block=args.frame_skip,
                                          primitive_dim=primitive_dim, batch_size=1))
        proprio = info.get("proprio", info.get("observation"))
        pr = _to_t(_to_time_batch(np.asarray(proprio).astype(np.float32), batch_size=1))

        def _expand(t, K):
            return t.unsqueeze(1).expand(t.shape[0], K, *t.shape[1:]).contiguous()

        K = args.num_candidates
        info_dict_base = {
            "pixels": _expand(px, K), "goal": _expand(gl, K),
            "action": _expand(ah, K), "proprio": _expand(pr, K),
        }

        # ----- CEM: iterate refining mean/std around candidates -----
        # Initial Gaussian around zero with sigma=0.5.
        mu = torch.zeros(args.horizon, action_dim_packed, device=device)
        sigma = 0.5 * torch.ones(args.horizon, action_dim_packed, device=device)
        for it in range(args.cem_iters):
            cand = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                K, args.horizon, action_dim_packed, device=device
            )
            cand_4d = cand.unsqueeze(0)  # (1, K, H, D)
            info_local = {k: v.clone() for k, v in info_dict_base.items()}
            with torch.no_grad():
                base_cost = base_model.get_cost(info_local, cand_4d).reshape(-1).cpu().numpy()
            elite_idx = np.argsort(base_cost)[:n_elite]
            elite = cand[elite_idx]  # (n_elite, H, D)
            # Refit mean / std.
            mu = elite.mean(dim=0)
            sigma = elite.std(dim=0).clamp_min(0.05)

        # Final K candidates from refined distribution.
        cand = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
            K, args.horizon, action_dim_packed, device=device
        )
        cand_4d = cand.unsqueeze(0)
        info_local = {k: v.clone() for k, v in info_dict_base.items()}

        # Compute all 5 signals.
        with torch.no_grad():
            base_cost = base_model.get_cost(info_local, cand_4d)
            predicted = info_local["predicted_pixels_embed"]
            z_traj = predicted.mean(dim=-2) if predicted.dim() == 4 else predicted  # (K, T, d)
            T_total = z_traj.shape[1]
            h = min(args.horizon, T_total - 1) if T_total > 1 else 0
            v_first = z_traj[:, -h, :] - cost_model.z_star if h > 0 else z_traj[:, 0, :] - cost_model.z_star
            v_last  = z_traj[:, -1, :] - cost_model.z_star
            VL_first = (v_first @ cost_model.P_lyap * v_first).sum(dim=-1)
            VL_last  = (v_last  @ cost_model.P_lyap * v_last ).sum(dim=-1)
            x_first = decoder(z_traj[:, -h, :] if h > 0 else z_traj[:, 0, :])
            x_last  = decoder(z_traj[:, -1, :])
            V_x_first = V_fn(x_first).reshape(-1)
            V_x_last  = V_fn(x_last).reshape(-1)

        scores = {
            "base_cost":         base_cost.reshape(-1).cpu().numpy(),
            "decoded_V_descent": (V_x_last - V_x_first).cpu().numpy(),
            "V_L_terminal":      VL_last.cpu().numpy(),
            "V_L_change":        (VL_last - VL_first).cpu().numpy(),
            "C3_value":          (V_x_last - V_x_first).cpu().numpy(),
        }

        # Execute each candidate from saved state and record realized V change.
        V_before = V_state(np.asarray(proprio).reshape(-1)[:17])
        realized_delta = np.zeros(K, dtype=np.float64)
        for k in range(K):
            try:
                pb.restoreState(stateId=saved, physicsClientId=client)
            except Exception:
                pass
            raw = cand[k, 0].cpu().numpy().reshape(args.frame_skip, primitive_dim)
            prim = np.clip(raw[0], a_low, a_high)
            try:
                _, _, _, _, info_k = world.envs.step(prim.reshape(1, -1))
            except Exception:
                realized_delta[k] = 0.0
                continue
            p_after = info_k.get("proprio", info_k.get("observation"))
            if p_after is None:
                realized_delta[k] = 0.0
                continue
            V_after = V_state(np.asarray(p_after).reshape(-1)[:17])
            realized_delta[k] = V_after - V_before

        try:
            pb.removeState(stateUniqueId=saved, physicsClientId=client)
        except Exception:
            pass

        # Per-state Spearman ρ.
        for sig, scores_arr in scores.items():
            if len(np.unique(scores_arr)) < 2 or len(np.unique(realized_delta)) < 2:
                continue
            rho, _ = spearmanr(scores_arr, realized_delta)
            if np.isfinite(rho):
                per_state_rhos[sig].append(float(rho))

        state_idx += 1
        if state_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"[ranking-elite] {state_idx}/{args.num_states} states ({elapsed:.1f}s)")

    # Aggregate.
    summary = {
        "n_states": args.num_states,
        "n_candidates_per_state": args.num_candidates,
        "cem_iters": args.cem_iters,
        "level": args.level,
        "wall_time_s": time.time() - t0,
        "per_state_rhos": per_state_rhos,
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))

    print()
    print(f"{'signal':<22}  {'mean ρ':>8}  {'median':>7}  {'sign_acc':>8}")
    print("-" * 55)
    for sig, agg in summary["aggregate"].items():
        if "mean" in agg:
            print(f"{sig:<22}  {agg['mean']:>+8.3f}  {agg['median']:>+7.3f}  {agg['sign_accuracy']:>8.2f}")
    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

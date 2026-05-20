"""End-to-end evaluation of stable-manifold-constrained MPC on rocket landing.

Loads a trained DINOWM checkpoint, the saved local-linear-model artifact
(from scripts/fit_local_linear_model.py), wires StableManifoldConstrainedCost
into the LagrangianSolver, and runs N episodes of swm.World with a
RuntimeSafetyMonitor watching the closed-loop trajectory.

Two-layer story (matches the plan):
  - Search-time:  C1 + C2 + C3 constraints prevent violations before action.
  - Runtime:      RuntimeSafetyMonitor watches realized δ and flags fallback.

Outputs per-episode JSON with:
  - episode return, success, length
  - C1/C2/C3 constraint-satisfaction rates (from solver diagnostics)
  - LyapunovMonitor summary (predicted δ traces vs realized δ traces)
  - whether runtime fallback ever triggered

Usage (always via srun on OSCAR):

  srun --pty --gres=gpu:1 --time=00:30:00 python scripts/evaluate_rocket_stable_mpc.py \\
      --checkpoint /oscar/scratch/aiyer40/rocket_lejepa_wm_union_vhead_rolloutprobe \\
      --spectral-artifact models/local_linear_model.pt \\
      --norm-dataset data/expert_trajectories_union/rocket_expert_union \\
      --episodes 5 --level calm \\
      --output results/stable_mpc_smoke.json

Ablations (toggle individual constraints):
  --ablation c1c2c3    full method (default)
  --ablation c1c2      latent stable manifold only (no decoded V)
  --ablation c3        monitor-as-hard-constraint only (no manifold)
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

# Pickle compatibility: training-time checkpoints reference __main__.ObsProbe.
# Inject those symbols before any torch.load happens (mirrors evaluate_rocket_clf.py L472-486).
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

# Import shared helpers from the existing CLF eval — avoids re-implementing
# the dinowm + probe loading boilerplate.
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
from stable_worldmodel.lyapunov import LyapunovConfig
from stable_worldmodel.policy import AutoCostModel, PlanConfig, WorldModelPolicy
from stable_worldmodel.runtime_safety import (
    RuntimeSafetyConfig,
    RuntimeSafetyMonitor,
)
from stable_worldmodel.solver import LagrangianSolver, CEMLagrangianSolver
from stable_worldmodel.solver.action_adapter import (
    ActionSpaceCostAdapter,
    DatasetColumnNormalizer,
    RocketActionAdapter,
    RocketActionNormalizer,
)
from stable_worldmodel.stable_mpc import (
    StableMPCConfig,
    StableManifoldConstrainedCost,
)


ABLATION_FLAGS = {
    "c1c2c3": dict(use_C1_unstable_bound=True,
                   use_C2_latent_lyap=True,
                   use_C3_decoded_lyap=True),
    "c1c2":   dict(use_C1_unstable_bound=True,
                   use_C2_latent_lyap=True,
                   use_C3_decoded_lyap=False),
    "c3":     dict(use_C1_unstable_bound=False,
                   use_C2_latent_lyap=False,
                   use_C3_decoded_lyap=True),
    "c1":     dict(use_C1_unstable_bound=True,
                   use_C2_latent_lyap=False,
                   use_C3_decoded_lyap=False),
    "c2":     dict(use_C1_unstable_bound=False,
                   use_C2_latent_lyap=True,
                   use_C3_decoded_lyap=False),
}


def build_cost_and_solver(
    *,
    checkpoint: str,
    spectral_artifact_path: Path,
    norm_dataset: str,
    decoder_path: str | None,
    decoder_arch: str,
    probe_output_space: str,
    device: str,
    horizon: int,
    eps_u: float,
    alpha: float,
    eta: float,
    cost_weight: float,
    solver_type: str,
    adaptive_weight: bool,
    innovation_threshold: float,
    innovation_scale: float,
    innovation_ema_beta: float,
    ablation: str,
    lr: float,
    n_inner_steps: int,
    n_outer_steps: int,
    num_samples: int,
    rho_init: float,
    rho_scale: float,
    history_size: int,
    frame_skip: int,
    normalize_actions: bool,
    project_actions: bool,
):
    """Return (cost_model, solver, plan_cfg, process_dict, decoder, V_fn)."""
    print(f"[eval] loading world model: {checkpoint}")
    base_model = AutoCostModel(checkpoint).to(device).eval()
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size

    # Probe (state decoder) — prefer the one baked into the checkpoint.
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, decoder_arch, device)
    if decoder is None:
        if decoder_path is None:
            raise RuntimeError(
                "Checkpoint has no .obs_probe; pass --decoder explicitly."
            )
        decoder = load_decoder(decoder_path, embed_dim=embed_dim,
                               architecture=decoder_arch, device=device)

    if probe_output_space == "normalized":
        p_mean, p_std = compute_proprio_stats(norm_dataset)
        decoder = ProbeStateAdapter(
            decoder, mean=p_mean, std=p_std, output_space=probe_output_space,
        ).to(device).eval()
    elif probe_output_space == "raw":
        decoder = ProbeStateAdapter(decoder, output_space=probe_output_space).to(device).eval()
    else:
        raise ValueError(f"Unsupported probe_output_space: {probe_output_space}")

    V_fn = make_rocket_V(
        alpha=0.5, beta=0.1, pad_pos=(0.0, 0.0, 0.0),
        pos_idx=(14, 15, 16),
    )
    plan_cfg = PlanConfig(
        horizon=horizon,
        receding_horizon=max(horizon // 3, 1),
        history_len=history_size,
        action_block=frame_skip,
        warm_start=True,
    )

    ablation_kwargs = ABLATION_FLAGS[ablation]
    cfg = StableMPCConfig(
        eps_u=eps_u, alpha=alpha, eta=eta,
        cost_weight=cost_weight,
        adaptive_weight=adaptive_weight,
        innovation_threshold=innovation_threshold,
        innovation_scale=innovation_scale,
        innovation_ema_beta=innovation_ema_beta,
        gate_c3_by_innovation=adaptive_weight,
        normalized_descent=True,
        **ablation_kwargs,
    )
    print(f"[eval] ablation={ablation} cfg={cfg}")

    cost_model = StableManifoldConstrainedCost(
        base_cost_model=base_model,
        decoder=decoder,
        compute_V=V_fn,
        spectral_artifact_path=spectral_artifact_path,
        cfg=cfg,
        device=device,
    )

    process: dict = {}
    proprio_normalizer = DatasetColumnNormalizer.from_dataset(
        norm_dataset, column="proprio",
    )
    process["proprio"] = proprio_normalizer
    if normalize_actions:
        action_normalizer = RocketActionNormalizer.from_dataset(
            norm_dataset, action_block=plan_cfg.action_block,
        )
        process["action"] = action_normalizer
        if project_actions:
            action_adapter = RocketActionAdapter(
                normalizer=action_normalizer,
                action_block=plan_cfg.action_block,
            )
            cost_model = ActionSpaceCostAdapter(cost_model, action_adapter)
            print("[eval] action-space adapter wrapped around stable-manifold cost")

    if solver_type == "cem_lag":
        topk = max(2, num_samples // 3)
        solver = CEMLagrangianSolver(
            model=cost_model,
            batch_size=1,
            num_samples=num_samples,
            topk=topk,
            n_inner_iters=n_inner_steps,
            n_outer_iters=n_outer_steps,
            var_scale=0.3,
            rho_init=rho_init,
            rho_scale=rho_scale,
            rho_max=1e4,
            device=device,
            seed=42,
        )
        print(f"[eval] using CEMLagrangianSolver (samples={num_samples}, topk={topk}, "
              f"inner={n_inner_steps}, outer={n_outer_steps})")
    else:
        solver = LagrangianSolver(
            model=cost_model,
            n_steps=n_inner_steps,
            n_outer_steps=n_outer_steps,
            batch_size=None,
            num_samples=num_samples,
            var_scale=0.1,
            rho_init=rho_init,
            rho_scale=rho_scale,
            rho_max=1e4,
            device=device,
            seed=42,
            optimizer_kwargs={"lr": lr},
        )
        print(f"[eval] using LagrangianSolver (gradient-based)")

    return cost_model, solver, plan_cfg, process, decoder, V_fn


def run_episode(
    *,
    world,
    policy,
    cost_model,
    decoder,
    V_fn,
    seed: int,
    level: str,
    history_size: int,
    frame_skip: int,
    max_steps: int,
    device: str,
) -> dict:
    """Run a single episode, returning a summary dict."""
    # Reactive monitor (runtime safety belt).
    safety = RuntimeSafetyMonitor(
        decoder=decoder,
        fallback_fn=None,                                  # flag only; do not override
        cfg=RuntimeSafetyConfig(window_size=10, rate_threshold=0.4, min_window=4),
        lyapunov_cfg=LyapunovConfig(
            alpha=0.5, beta=0.1, pad_pos=(0.0, 0.0, 0.0),
            pos_idx=(14, 15, 16),
        ),
        device=device,
    )

    obs, info = world.envs.reset(seed=seed, options=reset_options_for_level(level))
    z_prev = encode_pixels(cost_model.base if hasattr(cost_model, "base")
                           else cost_model, _first_image(info), device)

    ep_return = 0.0
    executed_actions: list[np.ndarray] = []
    primitive_action_dim = int(np.prod(world.envs.action_space.shape[1:]))

    t = 0
    done = False
    fallback_count = 0
    constraint_violation_max = 0.0
    while not done and t < max_steps:
        px_now = _to_chw_time_batch(info["pixels"])
        gl_now = _to_chw_time_batch(info["goal"])
        action_hist = _packed_action_history(
            executed_actions,
            history_size=history_size,
            action_block=frame_skip,
            primitive_dim=primitive_action_dim,
            batch_size=1,
        )
        proprio_arr = None
        if "proprio" in info:
            proprio_arr = info["proprio"]
        elif "observation" in info:
            proprio_arr = info["observation"]
        if proprio_arr is not None:
            proprio_arr = _to_time_batch(proprio_arr, batch_size=1)
        policy_input = {
            "pixels": px_now, "goal": gl_now, "action": action_hist,
        }
        if proprio_arr is not None:
            policy_input["proprio"] = proprio_arr

        action = policy.get_action(policy_input)
        # LagrangianSolver doesn't auto-project actions through physical bounds
        # the way CEMSolver does.  Clip to env action space here.
        action = np.clip(
            np.asarray(action),
            world.envs.action_space.low,
            world.envs.action_space.high,
        )
        # Capture realized latent transition for safety monitor.
        z_curr = encode_pixels(
            cost_model.base if hasattr(cost_model, "base") else cost_model,
            _first_image(info), device,
        )
        if z_prev is not None and z_curr is not None:
            safety.observe(z_prev, z_curr)
            if safety.should_fallback():
                fallback_count += 1
            # Disturbance proxy: norm of consecutive latent change.  Reaches the
            # underlying StableManifoldConstrainedCost via __getattr__ pass-through
            # on the ActionSpaceCostAdapter wrapper.
            try:
                inner_cost = cost_model.base if hasattr(cost_model, "base") else cost_model
                # Pool patch dim if present.
                z_a = z_prev.mean(dim=-2) if z_prev.dim() >= 3 else z_prev
                z_b = z_curr.mean(dim=-2) if z_curr.dim() >= 3 else z_curr
                delta_norm = float((z_b - z_a).norm().item())
                if hasattr(inner_cost, "update_innovation"):
                    inner_cost.update_innovation(delta_norm)
            except Exception as e:
                if t == 0:
                    print(f"[warn] could not update innovation: {e}")
        z_prev = z_curr

        obs, reward, terminated, truncated, info = world.envs.step(action)
        ep_return += float(reward[0] if hasattr(reward, "__len__") else reward)
        executed_actions.append(np.asarray(action[0]) if action.ndim > 1 else np.asarray(action))
        was_terminated = bool(terminated[0] if hasattr(terminated, "__len__") else terminated)
        was_truncated  = bool(truncated[0] if hasattr(truncated, "__len__") else truncated)
        done = was_terminated or was_truncated
        # Try to detect landing success from info or reward at terminal.
        success_flag = None
        if isinstance(info, dict):
            s = info.get("success", info.get("landed"))
            if s is not None:
                success_flag = bool(s[0]) if hasattr(s, "__len__") else bool(s)
        t += 1

    # Pull the adaptive traces from the inner cost (if available).
    inner_cost = cost_model.base if hasattr(cost_model, "base") else cost_model
    innovation_trace = list(getattr(inner_cost, "innovation_trace", []) or [])
    activation_trace = list(getattr(inner_cost, "activation_trace", []) or [])

    # Heuristic success: early termination (not truncation) usually means landing.
    # For PyFlyt rocket, terminated=True without max_steps reached implies touchdown.
    inferred_success = (t < max_steps) and bool(was_terminated) and not bool(was_truncated)
    summary = {
        "level": level,
        "seed": seed,
        "episode_length": t,
        "episode_return": ep_return,
        "success": (success_flag if success_flag is not None else inferred_success),
        "terminated": bool(was_terminated),
        "truncated": bool(was_truncated),
        "fallback_count": fallback_count,
        "safety": safety.summary(),
        "safety_trace": safety.trace(),
        "innovation_trace": innovation_trace,
        "activation_trace": activation_trace,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stable-manifold-constrained MPC eval (smoke + integration)",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--spectral-artifact", required=True)
    p.add_argument("--norm-dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decoder", default=None)
    p.add_argument("--decoder-arch", default="mlp_2")
    p.add_argument("--probe-output-space", default="normalized",
                   choices=["normalized", "raw"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--level", default="calm")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--n-inner-steps", type=int, default=30)
    p.add_argument("--n-outer-steps", type=int, default=5)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--rho-scale", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--solver", choices=["lagrangian", "cem_lag"], default="lagrangian",
                   help="Inner solver: 'lagrangian' = gradient-based, "
                        "'cem_lag' = hybrid CEM-sampler with augmented Lagrangian outer.")
    p.add_argument("--eps-u", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cost-weight", type=float, default=0.1,
                   help="Weight on latent terminal cost relative to base pixel-MSE cost. "
                        "0.0 = pure CEM baseline; 0.1 = default blend.")
    p.add_argument("--adaptive-weight", action="store_true",
                   help="Gate cost_weight and C3 by sigmoid of running innovation EMA. "
                        "Behaves like CEM when innovation is small; activates the full "
                        "Lyapunov machinery when disturbance grows.")
    p.add_argument("--innovation-threshold", type=float, default=8.0,
                   help="Sigmoid center for adaptive activation.")
    p.add_argument("--innovation-scale", type=float, default=4.0,
                   help="Sigmoid scale for adaptive activation.")
    p.add_argument("--innovation-ema-beta", type=float, default=0.2,
                   help="EMA coefficient for innovation tracking.")
    p.add_argument("--ablation", default="c1c2c3", choices=list(ABLATION_FLAGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normalize-actions", action="store_true")
    p.add_argument("--project-actions", action="store_true")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    cost_model, solver, plan_cfg, process, decoder, V_fn = build_cost_and_solver(
        checkpoint=args.checkpoint,
        spectral_artifact_path=Path(args.spectral_artifact),
        norm_dataset=args.norm_dataset,
        decoder_path=args.decoder,
        decoder_arch=args.decoder_arch,
        probe_output_space=args.probe_output_space,
        device=device,
        horizon=args.horizon,
        eps_u=args.eps_u,
        alpha=args.alpha,
        eta=args.eta,
        cost_weight=args.cost_weight,
        solver_type=args.solver,
        adaptive_weight=args.adaptive_weight,
        innovation_threshold=args.innovation_threshold,
        innovation_scale=args.innovation_scale,
        innovation_ema_beta=args.innovation_ema_beta,
        ablation=args.ablation,
        lr=args.lr,
        n_inner_steps=args.n_inner_steps,
        n_outer_steps=args.n_outer_steps,
        num_samples=args.num_samples,
        rho_init=args.rho_init,
        rho_scale=args.rho_scale,
        history_size=args.history_size,
        frame_skip=args.frame_skip,
        normalize_actions=args.normalize_actions,
        project_actions=args.project_actions,
    )

    print(f"[eval] running {args.episodes} episodes on level={args.level}")
    results = {
        "ablation": args.ablation,
        "horizon": args.horizon,
        "level": args.level,
        "eps_u": args.eps_u,
        "alpha": args.alpha,
        "eta": args.eta,
        "cost_weight": args.cost_weight,
        "lagrangian": {
            "n_inner_steps": args.n_inner_steps,
            "n_outer_steps": args.n_outer_steps,
            "num_samples": args.num_samples,
            "rho_init": args.rho_init,
            "rho_scale": args.rho_scale,
            "lr": args.lr,
        },
        "spectral_artifact": args.spectral_artifact,
        "episodes": [],
    }

    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=args.history_size, frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps, render_mode="rgb_array",
    )
    policy = WorldModelPolicy(
        solver=solver, config=plan_cfg, process=process if process else None,
    )
    world.set_policy(policy)

    for ep in range(args.episodes):
        t0 = time.time()
        ep_summary = run_episode(
            world=world, policy=policy, cost_model=cost_model,
            decoder=decoder, V_fn=V_fn,
            seed=args.seed + ep, level=args.level,
            history_size=args.history_size, frame_skip=args.frame_skip,
            max_steps=args.max_steps, device=device,
        )
        ep_summary["wall_time_s"] = round(time.time() - t0, 2)
        print(
            f"[eval] episode {ep+1}/{args.episodes}: "
            f"len={ep_summary['episode_length']} "
            f"return={ep_summary['episode_return']:.2f} "
            f"viol_rate={ep_summary['safety']['violation_rate']:.3f} "
            f"fallback={ep_summary['safety']['fallback_triggers']} "
            f"({ep_summary['wall_time_s']}s)"
        )
        results["episodes"].append(ep_summary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

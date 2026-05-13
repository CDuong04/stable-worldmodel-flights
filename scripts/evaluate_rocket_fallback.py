"""Evaluate RocketJEPA with a decoded-Lyapunov alarm and expert fallback.

The controller starts with the same world-model MPC/CLF-MPC policy used by
``evaluate_rocket_clf.py``. If the decoded descent monitor observes
``delta_t < -threshold`` for ``k`` consecutive transitions, control switches
to the PDG expert for the rest of that episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401
import stable_worldmodel as swm
from stable_worldmodel.envs.pdg_expert import PDGExpertPolicy
from stable_worldmodel.policy import AutoCostModel, PlanConfig, WorldModelPolicy
from stable_worldmodel.lyapunov import LyapunovConfig, LyapunovMonitor
from stable_worldmodel.clf_cost import CLFAugmentedCost, CLFCostConfig, make_rocket_V

from scripts.evaluate_rocket_clf import (
    ProbeStateAdapter,
    _extract_probe_from_wm,
    _first_image,
    _take_last_frame,
    compute_proprio_stats,
    encode_pixels,
    load_decoder,
    reset_options_for_level,
)


def _preload_obs_probe_symbol() -> None:
    import importlib
    import sys

    train_dir = str(Path(__file__).resolve().parent / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    try:
        mod = importlib.import_module("lejepa_wm")
        for name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
            if hasattr(mod, name):
                setattr(sys.modules["__main__"], name, getattr(mod, name))
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[warn] could not preload ObsProbe symbol: {exc}")


def _to_chw_batch(arr):
    arr = np.asarray(arr)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3:
        return arr.transpose(2, 0, 1)[None]
    if arr.ndim == 4:
        return arr.transpose(0, 3, 1, 2)
    return arr


def _extract_proprio(info):
    proprio_arr = info.get("proprio", info.get("observation"))
    if proprio_arr is None:
        return None
    arr = np.asarray(proprio_arr)
    if arr.ndim == 3:
        arr = arr[:, -1]
    if arr.ndim == 1:
        arr = arr[None]
    return arr


def _policy_input_from_info(info):
    px_now = _to_chw_batch(_take_last_frame(info["pixels"]))
    gl_now = _to_chw_batch(_take_last_frame(info["goal"]))
    policy_input = {
        "pixels": px_now,
        "goal": gl_now,
        "action": info["action"],
    }
    proprio_arr = _extract_proprio(info)
    if proprio_arr is not None:
        policy_input["proprio"] = proprio_arr
        policy_input["goal_proprio"] = proprio_arr
    return policy_input, proprio_arr


def _load_auto_threshold(calibration_dir: Path) -> tuple[float, int]:
    path = calibration_dir / "alarm_calibration.json"
    data = json.loads(path.read_text())
    rec = data["recommended"]
    return float(rec["threshold"]), int(rec["k"])


def build_components(
    model_name: str,
    decoder_path: str | None,
    decoder_arch: str,
    norm_dataset: str,
    probe_output_space: str,
    horizon: int,
    num_samples: int,
    n_steps: int,
    device: str,
    lambda_V: float,
    eta: float,
    alpha: float,
    beta: float,
    clf_normalized_descent: bool,
):
    _preload_obs_probe_symbol()

    print(f"Loading world model: {model_name}")
    base_model = AutoCostModel(model_name)
    try:
        base_model.to(device)
        base_model.eval()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[warn] base_model.to({device}) failed: {exc}")

    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size

    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, decoder_arch, device)
    if decoder is not None:
        print("Using ObsProbe baked into world-model checkpoint")
    else:
        if decoder_path is None:
            raise RuntimeError("No baked obs_probe and no --decoder path supplied")
        print(f"Loading decoder: {decoder_path} ({decoder_arch})")
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
        raise ValueError(f"Unsupported probe output space: {probe_output_space}")

    V_fn = make_rocket_V(alpha=alpha, beta=beta, pad_pos=(0.0, 0.0, 0.0),
                         pos_idx=(14, 15, 16))
    cost_model = CLFAugmentedCost(
        base_cost_model=base_model,
        decoder=decoder,
        compute_V=V_fn,
        cfg=CLFCostConfig(
            lambda_V=lambda_V,
            eta=eta,
            normalized_descent=clf_normalized_descent,
        ),
        device=device,
    )
    solver = swm.solver.CEMSolver(
        model=cost_model,
        num_samples=num_samples,
        var_scale=0.5,
        n_steps=n_steps,
        topk=max(num_samples // 10, 5),
        device=device,
    )
    plan_cfg = PlanConfig(
        horizon=horizon,
        receding_horizon=max(horizon // 3, 1),
        history_len=1,
        action_block=2,
        warm_start=True,
    )
    return base_model, decoder, solver, plan_cfg


def run_fallback_eval(
    model_name: str,
    decoder_path: str | None,
    decoder_arch: str,
    norm_dataset: str,
    probe_output_space: str,
    horizon: int,
    episodes: int,
    seed: int,
    levels: list[str],
    num_samples: int,
    n_steps: int,
    device: str,
    lambda_V: float,
    eta: float,
    alarm_threshold: float,
    alarm_k: int,
    alpha: float,
    beta: float,
    clf_normalized_descent: bool,
):
    base_model, decoder, solver, plan_cfg = build_components(
        model_name=model_name,
        decoder_path=decoder_path,
        decoder_arch=decoder_arch,
        norm_dataset=norm_dataset,
        probe_output_space=probe_output_space,
        horizon=horizon,
        num_samples=num_samples,
        n_steps=n_steps,
        device=device,
        lambda_V=lambda_V,
        eta=eta,
        alpha=alpha,
        beta=beta,
        clf_normalized_descent=clf_normalized_descent,
    )

    results = {
        "mode": "wm_with_pdg_fallback",
        "lambda_V": lambda_V,
        "eta": eta,
        "alarm_threshold": alarm_threshold,
        "alarm_k": alarm_k,
        "horizon": horizon,
        "num_samples": num_samples,
        "n_steps": n_steps,
        "by_level": {},
    }

    for level in levels:
        print(f"\n--- fallback eval: level={level} threshold={alarm_threshold:.6f} k={alarm_k} ---")
        world = swm.World(
            "swm/PFRocketLandingExt-v0",
            num_envs=1,
            image_shape=(224, 224),
            max_episode_steps=1200,
            render_mode="rgb_array",
        )
        wm_policy = WorldModelPolicy(solver=solver, config=plan_cfg)
        wm_policy.set_env(world.envs)
        expert_policy = PDGExpertPolicy()
        expert_policy.set_env(world.envs)

        ep_records = []
        for ep in range(episodes):
            if wm_policy._action_buffer is not None:
                wm_policy._action_buffer.clear()
            wm_policy._next_init = None
            expert_policy.reset()
            mon = LyapunovMonitor(
                decoder=decoder,
                device=device,
                cfg=LyapunovConfig(
                    alpha=alpha,
                    beta=beta,
                    pad_pos=(0.0, 0.0, 0.0),
                    pos_idx=(14, 15, 16),
                ),
            )
            obs, info = world.envs.reset(
                seed=seed + ep,
                options=reset_options_for_level(level),
            )
            z_prev = encode_pixels(base_model, _first_image(info), device)

            done = False
            ep_return = 0.0
            t = 0
            streak = 0
            fallback_active = False
            switchover_step = None

            while not done and t < 1200:
                policy_input, proprio_arr = _policy_input_from_info(info)
                if fallback_active:
                    if proprio_arr is None:
                        raise RuntimeError("PDG fallback requires raw proprio/observation in info")
                    action = expert_policy.get_action(proprio_arr)
                else:
                    action = wm_policy.get_action(policy_input)

                a_low = np.asarray(world.envs.action_space.low)
                a_high = np.asarray(world.envs.action_space.high)
                action = np.clip(np.asarray(action), a_low, a_high)
                obs, reward, terminated, truncated, info = world.envs.step(action)

                z_curr = encode_pixels(base_model, _first_image(info), device)
                step_info = mon.step(z_prev, z_curr)
                z_prev = z_curr

                if not fallback_active:
                    if step_info["delta_t"] < -alarm_threshold:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= alarm_k:
                        fallback_active = True
                        switchover_step = t

                ep_return += float(np.asarray(reward).sum())
                t += 1
                done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())

            summary = mon.summary()
            summary.update(mon.trace())
            summary.update({
                "success": bool(np.asarray(terminated).any()),
                "return": ep_return,
                "T": t,
                "seed": int(seed + ep),
                "fallback_triggered": bool(fallback_active),
                "switchover_step": switchover_step,
            })
            ep_records.append(summary)
            print(
                f"  ep {ep}: success={summary['success']} return={ep_return:.1f} "
                f"viol={summary['violation_rate']:.2f} fallback={fallback_active} "
                f"switch={switchover_step}"
            )

        agg = {
            "success_rate": float(np.mean([r["success"] for r in ep_records])),
            "violation_rate": float(np.mean([r["violation_rate"] for r in ep_records])),
            "mean_descent_rate": float(np.mean([r["mean_descent"] for r in ep_records])),
            "mean_return": float(np.mean([r["return"] for r in ep_records])),
            "mean_steps": float(np.mean([r["T"] for r in ep_records])),
            "fallback_trigger_rate": float(np.mean([r["fallback_triggered"] for r in ep_records])),
            "mean_switchover_step": float(np.mean([
                r["switchover_step"] for r in ep_records if r["switchover_step"] is not None
            ])) if any(r["switchover_step"] is not None for r in ep_records) else float("nan"),
            "n_episodes": len(ep_records),
            "episodes": ep_records,
        }
        results["by_level"][level] = agg
        print(
            f"[{level}] success={100*agg['success_rate']:.1f}% "
            f"viol={agg['violation_rate']:.3f} "
            f"fallback={100*agg['fallback_trigger_rate']:.1f}%"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--decoder-arch", default="mlp_2", choices=["linear", "mlp_2", "mlp_3"])
    parser.add_argument("--norm-dataset", default="data/expert_trajectories_union/rocket_expert_union")
    parser.add_argument("--probe-output-space", default="normalized", choices=["normalized", "raw"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", nargs="+", default=["extreme"])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda-V", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--alarm-threshold", default="auto")
    parser.add_argument("--alarm-k", type=int, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=Path("outputs/rocket_jepa/clf_paired"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--clf-descent-mode", default="normalized", choices=["normalized", "absolute"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.alarm_threshold == "auto":
        threshold, k = _load_auto_threshold(args.calibration_dir)
        if args.alarm_k is not None:
            k = args.alarm_k
    else:
        threshold = float(args.alarm_threshold)
        k = args.alarm_k if args.alarm_k is not None else 3

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = run_fallback_eval(
        model_name=args.model,
        decoder_path=args.decoder,
        decoder_arch=args.decoder_arch,
        norm_dataset=args.norm_dataset,
        probe_output_space=args.probe_output_space,
        horizon=args.horizon,
        episodes=args.episodes,
        seed=args.seed,
        levels=args.levels,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        device=args.device,
        lambda_V=args.lambda_V,
        eta=args.eta,
        alarm_threshold=threshold,
        alarm_k=k,
        alpha=args.alpha,
        beta=args.beta,
        clf_normalized_descent=(args.clf_descent_mode == "normalized"),
    )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()

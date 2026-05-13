"""Collect on-policy RocketJEPA MPC/CLF rollouts as HF training data.

This complements expert data with the action distribution the planner actually
produces. The resulting dataset has the same columns expected by
``StepsDataset``: pixels path, proprio state, primitive action, episode_idx,
step_idx.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from datasets import Dataset
from PIL import Image

from stable_worldmodel.clf_cost import CLFAugmentedCost, CLFCostConfig, make_rocket_V
from stable_worldmodel.policy import AutoCostModel, PlanConfig, WorldModelPolicy
from stable_worldmodel.solver.action_adapter import (
    ActionSpaceCostAdapter,
    DatasetColumnNormalizer,
    RocketActionAdapter,
    RocketActionNormalizer,
)

from evaluate_rocket_clf import (
    ProbeStateAdapter,
    _extract_probe_from_wm,
    _packed_action_history,
    _to_chw_time_batch,
    _to_time_batch,
    compute_proprio_stats,
    reset_options_for_level,
)


def _preload_probe_symbols() -> None:
    train_dir = str(Path(__file__).resolve().parent / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    mod = importlib.import_module("lejepa_wm")
    for name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
        if hasattr(mod, name):
            setattr(sys.modules["__main__"], name, getattr(mod, name))


def _first_image(arr) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _state_from_info(info):
    if "proprio" in info:
        return np.asarray(info["proprio"])
    if "observation" in info:
        return np.asarray(info["observation"])
    raise KeyError("info has no proprio/observation state")


def build_policy(args, lambda_v: float, runtime_shield: bool):
    _preload_probe_symbols()
    base_model = AutoCostModel(args.model)
    base_model.to(args.device).eval()
    backbone = base_model.backbone.backbone if hasattr(base_model.backbone, "backbone") else base_model.backbone
    embed_dim = backbone.config.hidden_size
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, args.decoder_arch, args.device)
    if decoder is None:
        raise RuntimeError("Checkpoint has no baked-in ObsProbe; on-policy collection needs it.")
    if args.probe_output_space == "normalized":
        p_mean, p_std = compute_proprio_stats(args.norm_dataset)
        decoder = ProbeStateAdapter(decoder, mean=p_mean, std=p_std, output_space="normalized").to(args.device).eval()
    else:
        decoder = ProbeStateAdapter(decoder, output_space="raw").to(args.device).eval()

    V_fn = make_rocket_V(alpha=args.alpha, beta=args.beta, pad_pos=(0.0, 0.0, 0.0), pos_idx=(14, 15, 16))
    cost_model = CLFAugmentedCost(
        base_cost_model=base_model,
        decoder=decoder,
        compute_V=V_fn,
        cfg=CLFCostConfig(
            lambda_V=lambda_v,
            eta=args.eta if lambda_v > 0 else 0.0,
            normalized_descent=True,
            hard_filter=False,
            runtime_shield=runtime_shield and lambda_v > 0,
            shield_margin=0.0,
            calibration_epsilon=args.shield_calibration_epsilon,
        ),
        device=args.device,
    )
    plan_cfg = PlanConfig(
        horizon=args.horizon,
        receding_horizon=max(args.horizon // 3, 1),
        history_len=args.history_size,
        action_block=args.frame_skip,
        warm_start=True,
    )
    action_normalizer = RocketActionNormalizer.from_dataset(args.norm_dataset, action_block=plan_cfg.action_block)
    action_adapter = RocketActionAdapter(normalizer=action_normalizer, action_block=plan_cfg.action_block)
    cost_model = ActionSpaceCostAdapter(cost_model, action_adapter)
    process = {
        "proprio": DatasetColumnNormalizer.from_dataset(args.norm_dataset, column="proprio"),
        "action": action_normalizer,
    }
    solver = swm.solver.CEMSolver(
        model=cost_model,
        num_samples=args.num_samples,
        var_scale=0.5,
        n_steps=args.n_steps,
        topk=max(args.num_samples // 10, 5),
        device=args.device,
        diagnostics_enabled=False,
        return_best_candidate=True,
    )
    return WorldModelPolicy(solver=solver, config=plan_cfg, process=process), plan_cfg


def collect(args):
    out_dir = Path(args.out_dir)
    img_root = out_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    rows = []
    ep_idx = 0
    arms = []
    if "std" in args.arms:
        arms.append(("std", 0.0, False))
    if "clf" in args.arms:
        for lv in args.lambda_v:
            arms.append((f"clf_lam{lv:g}", float(lv), False))
    if "shield" in args.arms:
        for lv in args.lambda_v:
            arms.append((f"shield_lam{lv:g}", float(lv), True))

    for arm_name, lv, shield in arms:
        policy, plan_cfg = build_policy(args, lv, shield)
        world = swm.World(
            "swm/PFRocketLandingExt-v0",
            num_envs=1,
            image_shape=(224, 224),
            history_size=args.history_size,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            render_mode="rgb_array",
        )
        world.set_policy(policy)
        for level in args.levels:
            for ep in range(args.episodes):
                obs, info = world.envs.reset(
                    seed=args.seed + ep_idx,
                    options=reset_options_for_level(level),
                )
                executed_actions: list[np.ndarray] = []
                primitive_dim = int(np.prod(world.envs.action_space.shape[1:]))
                done = False
                step = 0
                ep_dir = img_root / f"ep_{ep_idx:05d}_{arm_name}_{level}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                while not done and step < args.max_steps:
                    state = _state_from_info(info)
                    action_hist = _packed_action_history(
                        executed_actions,
                        history_size=args.history_size,
                        action_block=args.frame_skip,
                        primitive_dim=primitive_dim,
                        batch_size=1,
                    )
                    policy_input = {
                        "pixels": _to_chw_time_batch(info["pixels"]),
                        "goal": _to_chw_time_batch(info["goal"]),
                        "action": action_hist,
                        "proprio": _to_time_batch(state, batch_size=1),
                    }
                    action = policy.get_action(policy_input)
                    a_low = np.asarray(world.envs.action_space.low)
                    a_high = np.asarray(world.envs.action_space.high)
                    action = np.clip(np.asarray(action), a_low, a_high)

                    img_path = ep_dir / f"{step:04d}.png"
                    Image.fromarray(_first_image(info["pixels"]).astype(np.uint8)).save(img_path)
                    rows.append({
                        "pixels": str(img_path.resolve()),
                        "proprio": np.asarray(state).reshape(-1).astype(np.float32).tolist(),
                        "action": np.asarray(action).reshape(-1).astype(np.float32).tolist(),
                        "episode_idx": int(ep_idx),
                        "step_idx": int(step),
                        "source": arm_name,
                        "level": level,
                    })

                    obs, reward, terminated, truncated, info = world.envs.step(action)
                    executed_actions.append(np.asarray(action).reshape(-1).copy())
                    done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
                    step += 1
                print(f"[collect] ep={ep_idx} arm={arm_name} level={level} steps={step}", flush=True)
                ep_idx += 1

    ds = Dataset.from_list(rows)
    dataset_path = out_dir / "dataset"
    ds.save_to_disk(str(dataset_path))
    print(f"Saved {len(rows)} rows / {ep_idx} episodes to {dataset_path}")


def main():
    p = argparse.ArgumentParser(description="Collect RocketJEPA on-policy MPC/CLF data")
    p.add_argument("--model", default="/oscar/scratch/aiyer40/rocket_lejepa_wm_union")
    p.add_argument("--norm-dataset", default="data/expert_trajectories_union/rocket_expert_union")
    p.add_argument("--probe-output-space", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--decoder-arch", choices=["linear", "mlp_2", "mlp_3"], default="mlp_2")
    p.add_argument("--out-dir", default="data/expert_trajectories_onpolicy/rocket_onpolicy_cem_clf")
    p.add_argument("--levels", nargs="+", default=["hard", "extreme"])
    p.add_argument("--arms", nargs="+", choices=["std", "clf", "shield"], default=["std", "clf", "shield"])
    p.add_argument("--lambda-v", type=float, nargs="+", default=[5.0, 10.0])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--shield-calibration-epsilon", type=float, default=0.0)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    collect(p.parse_args())


if __name__ == "__main__":
    main()

"""Closed-loop rocket eval with optional CLF-augmented CEM cost (P1 + P2 arms).

This is the rocket_jepa.tex hypothesis-test entry point. It runs the same
LeJEPA-WM closed-loop pipeline as evaluate_rocket_lejepa.py but lets the
caller swap between:

    --lambda-V 0       --> standard CEM (paper hypothesis P1)
    --lambda-V <pos>   --> CLF-augmented CEM (paper hypothesis P2)

The same trained world-model checkpoint and the same state probe d_phi
are used for both arms, so the only thing changing between P1 and P2 is
the cost augmentation. Output JSONs land side by side under
/users/aiyer40/scratch/results_lejepa_v4/clf_eval/<ckpt>__lambda<x>.json
so the downstream paired test can read them straight off disk.

Usage
-----
    # Standard CEM (P1):
    python -m scripts.evaluate_rocket_clf \
        --model lejepa_wm_union --decoder /path/to/probe.pt \
        --lambda-V 0 --episodes 50 --levels easy hard

    # CLF-augmented CEM (P2):
    python -m scripts.evaluate_rocket_clf \
        --model lejepa_wm_union --decoder /path/to/probe.pt \
        --lambda-V 1.0 --eta 0.0 --episodes 50 --levels easy hard
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from datasets import load_from_disk

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401  (registers env)
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel, WorldModelPolicy, PlanConfig
from stable_worldmodel.lyapunov import LyapunovMonitor, LyapunovConfig
from stable_worldmodel.clf_cost import CLFAugmentedCost, CLFCostConfig, make_rocket_V
from stable_worldmodel.solver.action_adapter import (
    ActionSpaceCostAdapter,
    DatasetColumnNormalizer,
    RocketActionAdapter,
    RocketActionNormalizer,
)

# `evaluate_rocket_lejepa.py` imports a private StateDecoder class that's
# not on this script's path; we inline the small helpers it shares so the
# CLF eval works whether or not the legacy state-decoder path is set up.


def compute_proprio_stats(dataset_path: str):
    ds = load_from_disk(dataset_path)
    proprio = np.asarray(ds["proprio"], dtype=np.float32)
    mean = proprio.mean(axis=0)
    std = np.maximum(proprio.std(axis=0), 1e-6)
    return mean, std


class ProbeStateAdapter(torch.nn.Module):
    """Wrap an observation probe so it returns the physical state used by V."""

    def __init__(
        self,
        probe: torch.nn.Module,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        output_space: str = "normalized",
    ):
        super().__init__()
        self.probe = probe
        self.output_space = output_space
        if output_space == "normalized":
            if mean is None or std is None:
                raise ValueError("normalized probe output requires mean/std stats")
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        else:
            self.mean = None
            self.std = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.probe(z)
        if self.output_space == "normalized":
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            return x * std + mean
        return x


def reset_options_for_level(level: str) -> dict:
    """Parse simple and compound disturbance labels into env reset options."""
    if level in ("none", "default", "nominal"):
        return {}
    if level in ("easy", "medium", "hard", "extreme"):
        return {"perturbation_level": level}
    if level.startswith("wind:"):
        return {"wind_level": level.split(":", 1)[1]}
    if level.startswith("kick:"):
        return {"lateral_kick_level": level.split(":", 1)[1]}
    if level in ("movpad", "moving_pad"):
        return {"moving_pad": True}
    if level.startswith("combo:"):
        opts = {}
        for tok in level.split(":", 1)[1].split("+"):
            if tok.startswith("wind="):
                opts["wind_level"] = tok.split("=", 1)[1]
            elif tok.startswith("kick="):
                opts["lateral_kick_level"] = tok.split("=", 1)[1]
            elif tok in ("movpad", "moving_pad"):
                opts["moving_pad"] = True
            elif tok.startswith("perturb="):
                opts["perturbation_level"] = tok.split("=", 1)[1]
        return opts
    return {"perturbation_level": level}


_ORACLE_STATE_ATTRS = (
    "landing_pad_contact",
    "ang_vel",
    "lin_vel",
    "lin_pos",
    "ground_lin_vel",
    "previous_ang_vel",
    "previous_lin_vel",
    "previous_lin_pos",
    "previous_ground_lin_vel",
    "state",
    "reward",
    "termination",
    "truncation",
    "info",
    "action",
    "ou_pad",
    "moving_pad_enabled",
    "_pad_velocity",
    "_pad_position",
    "landing_pad_position",
    "wind",
    "gust",
    "lateral_kick",
    "_wind_log",
)


def _raw_env_from_world(world):
    env0 = world.envs.envs[0]
    return getattr(env0, "unwrapped", env0)


def _snapshot_oracle_env(raw_env):
    return {
        name: copy.deepcopy(getattr(raw_env, name))
        for name in _ORACLE_STATE_ATTRS
        if hasattr(raw_env, name)
    }


def _restore_oracle_env(raw_env, state_id, snapshot):
    p.restoreState(stateId=state_id, physicsClientId=raw_env.env._client)
    for name, value in snapshot.items():
        setattr(raw_env, name, copy.deepcopy(value))


def _true_rocket_V_np(
    x,
    *,
    alpha: float,
    beta: float,
    pos_idx=(14, 15, 16),
    vel_idx=(3, 4, 5),
    omega_idx=(10, 11, 12),
):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    pos = x[list(pos_idx)]
    vel = x[list(vel_idx)]
    omega = x[list(omega_idx)]
    return float(np.dot(pos, pos) + alpha * np.dot(vel, vel) + beta * np.dot(omega, omega))


def _oracle_action_candidates(nominal, low, high, *, n_candidates, sigma, rng):
    nominal = np.asarray(nominal, dtype=np.float32).reshape(-1)
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    candidates = [np.clip(nominal, low, high)]

    if nominal.size == 7:
        candidates.extend([
            np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 0.5, 0.5, 0, 0], dtype=np.float32),
            np.zeros(7, dtype=np.float32),
        ])

    scale = np.maximum(high - low, 1e-6) * float(sigma)
    while len(candidates) < max(1, int(n_candidates)):
        candidates.append(nominal + rng.normal(0.0, scale, size=nominal.shape).astype(np.float32))

    arr = np.stack(candidates[: max(1, int(n_candidates))], axis=0)
    return np.clip(arr, low, high)


def _select_oracle_one_step_action(
    raw_env,
    nominal_action,
    low,
    high,
    *,
    alpha: float,
    beta: float,
    eta: float,
    n_candidates: int,
    sigma: float,
    rng,
):
    """Use the simulator as an oracle one-step action shield.

    This is a diagnostic upper bound for the learned safety observer: if this
    cannot reduce realized violations, the fixed Lyapunov signal is the wrong
    contract. If it can, the remaining failure is observer/model ranking.
    """
    state_id = p.saveState(physicsClientId=raw_env.env._client)
    snapshot = _snapshot_oracle_env(raw_env)
    x0 = np.asarray(raw_env.state, dtype=np.float64).copy()
    V0 = _true_rocket_V_np(x0, alpha=alpha, beta=beta)
    candidates = _oracle_action_candidates(
        nominal_action, low, high,
        n_candidates=n_candidates, sigma=sigma, rng=rng,
    )

    deltas = []
    terminated_flags = []
    truncated_flags = []
    try:
        for cand in candidates:
            _restore_oracle_env(raw_env, state_id, snapshot)
            obs1, _reward, terminated, truncated, _info1 = raw_env.step(cand)
            x1 = np.asarray(getattr(raw_env, "state", obs1), dtype=np.float64)
            V1 = _true_rocket_V_np(x1, alpha=alpha, beta=beta)
            deltas.append((V0 - V1) / (abs(V0) + 1e-6))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
    finally:
        _restore_oracle_env(raw_env, state_id, snapshot)
        p.removeState(stateUniqueId=state_id, physicsClientId=raw_env.env._client)

    deltas = np.asarray(deltas, dtype=np.float64)
    safe = deltas >= float(eta)
    if np.any(safe):
        deviations = np.linalg.norm(candidates - candidates[0], axis=1)
        masked = np.where(safe, deviations, np.inf)
        chosen_idx = int(np.argmin(masked))
        mode = "safe_closest"
    else:
        chosen_idx = int(np.argmax(deltas))
        mode = "least_violating"

    return candidates[chosen_idx].reshape(np.asarray(nominal_action).shape), {
        "oracle_active": True,
        "oracle_mode": mode,
        "oracle_n_candidates": int(candidates.shape[0]),
        "oracle_eta": float(eta),
        "oracle_nominal_delta": float(deltas[0]),
        "oracle_chosen_delta": float(deltas[chosen_idx]),
        "oracle_best_delta": float(np.max(deltas)),
        "oracle_safe_rate": float(np.mean(safe)),
        "oracle_changed_action": bool(chosen_idx != 0),
        "oracle_chosen_idx": chosen_idx,
        "oracle_any_candidate_terminated": bool(np.any(terminated_flags)),
        "oracle_any_candidate_truncated": bool(np.any(truncated_flags)),
    }

def _take_last_frame(px):
    """Reduce a (B, T, H, W, C) or (1, T, H, W, C) tensor/array to (B, H, W, C).

    For the policy / dinowm.get_cost path we need a single frame per env --
    the cost model adds its own time dim via unsqueeze(1).
    """
    arr = np.asarray(px)
    if arr.ndim == 5:
        # (B, T, H, W, C) -> take last temporal frame
        arr = arr[:, -1]
    elif arr.ndim == 4:
        # Either (T, H, W, C) [single env, temporal] or (B, H, W, C) [no time].
        # If channels are NOT in the last position, can't tell; assume
        # last-axis channel which is gym convention -> first dim is batch.
        # If the dataset feeds (T, H, W, C) we'd be wrong, but for the rocket
        # vector-env we always have (1, ...).
        pass
    return arr


def _to_chw_time_batch(arr):
    """Convert env images to DINOWM's channel-first batch/time layout."""
    arr = np.asarray(arr)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3:           # (H, W, C) -> (1, 1, C, H, W)
        return arr.transpose(2, 0, 1)[None, None]
    if arr.ndim == 4:
        # Vector env without history: (B, H, W, C).
        if arr.shape[-1] in (3, 4):
            return arr.transpose(0, 3, 1, 2)[:, None]
        return arr
    if arr.ndim == 5:           # (B, T, H, W, C) -> (B, T, C, H, W)
        return arr.transpose(0, 1, 4, 2, 3)
    return arr


def _to_time_batch(arr, batch_size: int = 1):
    """Normalize vector-env state histories to (B, T, D)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, None]
    if arr.ndim == 2:
        if arr.shape[0] == batch_size:
            return arr[:, None]
        return arr[None]
    return arr


def _packed_action_history(
    executed_actions: list[np.ndarray],
    history_size: int,
    action_block: int,
    primitive_dim: int,
    batch_size: int = 1,
) -> np.ndarray:
    """Build packed known action blocks between observed history frames."""
    n_hist = max(history_size - 1, 0)
    if n_hist == 0:
        return np.zeros((batch_size, 0, primitive_dim * action_block), dtype=np.float32)

    t = len(executed_actions)
    zero = np.zeros(primitive_dim, dtype=np.float32)
    blocks = []
    for i in range(n_hist):
        start = t - (n_hist - i) * action_block
        pieces = []
        for j in range(action_block):
            idx = start + j
            if 0 <= idx < t:
                pieces.append(np.asarray(executed_actions[idx], dtype=np.float32).reshape(-1))
            else:
                pieces.append(zero)
        blocks.append(np.concatenate(pieces, axis=0))
    return np.stack(blocks, axis=0)[None]


def _first_image(source):
    """Pull a (H, W, C) uint8 image out of a vector-env obs/info dict.

    Possible incoming shapes:
        (H, W, C)            -- unvectorised env
        (1, H, W, C)         -- vector-env wrapper, single env
        (T, H, W, C)         -- temporal window of frames
        (1, T, H, W, C)      -- vector + temporal (the rocket case)
    We always return the most recent frame as (H, W, C).
    """
    if isinstance(source, dict) and "pixels" in source:
        px = source["pixels"]
    else:
        px = source
    if isinstance(px, (list, tuple)):
        px = px[0]
    arr = np.asarray(px)
    # Strip leading env dim if present.
    if arr.ndim == 5:
        arr = arr[0]                      # (T, H, W, C)
    if arr.ndim == 4:
        # Either (1, H, W, C) [single env] or (T, H, W, C) [temporal window].
        # Take the last frame in either case (most recent observation).
        arr = arr[-1]
    # The rocket env renders RGBA; DINO expects RGB. Drop the alpha channel.
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def encode_pixels(world_model, pixels, device):
    """Encode (H, W, 3) uint8 pixels -> (embed_dim,) DINO patch-mean embedding.

    Accepts either a DINOWM instance directly (our case after AutoCostModel)
    or a wrapper with `.model` pointing at the DINOWM (legacy v3 path).
    """
    if pixels.ndim == 3:
        pixels = pixels[None]
    x = torch.as_tensor(pixels, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    x = (x - 0.5) / 0.5
    info = {"pixels": x.unsqueeze(1)}                  # (B, T=1, 3, H, W)
    enc_obj = world_model.model if hasattr(world_model, "model") and hasattr(world_model.model, "encode") else world_model
    with torch.no_grad():
        info = enc_obj.encode(info, target="embed", pixels_key="pixels")
    z = info["pixels_embed"].mean(dim=2).squeeze(1)    # average over patches -> (B, embed_dim)
    return z.squeeze(0)


def load_decoder(path, embed_dim, state_dim=17, architecture="mlp_2", device="cpu"):
    """Lazy import: only used when --decoder is supplied explicitly.

    The default code path uses the ObsProbe baked into the WM checkpoint, so
    StateDecoder is not required for typical use.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from train_state_decoder import StateDecoder
    decoder = StateDecoder(embed_dim=embed_dim, state_dim=state_dim, architecture=architecture)
    sd = torch.load(path, map_location=device, weights_only=False)
    state = sd.get("state_dict") or sd.get("decoder") or sd if isinstance(sd, dict) else sd
    decoder.load_state_dict(state)
    return decoder.to(device).eval()


def _extract_probe_from_wm(world_model, embed_dim, state_dim, arch, device):
    """Pull the ObsProbe baked into the LeJEPA-WM checkpoint, if present.

    The lejepa_wm.py training loop attaches its parallel-detached probe as
    `world_model.obs_probe`, so the saved checkpoint object carries it
    along. We prefer this path because it guarantees the probe matches
    the encoder's embedding dim and was trained on the same data.
    """
    probe = getattr(world_model, "obs_probe", None)
    if probe is None:
        return None
    return probe.to(device).eval()


def run_clf_eval(
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
    alpha: float = 0.5,
    beta: float = 0.1,
    clf_normalized_descent: bool = True,
    plan_diagnostics: bool = False,
    normalize_actions: bool = False,
    project_actions: bool = False,
    hard_clf_filter: bool = False,
    hard_clf_weight: float = 1e6,
    return_best_candidate: bool = False,
    runtime_shield: bool = False,
    shield_margin: float = 0.0,
    shield_calibration_epsilon: float = 0.0,
    shield_full_rollout: bool = True,
    shield_use_v_head: bool = False,
    shield_use_delta_head: bool = False,
    use_v_head: bool = True,
    use_delta_head: bool = True,
    history_size: int = 3,
    frame_skip: int = 2,
    oracle_action_shield: bool = False,
    oracle_candidates: int = 32,
    oracle_sigma: float = 0.25,
    oracle_eta: float = 0.0,
):
    # The training-time pickle of the WM object references the ObsProbe
    # class as `__main__.ObsProbe` (since lejepa_wm.py was the __main__
    # entry point). Inject it into __main__ before torch.load to satisfy
    # the unpickler.
    import sys as _sys, importlib as _importlib
    _train_dir = str(Path(__file__).resolve().parent / "train")
    if _train_dir not in _sys.path:
        _sys.path.insert(0, _train_dir)
    try:
        _lejepa_mod = _importlib.import_module("lejepa_wm")
        for _name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
            if hasattr(_lejepa_mod, _name):
                setattr(_sys.modules["__main__"], _name, getattr(_lejepa_mod, _name))
    except Exception as _e:
        print(f"[warn] could not preload ObsProbe symbol: {_e}")

    print(f"Loading world model: {model_name}")
    base_model = AutoCostModel(model_name)
    # AutoCostModel returns the module on CPU. Move it (and any submodules)
    # to the eval device so the encode path doesn't hit a CUDA/CPU mismatch.
    try:
        base_model.to(device)
        base_model.eval()
    except Exception as _e:
        print(f"[warn] base_model.to({device}) failed: {_e}")
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size

    # Prefer probe baked into the checkpoint; fall back to --decoder.
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, decoder_arch, device)
    if decoder is not None:
        print(f"Using ObsProbe baked into world-model checkpoint "
              f"(arch from training, no separate decoder file needed)")
    else:
        if decoder_path is None:
            raise RuntimeError(
                f"World model '{model_name}' has no .obs_probe attribute and "
                f"no --decoder path was provided. Either retrain with the "
                f"probe enabled (lejepa_wm.py) or pass --decoder.")
        print(f"Loading decoder: {decoder_path} (arch={decoder_arch})")
        decoder = load_decoder(decoder_path, embed_dim=embed_dim,
                               architecture=decoder_arch, device=device)

    if probe_output_space == "normalized":
        print(f"Loading proprio normalization stats: {norm_dataset}")
        p_mean, p_std = compute_proprio_stats(norm_dataset)
        decoder = ProbeStateAdapter(
            decoder, mean=p_mean, std=p_std, output_space=probe_output_space,
        ).to(device).eval()
        print("Probe outputs normalized state; CLF/monitor will unnormalize before V")
    elif probe_output_space == "raw":
        decoder = ProbeStateAdapter(decoder, output_space=probe_output_space).to(device).eval()
        print("Probe outputs raw state; CLF/monitor will use decoder output directly")
    else:
        raise ValueError(f"Unsupported probe_output_space: {probe_output_space}")

    # Wrap base cost with CLF augmentation. lambda_V=0 -> base cost unchanged.
    # The rocket observation contains target_relative = pad_pos - position in
    # dims 14:17. Its norm is the same as ||position - pad_pos|| and remains
    # correct when the pad moves.
    V_fn = make_rocket_V(
        alpha=alpha, beta=beta, pad_pos=(0.0, 0.0, 0.0),
        pos_idx=(14, 15, 16),
    )
    plan_cfg = PlanConfig(horizon=horizon, receding_horizon=max(horizon // 3, 1),
                          history_len=history_size, action_block=frame_skip,
                          warm_start=True)
    cost_model = CLFAugmentedCost(
        base_cost_model=base_model,
        decoder=decoder,
        compute_V=V_fn,
        cfg=CLFCostConfig(
            lambda_V=lambda_V, eta=eta,
            normalized_descent=clf_normalized_descent,
            hard_filter=hard_clf_filter and lambda_V > 0.0,
            hard_filter_weight=hard_clf_weight,
            runtime_shield=runtime_shield and lambda_V > 0.0,
            shield_margin=shield_margin,
            calibration_epsilon=shield_calibration_epsilon,
            shield_full_rollout=shield_full_rollout,
            shield_use_v_head=shield_use_v_head,
            shield_use_delta_head=shield_use_delta_head,
            use_v_head=use_v_head,
            use_delta_head=use_delta_head,
        ),
        device=device,
    )

    if project_actions and not normalize_actions:
        print("[warn] action projection requires normalized action codec; enabling --normalize-actions")
        normalize_actions = True

    process = {}
    action_stats = None
    proprio_normalizer = DatasetColumnNormalizer.from_dataset(
        norm_dataset,
        column="proprio",
    )
    process["proprio"] = proprio_normalizer
    proprio_stats = {
        "mean": [float(x) for x in proprio_normalizer.mean.tolist()],
        "std": [float(x) for x in proprio_normalizer.std.tolist()],
    }
    print("Using training proprio normalizer for model context")
    if normalize_actions:
        action_normalizer = RocketActionNormalizer.from_dataset(
            norm_dataset,
            action_block=plan_cfg.action_block,
        )
        process["action"] = action_normalizer
        action_stats = {
            "mean": [float(x) for x in action_normalizer.mean.tolist()],
            "std": [float(x) for x in action_normalizer.std.tolist()],
            "action_block": int(action_normalizer.action_block),
        }
        print("Using training action normalizer for policy history and execution inverse")
        print(f"  action mean = {np.array2string(action_normalizer.mean, precision=4)}")
        print(f"  action std  = {np.array2string(action_normalizer.std, precision=4)}")

        if project_actions:
            action_adapter = RocketActionAdapter(
                normalizer=action_normalizer,
                action_block=plan_cfg.action_block,
            )
            cost_model = ActionSpaceCostAdapter(cost_model, action_adapter)
            print("CEM candidates will be projected through physical rocket action bounds before scoring")

    solver = swm.solver.CEMSolver(
        model=cost_model,
        num_samples=num_samples, var_scale=0.5,
        n_steps=n_steps, topk=max(num_samples // 10, 5), device=device,
        diagnostics_enabled=plan_diagnostics,
        return_best_candidate=return_best_candidate,
    )

    arm_label = "CLF-MPC" if lambda_V > 0 else "standard-CEM"
    results = {"arm": arm_label, "lambda_V": lambda_V, "eta": eta,
               "horizon": horizon, "num_samples": num_samples, "n_steps": n_steps,
               "probe_output_space": probe_output_space,
               "norm_dataset": norm_dataset,
               "clf_normalized_descent": clf_normalized_descent,
               "plan_diagnostics": plan_diagnostics,
               "normalize_actions": normalize_actions,
               "project_actions": project_actions,
               "hard_clf_filter": hard_clf_filter and lambda_V > 0.0,
               "hard_clf_weight": hard_clf_weight,
               "return_best_candidate": return_best_candidate,
               "runtime_shield": runtime_shield and lambda_V > 0.0,
               "shield_margin": shield_margin,
               "shield_calibration_epsilon": shield_calibration_epsilon,
               "shield_full_rollout": shield_full_rollout,
               "shield_use_v_head": shield_use_v_head,
               "shield_use_delta_head": shield_use_delta_head,
               "use_v_head": use_v_head,
               "use_delta_head": use_delta_head,
               "history_size": history_size,
               "frame_skip": frame_skip,
               "oracle_action_shield": oracle_action_shield,
               "oracle_candidates": oracle_candidates,
               "oracle_sigma": oracle_sigma,
               "oracle_eta": oracle_eta,
               "action_stats": action_stats,
               "proprio_stats": proprio_stats,
               "v_pos_idx": [14, 15, 16],
               "by_level": {}}

    for level in levels:
        print(f"\n--- {arm_label} (lambda_V={lambda_V}, eta={eta}): {level} ---")
        world = swm.World(
            "swm/PFRocketLandingExt-v0",
            num_envs=1, image_shape=(224, 224),
            history_size=history_size, frame_skip=frame_skip,
            max_episode_steps=1200, render_mode="rgb_array",
        )
        policy = WorldModelPolicy(
            solver=solver,
            config=plan_cfg,
            process=process if process else None,
        )
        world.set_policy(policy)

        ep_records = []
        for ep in range(episodes):
            mon = LyapunovMonitor(
                decoder=decoder, device=device,
                cfg=LyapunovConfig(
                    alpha=alpha, beta=beta, pad_pos=(0.0, 0.0, 0.0),
                    pos_idx=(14, 15, 16),
                ),
            )
            # swm.World.reset() returns None and writes into world.states/infos.
            # Call envs.reset directly. swm.MegaWrapper stores 'pixels' under
            # info, so we read images from info, not from obs.
            obs, info = world.envs.reset(
                seed=seed + ep,
                options=reset_options_for_level(level),
            )
            z_prev = encode_pixels(base_model, _first_image(info), device)

            done = False
            ep_return = 0.0
            plan_diag_records = []
            oracle_diag_records = []
            executed_actions: list[np.ndarray] = []
            primitive_action_dim = int(np.prod(world.envs.action_space.shape[1:]))
            oracle_rng = np.random.default_rng(seed + ep + 7919)
            t = 0
            while not done and t < 1200:
                # Match the training contract: 3-frame, frameskip-2 image and
                # proprio histories plus packed action blocks for the known
                # intervals between observed frames.
                px_now = _to_chw_time_batch(info["pixels"])
                gl_now = _to_chw_time_batch(info["goal"])
                action_hist = _packed_action_history(
                    executed_actions,
                    history_size=history_size,
                    action_block=frame_skip,
                    primitive_dim=primitive_action_dim,
                    batch_size=1,
                )
                # Training data exposes the 17-D rocket state under "proprio";
                # the swm.MegaWrapper at inference time stores the raw env
                # observation under "observation" (since the env's obs is a
                # Box, not a dict). Alias it so the encoder finds the proprio
                # branch and produces the same 448-D embedding the predictor
                # was trained on (pixels-384 + proprio-32 + action-32).
                proprio_arr = None
                if "proprio" in info:
                    proprio_arr = info["proprio"]
                elif "observation" in info:
                    proprio_arr = info["observation"]
                if proprio_arr is not None:
                    proprio_arr = _to_time_batch(proprio_arr, batch_size=1)
                policy_input = {
                    "pixels": px_now,
                    "goal": gl_now,
                    "action": action_hist,
                }
                if proprio_arr is not None:
                    policy_input["proprio"] = proprio_arr
                action = policy.get_action(policy_input)
                if plan_diagnostics and getattr(policy, "just_replanned", False):
                    outputs = getattr(policy, "last_solver_outputs", None) or {}
                    diag = outputs.get("diagnostics")
                    if diag:
                        def _first_env(value):
                            if isinstance(value, list) and len(value) == 1:
                                return value[0]
                            return value

                        rec = {
                            "t": int(t),
                            "available": bool(diag.get("available", False)),
                        }
                        for key in (
                            "base_cost", "clf_penalty", "total_cost",
                            "pred_violation_rate", "pred_mean_descent",
                            "pred_min_descent", "pred_first_delta",
                            "pred_max_step_violation", "pred_feasible",
                            "pred_V_trace", "pred_delta_trace",
                            "pred_step_violation_trace",
                            "hard_filter_penalty",
                        ):
                            if key in diag:
                                rec[key] = _first_env(diag[key])
                        if "diagnostics_error" in outputs:
                            rec["diagnostics_error"] = outputs["diagnostics_error"]
                        shield_diag = outputs.get("shield_diagnostics") or []
                        if shield_diag:
                            latest_shield = shield_diag[-1]
                            for key in (
                                "shield_feasible_rate",
                                "shield_any_feasible",
                                "shield_full_rollout",
                                "shield_first_exec_idx",
                                "shield_chosen_violation",
                                "shield_chosen_raw_violation",
                                "shield_chosen_max_violation",
                                "shield_chosen_max_raw_violation",
                                "shield_chosen_sum_violation",
                                "shield_calibration_epsilon",
                                "shield_using_v_head",
                                "shield_using_delta_head",
                            ):
                                if key in latest_shield:
                                    rec[key] = _first_env(latest_shield[key])
                        plan_diag_records.append(rec)
                # CEM samples actions freely from a Gaussian; clip to the
                # env's action-space bounds before stepping. The rocket env's
                # ignition / throttle channels (indices 3-4) live in [0, 1]
                # while the gimbal / RCS channels live in [-1, 1].
                a_low = np.asarray(world.envs.action_space.low)
                a_high = np.asarray(world.envs.action_space.high)
                action = np.clip(np.asarray(action), a_low, a_high)
                if oracle_action_shield:
                    raw_env = _raw_env_from_world(world)
                    shielded_action, oracle_diag = _select_oracle_one_step_action(
                        raw_env,
                        action,
                        a_low,
                        a_high,
                        alpha=alpha,
                        beta=beta,
                        eta=oracle_eta,
                        n_candidates=oracle_candidates,
                        sigma=oracle_sigma,
                        rng=oracle_rng,
                    )
                    oracle_diag["t"] = int(t)
                    oracle_diag_records.append(oracle_diag)
                    action = np.clip(np.asarray(shielded_action), a_low, a_high)
                obs, reward, terminated, truncated, info = world.envs.step(action)
                executed_actions.append(np.asarray(action).reshape(-1).copy())
                z_curr = encode_pixels(base_model, _first_image(info), device)
                mon.step(z_prev, z_curr)
                z_prev = z_curr
                ep_return += float(np.asarray(reward).sum())
                t += 1
                done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())

            summary = mon.summary()
            summary.update(mon.trace())
            if plan_diag_records:
                pred_viol = [
                    float(r["pred_violation_rate"])
                    for r in plan_diag_records
                    if r.get("available") and "pred_violation_rate" in r
                ]
                pred_desc = [
                    float(r["pred_mean_descent"])
                    for r in plan_diag_records
                    if r.get("available") and "pred_mean_descent" in r
                ]
                first_delta = [
                    float(r["pred_first_delta"])
                    for r in plan_diag_records
                    if r.get("available") and "pred_first_delta" in r
                ]
                penalty = [
                    float(r["clf_penalty"])
                    for r in plan_diag_records
                    if r.get("available") and "clf_penalty" in r
                ]
                hard_penalty = [
                    float(r["hard_filter_penalty"])
                    for r in plan_diag_records
                    if r.get("available") and "hard_filter_penalty" in r
                ]
                max_step_violation = [
                    float(r["pred_max_step_violation"])
                    for r in plan_diag_records
                    if r.get("available") and "pred_max_step_violation" in r
                ]
                feasible = [
                    float(bool(r["pred_feasible"]))
                    for r in plan_diag_records
                    if r.get("available") and "pred_feasible" in r
                ]
                shield_feasible = [
                    float(r["shield_feasible_rate"])
                    for r in plan_diag_records
                    if r.get("available") and "shield_feasible_rate" in r
                ]
                shield_chosen = [
                    float(r["shield_chosen_violation"])
                    for r in plan_diag_records
                    if r.get("available") and "shield_chosen_violation" in r
                ]
                summary.update({
                    "plan_diagnostics": plan_diag_records,
                    "plan_pred_violation_rate": float(np.mean(pred_viol)) if pred_viol else float("nan"),
                    "plan_pred_mean_descent": float(np.mean(pred_desc)) if pred_desc else float("nan"),
                    "plan_pred_first_delta": float(np.mean(first_delta)) if first_delta else float("nan"),
                    "plan_clf_penalty": float(np.mean(penalty)) if penalty else float("nan"),
                    "plan_hard_filter_penalty": float(np.mean(hard_penalty)) if hard_penalty else float("nan"),
                    "plan_pred_max_step_violation": (
                        float(np.mean(max_step_violation)) if max_step_violation else float("nan")
                    ),
                    "plan_pred_feasible_rate": float(np.mean(feasible)) if feasible else float("nan"),
                    "shield_feasible_rate": (
                        float(np.mean(shield_feasible)) if shield_feasible else float("nan")
                    ),
                    "shield_chosen_violation": (
                        float(np.mean(shield_chosen)) if shield_chosen else float("nan")
                    ),
                    "n_replans": len(plan_diag_records),
                })
            if oracle_diag_records:
                summary.update({
                    "oracle_diagnostics": oracle_diag_records,
                    "oracle_nominal_violation_rate": float(np.mean([
                        r["oracle_nominal_delta"] < oracle_eta
                        for r in oracle_diag_records
                    ])),
                    "oracle_chosen_violation_rate": float(np.mean([
                        r["oracle_chosen_delta"] < oracle_eta
                        for r in oracle_diag_records
                    ])),
                    "oracle_best_violation_rate": float(np.mean([
                        r["oracle_best_delta"] < oracle_eta
                        for r in oracle_diag_records
                    ])),
                    "oracle_safe_candidate_rate": float(np.mean([
                        r["oracle_safe_rate"] for r in oracle_diag_records
                    ])),
                    "oracle_action_change_rate": float(np.mean([
                        r["oracle_changed_action"] for r in oracle_diag_records
                    ])),
                })
            summary.update({"success": bool(np.asarray(terminated).any()),
                            "return": ep_return, "T": t,
                            "seed": int(seed + ep)})
            ep_records.append(summary)
            print(f"  ep {ep}: success={summary['success']} return={ep_return:.1f} "
                  f"mean_descent={summary['mean_descent']:+.3f} "
                  f"viol={summary['violation_rate']:.2f}"
                  + (f" pred_viol={summary.get('plan_pred_violation_rate', float('nan')):.2f}"
                     if plan_diag_records else "")
                  + (f" oracle_chosen_viol={summary.get('oracle_chosen_violation_rate', float('nan')):.2f}"
                     if oracle_diag_records else ""))

        success = float(np.mean([r["success"] for r in ep_records]))
        agg = {
            "success_rate": success,
            "mean_descent_rate": float(np.mean([r["mean_descent"] for r in ep_records])),
            "violation_rate":   float(np.mean([r["violation_rate"] for r in ep_records])),
            "mean_return":      float(np.mean([r["return"] for r in ep_records])),
            "mean_steps":       float(np.mean([r["T"] for r in ep_records])),
            "n_episodes":       len(ep_records),
            "episodes":         ep_records,
        }
        if any("plan_pred_violation_rate" in r for r in ep_records):
            agg.update({
                "plan_pred_violation_rate": float(np.nanmean([
                    r.get("plan_pred_violation_rate", np.nan) for r in ep_records
                ])),
                "plan_pred_mean_descent": float(np.nanmean([
                    r.get("plan_pred_mean_descent", np.nan) for r in ep_records
                ])),
                "plan_clf_penalty": float(np.nanmean([
                    r.get("plan_clf_penalty", np.nan) for r in ep_records
                ])),
                "plan_hard_filter_penalty": float(np.nanmean([
                    r.get("plan_hard_filter_penalty", np.nan) for r in ep_records
                ])),
                "plan_pred_max_step_violation": float(np.nanmean([
                    r.get("plan_pred_max_step_violation", np.nan) for r in ep_records
                ])),
                "plan_pred_feasible_rate": float(np.nanmean([
                    r.get("plan_pred_feasible_rate", np.nan) for r in ep_records
                ])),
                "shield_feasible_rate": float(np.nanmean([
                    r.get("shield_feasible_rate", np.nan) for r in ep_records
                ])),
                "shield_chosen_violation": float(np.nanmean([
                    r.get("shield_chosen_violation", np.nan) for r in ep_records
                ])),
                "mean_replans": float(np.mean([
                    r.get("n_replans", 0) for r in ep_records
                ])),
            })
        if any("oracle_chosen_violation_rate" in r for r in ep_records):
            agg.update({
                "oracle_nominal_violation_rate": float(np.nanmean([
                    r.get("oracle_nominal_violation_rate", np.nan) for r in ep_records
                ])),
                "oracle_chosen_violation_rate": float(np.nanmean([
                    r.get("oracle_chosen_violation_rate", np.nan) for r in ep_records
                ])),
                "oracle_best_violation_rate": float(np.nanmean([
                    r.get("oracle_best_violation_rate", np.nan) for r in ep_records
                ])),
                "oracle_safe_candidate_rate": float(np.nanmean([
                    r.get("oracle_safe_candidate_rate", np.nan) for r in ep_records
                ])),
                "oracle_action_change_rate": float(np.nanmean([
                    r.get("oracle_action_change_rate", np.nan) for r in ep_records
                ])),
            })
        print(f"[{level}] success={success*100:.1f}% "
              f"mean_desc={agg['mean_descent_rate']:+.3f} "
              f"viol={agg['violation_rate']:.2f}")
        results["by_level"][level] = agg
    return results


def main():
    p = argparse.ArgumentParser(description="Rocket CLF-augmented CEM eval")
    p.add_argument("--model", required=True,
                   help="WM object name (without _object.ckpt)")
    p.add_argument("--decoder", default=None,
                   help="Optional state-decoder .pt path. If omitted, the "
                        "ObsProbe baked into the WM checkpoint is used.")
    p.add_argument("--decoder-arch", default="mlp_2",
                   choices=["linear", "mlp_2", "mlp_3"])
    p.add_argument("--norm-dataset",
                   default="data/expert_trajectories_union/rocket_expert_union",
                   help="Dataset used to recover proprio mean/std for normalized probes.")
    p.add_argument("--probe-output-space", default="normalized",
                   choices=["normalized", "raw"],
                   help="Space emitted by the observation probe before this evaluator adapts it.")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels", nargs="+",
                   default=["easy", "medium", "hard", "extreme"])
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lambda-V", type=float, default=0.0,
                   help="CLF descent-penalty weight. 0 = standard CEM (P1).")
    p.add_argument("--eta", type=float, default=0.0,
                   help="Slack in the descent inequality.")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--clf-descent-mode", default="normalized",
                   choices=["normalized", "absolute"],
                   help="Penalty mode: normalized matches the monitor's fractional descent.")
    p.add_argument("--plan-diagnostics", action="store_true",
                   help="Log predicted V/delta traces for each selected MPC plan.")
    p.add_argument("--normalize-actions", action="store_true",
                   help="Run CEM in the normalized action space used during WM training.")
    p.add_argument("--project-actions", action="store_true",
                   help="Project normalized CEM candidates through physical rocket action bounds before scoring.")
    p.add_argument("--hard-clf-filter", action="store_true",
                   help="Prefer CLF-feasible candidates via a large predicted-violation penalty.")
    p.add_argument("--hard-clf-weight", type=float, default=1e6,
                   help="Large infeasibility weight used by --hard-clf-filter.")
    p.add_argument("--return-best-candidate", action="store_true",
                   help="Execute the best scored final CEM candidate instead of the elite mean.")
    p.add_argument("--runtime-shield", action="store_true",
                   help="At execution time choose a final candidate satisfying the calibrated CLF shield.")
    p.add_argument("--shield-margin", type=float, default=0.0,
                   help="Allowed calibrated shield violation for --runtime-shield.")
    p.add_argument("--shield-calibration-epsilon", type=float, default=0.0,
                   help="Conservative prediction-error margin added to the CLF shield inequality.")
    p.add_argument("--shield-one-step", action="store_true",
                   help="Apply the shield only to the first executed transition instead of the full rollout.")
    p.add_argument("--shield-use-v-head", action="store_true",
                   help="Use the checkpoint V head inside the runtime shield. Default shield uses decoded-state V.")
    p.add_argument("--shield-use-delta-head", action="store_true",
                   help="Use the checkpoint delta head inside the runtime shield. Default shield derives deltas from decoded-state V.")
    p.add_argument("--disable-v-head", action="store_true",
                   help="Ignore a checkpoint's direct V head and use decoded-state V instead.")
    p.add_argument("--disable-delta-head", action="store_true",
                   help="Ignore a checkpoint's direct DeltaV head and derive deltas from V traces.")
    p.add_argument("--history-size", type=int, default=3,
                   help="Number of observed frames to feed the WM planner.")
    p.add_argument("--frame-skip", type=int, default=2,
                   help="Primitive env steps per WM action block.")
    p.add_argument("--oracle-action-shield", action="store_true",
                   help="Use a simulator one-step oracle to gate the executed action. "
                        "This is a diagnostic upper bound for a learned action-conditioned safety observer.")
    p.add_argument("--oracle-candidates", type=int, default=32,
                   help="Number of one-step action candidates tested by --oracle-action-shield.")
    p.add_argument("--oracle-sigma", type=float, default=0.25,
                   help="Local perturbation scale as a fraction of action range for oracle candidates.")
    p.add_argument("--oracle-eta", type=float, default=0.0,
                   help="Minimum true one-step Lyapunov descent required by the oracle shield.")
    p.add_argument("--out", default=None,
                   help="Output JSON. Default: results_lejepa_v4/clf_eval/<model>__lambda<x>.json")
    args = p.parse_args()

    out = args.out or (
        f"/users/aiyer40/scratch/results_lejepa_v4/clf_eval/"
        f"{args.model}__lambda{args.lambda_V:g}_eta{args.eta:g}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    results = run_clf_eval(
        args.model, args.decoder, args.decoder_arch,
        args.norm_dataset, args.probe_output_space,
        args.horizon, args.episodes,
        args.seed, args.levels, args.num_samples, args.n_steps, args.device,
        lambda_V=args.lambda_V, eta=args.eta,
        alpha=args.alpha, beta=args.beta,
        clf_normalized_descent=(args.clf_descent_mode == "normalized"),
        plan_diagnostics=args.plan_diagnostics,
        normalize_actions=args.normalize_actions,
        project_actions=args.project_actions,
        hard_clf_filter=args.hard_clf_filter,
        hard_clf_weight=args.hard_clf_weight,
        return_best_candidate=args.return_best_candidate,
        runtime_shield=args.runtime_shield,
        shield_margin=args.shield_margin,
        shield_calibration_epsilon=args.shield_calibration_epsilon,
        shield_full_rollout=not args.shield_one_step,
        shield_use_v_head=args.shield_use_v_head,
        shield_use_delta_head=args.shield_use_delta_head,
        use_v_head=not args.disable_v_head,
        use_delta_head=not args.disable_delta_head,
        history_size=args.history_size,
        frame_skip=args.frame_skip,
        oracle_action_shield=args.oracle_action_shield,
        oracle_candidates=args.oracle_candidates,
        oracle_sigma=args.oracle_sigma,
        oracle_eta=args.oracle_eta,
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()

"""SE-3: Innovation EMA tracks wind disturbance magnitude.

Runs episodes at each disturbance level; at each step records:
  - the latent innovation ||z_t - z_{t-1}||
  - the EMA of innovation D_t
  - the ground-truth wind force magnitude ||F_wind||
Then computes correlation between D_t and ||F_wind||.

A positive correlation justifies the adaptive gating: the innovation IS a
disturbance estimator, not just a heuristic.

Compute cost: ~5-10 min on GPU.

Outputs:
  results/se3_innovation_wind.json
  results/paper_figures/fig_se3_innovation_wind.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
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

from evaluate_rocket_clf import _first_image, encode_pixels, reset_options_for_level
import stable_worldmodel as swm
from stable_worldmodel.envs.pdg_expert_robust import wind_force_from_env
from stable_worldmodel.policy import AutoCostModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rocket_lejepa_wm_union_vhead_rolloutprobe")
    ap.add_argument("--num-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--ema-beta", type=float, default=0.2)
    ap.add_argument("--output", default="results/se3_innovation_wind.json")
    ap.add_argument("--fig-out", default="results/paper_figures/fig_se3_innovation_wind.pdf")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    base_model = AutoCostModel(args.checkpoint).to(device).eval()

    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=3, frame_skip=2,
        max_episode_steps=args.max_steps + 50, render_mode="rgb_array",
    )

    all_innov = []        # per-step latent innovation
    all_ema = []          # EMA running estimate
    all_wind = []         # ground-truth ||F_wind||
    all_level = []        # disturbance level label

    rng = np.random.default_rng(42)
    levels = ["easy", "medium", "hard", "extreme"]
    for level in levels:
        for ep in range(args.num_episodes):
            seed = 42 + ep
            opts = reset_options_for_level(level)
            obs, info = world.envs.reset(seed=seed, options=opts)
            z_prev = encode_pixels(base_model, _first_image(info), device)
            ema = 0.0

            for t in range(args.max_steps):
                action = rng.uniform(world.envs.action_space.low, world.envs.action_space.high)
                obs, _, terminated, truncated, info = world.envs.step(action)
                z_curr = encode_pixels(base_model, _first_image(info), device)

                z_a = z_prev.mean(dim=-2) if z_prev.dim() >= 3 else z_prev
                z_b = z_curr.mean(dim=-2) if z_curr.dim() >= 3 else z_curr
                innov = float((z_b - z_a).reshape(-1).norm().item())
                ema = (1.0 - args.ema_beta) * ema + args.ema_beta * innov

                # Ground-truth wind force.
                unwrapped = world.envs.envs[0].unwrapped
                F_wind = wind_force_from_env(unwrapped)
                wind_mag = float(np.linalg.norm(F_wind))

                all_innov.append(innov)
                all_ema.append(ema)
                all_wind.append(wind_mag)
                all_level.append(level)

                z_prev = z_curr
                done = (hasattr(terminated, "__len__") and bool(terminated[0])) or \
                       (hasattr(truncated, "__len__") and bool(truncated[0]))
                if done:
                    break
        print(f"[se3] level={level}: collected {sum(1 for l in all_level if l == level)} samples")

    innov = np.asarray(all_innov)
    ema   = np.asarray(all_ema)
    wind  = np.asarray(all_wind)
    labels = np.asarray(all_level)

    # Pearson correlation across all samples and per-level means.
    rho_innov  = float(np.corrcoef(innov, wind)[0, 1]) if np.std(innov) > 0 and np.std(wind) > 0 else float("nan")
    rho_ema    = float(np.corrcoef(ema,   wind)[0, 1]) if np.std(ema)   > 0 and np.std(wind) > 0 else float("nan")

    per_level = {}
    for lvl in levels:
        mask = labels == lvl
        per_level[lvl] = {
            "n": int(mask.sum()),
            "innov_mean":  float(innov[mask].mean()) if mask.any() else 0,
            "ema_mean":    float(ema[mask].mean())   if mask.any() else 0,
            "wind_mean":   float(wind[mask].mean())  if mask.any() else 0,
            "wind_max":    float(wind[mask].max())   if mask.any() else 0,
        }

    print()
    print(f"  level     n     innov  ema     wind_mag")
    for lvl in levels:
        p = per_level[lvl]
        print(f"  {lvl:<7}  {p['n']:>4}  {p['innov_mean']:>5.2f}  {p['ema_mean']:>5.2f}  {p['wind_mean']:>7.2f}")
    print()
    print(f"[se3] corr(innov, |F_wind|) = {rho_innov:+.3f}")
    print(f"[se3] corr(ema,   |F_wind|) = {rho_ema:+.3f}")

    out = {
        "n_samples": int(len(innov)),
        "ema_beta": args.ema_beta,
        "max_steps": args.max_steps,
        "corr_innov_wind": rho_innov,
        "corr_ema_wind":   rho_ema,
        "per_level":       per_level,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[se3] wrote {args.output}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

        # Panel A: scatter of EMA vs |F_wind| colored by level
        colors = {"easy": "#4C72B0", "medium": "#55A467", "hard": "#DD8452", "extreme": "#C44E52"}
        for lvl in levels:
            mask = labels == lvl
            axes[0].scatter(wind[mask], ema[mask], s=4, alpha=0.3, color=colors[lvl], label=lvl)
        axes[0].set_xlabel(r"$\|F_{\mathrm{wind}}\|$ (N)")
        axes[0].set_ylabel(r"Innovation EMA $D_t$")
        axes[0].set_title(f"SE-3a: D_t tracks wind magnitude ($\\rho = {rho_ema:+.3f}$)")
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Panel B: per-level means as bar plot
        means = [per_level[l]["ema_mean"] for l in levels]
        winds = [per_level[l]["wind_mean"] for l in levels]
        x_pos = np.arange(len(levels))
        ax_b2 = axes[1].twinx()
        bars1 = axes[1].bar(x_pos - 0.2, means, width=0.4, label=r"$D_t$ mean", color="#4C72B0", alpha=0.8)
        bars2 = ax_b2.bar(x_pos + 0.2, winds, width=0.4, label=r"$\|F_{\mathrm{wind}}\|$ mean", color="#DD8452", alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(levels)
        axes[1].set_ylabel(r"Innovation EMA $D_t$", color="#4C72B0")
        ax_b2.set_ylabel(r"$\|F_{\mathrm{wind}}\|$ (N)", color="#DD8452")
        axes[1].set_title("SE-3b: per-level means scale together")
        axes[1].spines["top"].set_visible(False)
        ax_b2.spines["top"].set_visible(False)

        fig.tight_layout()
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_out, dpi=160)
        fig.savefig(Path(args.fig_out).with_suffix(".png"), dpi=160)
        print(f"[se3] wrote {args.fig_out}")
    except Exception as e:
        print(f"[se3] plotting failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

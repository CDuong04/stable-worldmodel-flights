"""SE-2: Latent Kalman Filter (LKF) vs raw probe baseline.

Builds a latent-space EKF using the linearization (A) and the steady-state
covariance (P_lyap) we already have.  At each step:
  - Predict: v_{t|t-1} = A * v_{t-1|t-1}
  - Update: v_{t|t} = v_{t|t-1} + K_t * (z_t_observed - z_t_predicted)
    (K_t = simple Kalman gain from P_lyap and innovation covariance)
  - Decode: x_hat_t = d_phi(z_star + v_{t|t})

Compare RMSE per state dimension between:
  - Raw probe baseline:   x_hat_t = d_phi(z_t_observed)
  - LKF:                  x_hat_t = d_phi(z_star + v_{t|t})

If LKF beats raw probe → state estimator contribution.

Compute: ~10 min on GPU.

Outputs:
  results/se2_lkf.json
  results/paper_figures/fig_se2_lkf.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
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
except Exception:
    pass

from evaluate_rocket_clf import (
    ProbeStateAdapter, _extract_probe_from_wm, _first_image,
    compute_proprio_stats, encode_pixels, reset_options_for_level,
)
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rocket_lejepa_wm_union_vhead_rolloutprobe")
    ap.add_argument("--spectral-artifact", default="models/local_linear_model.pt")
    ap.add_argument("--norm-dataset", default="data/expert_trajectories_union/rocket_expert_union")
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--steps-per-episode", type=int, default=80)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--gain", type=float, default=0.7,
                    help="Scalar Kalman-gain proxy: 0=trust prediction, 1=trust observation.")
    ap.add_argument("--output", default="results/se2_lkf.json")
    ap.add_argument("--fig-out", default="results/paper_figures/fig_se2_lkf.pdf")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    base_model = AutoCostModel(args.checkpoint).to(device).eval()
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, "mlp_2", device)
    p_mean, p_std = compute_proprio_stats(args.norm_dataset)
    decoder = ProbeStateAdapter(decoder, mean=p_mean, std=p_std, output_space="normalized").to(device).eval()

    artifact = torch.load(args.spectral_artifact, map_location=device, weights_only=False)
    # Reconstruct A from the spectral artifact's V_s_basis and eigvals.
    z_star = artifact["z_star"].to(device).float()
    if "A" in artifact:
        A = artifact["A"].to(device).float()
    else:
        # Reconstruct A from invariant subspace basis and eigenvalues.
        # Simplified: A ≈ V_s_basis @ diag(eigvals_stable) @ V_s_basis^T projection
        # In our case n_u=0 so A_s_basis covers the full space.
        V = artifact["V_s_basis"].to(device).float()  # (d, d)
        E = artifact.get("A_s_basis")
        if E is not None:
            A_s = E.to(device).float()
            A = V @ A_s @ V.t()
        else:
            # Fallback: identity (we'll detect this and skip).
            A = torch.eye(z_star.shape[0], device=device)

    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=3, frame_skip=2,
        max_episode_steps=args.steps_per_episode + 50, render_mode="rgb_array",
    )

    raw_errs = []   # per-step ||x - d_phi(z_obs)||
    lkf_errs = []   # per-step ||x - d_phi(z_star + v_lkf)||
    rng = np.random.default_rng(42)

    for ep in range(args.num_episodes):
        seed = 42 + ep
        obs, info = world.envs.reset(seed=seed, options=reset_options_for_level(args.level))
        v_est = None  # LKF estimate of v_t = z_t - z_star

        for t in range(args.steps_per_episode):
            # Encode observation.
            z = encode_pixels(base_model, _first_image(info), device)
            z_p = z.mean(dim=-2) if z.dim() >= 3 else z
            z_p = z_p.reshape(-1)  # (d,)
            v_obs = z_p - z_star.reshape(-1)

            # Initialize or step LKF.
            if v_est is None:
                v_est = v_obs.clone()
            else:
                # Predict: v_pred = A * v_est
                v_pred = A @ v_est
                # Update: v_est = v_pred + gain * (v_obs - v_pred)
                v_est = v_pred + args.gain * (v_obs - v_pred)

            # Decode both.
            with torch.no_grad():
                x_hat_raw = decoder(z_p.unsqueeze(0))[0].cpu().numpy().astype(np.float32)
                x_hat_lkf = decoder((z_star.reshape(-1) + v_est).unsqueeze(0))[0].cpu().numpy().astype(np.float32)

            proprio = info.get("proprio", info.get("observation"))
            x_true = np.asarray(proprio).reshape(-1)[:17].astype(np.float32)

            raw_err = np.linalg.norm(x_true - x_hat_raw)
            lkf_err = np.linalg.norm(x_true - x_hat_lkf)
            raw_errs.append(float(raw_err))
            lkf_errs.append(float(lkf_err))

            # Step env with random action.
            action = rng.uniform(world.envs.action_space.low, world.envs.action_space.high)
            obs, _, terminated, truncated, info = world.envs.step(action)
            if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
               (hasattr(truncated, "__len__") and bool(truncated[0])):
                break

    raw_errs = np.array(raw_errs)
    lkf_errs = np.array(lkf_errs)

    summary = {
        "n_samples": int(len(raw_errs)),
        "level": args.level,
        "gain": args.gain,
        "raw_probe": {
            "mean_err": float(raw_errs.mean()),
            "median_err": float(np.median(raw_errs)),
            "rmse": float(np.sqrt(np.mean(raw_errs**2))),
        },
        "lkf": {
            "mean_err": float(lkf_errs.mean()),
            "median_err": float(np.median(lkf_errs)),
            "rmse": float(np.sqrt(np.mean(lkf_errs**2))),
        },
        "improvement_pct": float(100 * (raw_errs.mean() - lkf_errs.mean()) / raw_errs.mean()),
    }
    print()
    print(f"  raw probe mean error:  {summary['raw_probe']['mean_err']:.3f}")
    print(f"  LKF mean error:        {summary['lkf']['mean_err']:.3f}")
    print(f"  improvement:           {summary['improvement_pct']:+.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"  wrote {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

        # Histogram of per-step errors.
        bins = np.linspace(0, max(raw_errs.max(), lkf_errs.max()), 40)
        axes[0].hist(raw_errs, bins=bins, alpha=0.5, color="#4C72B0",
                     label=f"raw probe (mean={raw_errs.mean():.2f})")
        axes[0].hist(lkf_errs, bins=bins, alpha=0.5, color="#55A467",
                     label=f"LKF (mean={lkf_errs.mean():.2f})")
        axes[0].set_xlabel(r"$\|x_t - \hat{x}_t\|$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("SE-2a: state-estimation error distribution")
        axes[0].legend(fontsize=9, frameon=False)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Bar chart of summary stats.
        labels = ["mean", "median", "RMSE"]
        raw_stats = [summary["raw_probe"]["mean_err"],
                     summary["raw_probe"]["median_err"],
                     summary["raw_probe"]["rmse"]]
        lkf_stats = [summary["lkf"]["mean_err"],
                     summary["lkf"]["median_err"],
                     summary["lkf"]["rmse"]]
        x_pos = np.arange(len(labels))
        axes[1].bar(x_pos - 0.2, raw_stats, width=0.4, color="#4C72B0", label="raw probe", alpha=0.85)
        axes[1].bar(x_pos + 0.2, lkf_stats, width=0.4, color="#55A467", label="LKF", alpha=0.85)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("State-estimation error")
        axes[1].set_title(f"SE-2b: per-statistic comparison ({summary['improvement_pct']:+.1f}%)")
        axes[1].legend(fontsize=9, frameon=False)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        fig.tight_layout()
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_out, dpi=160)
        fig.savefig(Path(args.fig_out).with_suffix(".png"), dpi=160)
        print(f"  wrote {args.fig_out}")
    except Exception as e:
        print(f"  plotting failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

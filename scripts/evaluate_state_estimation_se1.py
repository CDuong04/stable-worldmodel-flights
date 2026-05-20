"""SE-1: Probe error variance predicted by P_lyap eigenvalues.

Tests whether the per-direction probe error (||d_phi(z) - x|| projected onto
eigenvectors of P_lyap) correlates with the corresponding eigenvalues.  A
positive correlation justifies using P_lyap as a calibration matrix for the
probe: slow-contracting directions (large eigenvalue) are less reliable.

This is post-hoc analysis on the existing rocket_lejepa world model + decoder
+ spectral artifact.  Compute cost: ~3-5 min on GPU.

Outputs:
  results/se1_probe_calibration.json
    - eigenvalues, per-eigvec MSE, fit slope, R^2
  results/paper_figures/fig_se1_probe_calibration.pdf
    - scatter of (eigenvalue, MSE) with linear fit
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

# Pickle compat
try:
    import importlib as _importlib
    _lejepa_mod = _importlib.import_module("lejepa_wm")
    for _name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
        if hasattr(_lejepa_mod, _name):
            setattr(sys.modules["__main__"], _name, getattr(_lejepa_mod, _name))
except Exception as e:
    print(f"[warn] {e}")

import gymnasium as gym
from evaluate_rocket_clf import (
    ProbeStateAdapter, _extract_probe_from_wm, _first_image, _to_chw_time_batch,
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
    ap.add_argument("--output", default="results/se1_probe_calibration.json")
    ap.add_argument("--fig-out", default="results/paper_figures/fig_se1_probe_calibration.pdf")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- Load model + decoder + spectral artifact ----
    base_model = AutoCostModel(args.checkpoint).to(device).eval()
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size
    decoder = _extract_probe_from_wm(base_model, embed_dim, 17, "mlp_2", device)
    p_mean, p_std = compute_proprio_stats(args.norm_dataset)
    decoder = ProbeStateAdapter(decoder, mean=p_mean, std=p_std, output_space="normalized").to(device).eval()

    artifact = torch.load(args.spectral_artifact, map_location=device, weights_only=False)
    P_lyap = artifact["P_lyap"].to(device).float()  # (d, d)
    z_star = artifact["z_star"].to(device).float()

    # Eigendecomposition of P_lyap: P = U diag(L) U^T.
    eig_vals, eig_vecs = torch.linalg.eigh(P_lyap.cpu())
    eig_vals = eig_vals.numpy()                   # (d,)
    eig_vecs = eig_vecs.numpy()                   # (d, d)
    d = eig_vals.shape[0]
    print(f"[se1] P_lyap eig range [{eig_vals.min():.3f}, {eig_vals.max():.3f}], d={d}")

    # ---- Collect (latent residual, probe error) pairs ----
    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=1, image_shape=(224, 224),
        history_size=3, frame_skip=2,
        max_episode_steps=1000, render_mode="rgb_array",
    )

    all_v = []          # latent residual v = z - z_star, (N, d)
    all_x = []          # ground truth proprio,  (N, 17)
    all_x_hat = []      # probe(z),              (N, 17)

    rng = np.random.default_rng(42)
    for ep in range(args.num_episodes):
        seed = 42 + ep
        obs, info = world.envs.reset(seed=seed, options=reset_options_for_level(args.level))
        for t in range(args.steps_per_episode):
            # Encode current pixels.
            z = encode_pixels(base_model, _first_image(info), device)
            # Pool patches.
            z_p = z.mean(dim=-2) if z.dim() >= 3 else z
            v = (z_p.reshape(-1) - z_star.reshape(-1)).detach().cpu().numpy()  # (d,)
            # Ground-truth proprio (raw, 17-D).
            proprio = info.get("proprio", info.get("observation"))
            x = np.asarray(proprio).reshape(-1)[:17].astype(np.float32)
            # Probe estimate from latent.
            with torch.no_grad():
                x_hat = decoder(z_p.reshape(1, -1))[0].detach().cpu().numpy().astype(np.float32)
            all_v.append(v)
            all_x.append(x)
            all_x_hat.append(x_hat)

            # Step with random action to traverse the state space.
            action = rng.uniform(world.envs.action_space.low, world.envs.action_space.high)
            obs, _, terminated, truncated, info = world.envs.step(action)
            if (hasattr(terminated, "__len__") and bool(terminated[0])) or \
               (hasattr(truncated, "__len__") and bool(truncated[0])):
                break
        if ep % 5 == 0:
            print(f"[se1] ep {ep}/{args.num_episodes}: collected {len(all_v)} samples")

    V = np.stack(all_v, axis=0)                # (N, d)
    X = np.stack(all_x, axis=0)                # (N, 17)
    Xh = np.stack(all_x_hat, axis=0)           # (N, 17)
    N = V.shape[0]
    print(f"[se1] collected N={N} samples")

    # Per-direction probe error: project the residual ||x - x_hat||^2 doesn't
    # directly factorize into latent eigenvectors.  Instead, we look at the
    # *latent* residual variance per eigenvector and the *state* error magnitude:
    #   - latent variance per eigvec: var(U^T v)_k for k=1..d
    #   - per-sample state error: ||x - x_hat||
    # Hypothesis: samples with large projection onto eigvec k (proxy for v far
    # along k-th eigendirection) have larger state error.  Aggregate by binning
    # along each eigvec and computing mean state error per bin.

    # Simpler: compute variance of (U^T v) per eigvec, and Pearson correlation
    # between |U^T v|_k and ||x - x_hat|| across samples.
    Vproj = V @ eig_vecs                        # (N, d)
    err = np.linalg.norm(X - Xh, axis=1)        # (N,)

    # Per-eigvec correlation between |projection| and state error.
    correlations = np.zeros(d)
    for k in range(d):
        corr_matrix = np.corrcoef(np.abs(Vproj[:, k]), err)
        if np.isfinite(corr_matrix[0, 1]):
            correlations[k] = corr_matrix[0, 1]

    # Aggregate: for each eigvec k, the variance of the latent projection is
    # var(Vproj[:, k]).  Per-eigvec MSE proxy: average squared projection * average squared err.
    proj_variance = np.var(Vproj, axis=0)         # (d,)

    # The key plot: scatter (eigenvalue, mean abs projection error contribution).
    # mean_proj_err[k] = E[ |Vproj[:, k]| * err ] - a measure of "how often does direction k coincide with high state error"
    mean_proj_err = np.mean(np.abs(Vproj) * err.reshape(-1, 1), axis=0)  # (d,)

    # Pearson correlation between eig_vals and mean_proj_err.
    rho_lambda_err = float(np.corrcoef(eig_vals, mean_proj_err)[0, 1])
    rho_lambda_var = float(np.corrcoef(eig_vals, proj_variance)[0, 1])

    # Linear fit: mean_proj_err = a + b * eig_vals.
    coef = np.polyfit(eig_vals, mean_proj_err, deg=1)
    slope = float(coef[0])
    intercept = float(coef[1])
    fit_vals = slope * eig_vals + intercept
    ss_res = float(np.sum((mean_proj_err - fit_vals) ** 2))
    ss_tot = float(np.sum((mean_proj_err - np.mean(mean_proj_err)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print()
    print(f"[se1] eigenvalues range:        [{eig_vals.min():.3f}, {eig_vals.max():.3f}]")
    print(f"[se1] mean state error:          {err.mean():.3f}")
    print(f"[se1] corr(eig_val, mean_err):   {rho_lambda_err:+.3f}")
    print(f"[se1] corr(eig_val, var(proj)):  {rho_lambda_var:+.3f}")
    print(f"[se1] linear fit slope:          {slope:+.4f} (R^2 = {r_squared:.3f})")

    out = {
        "n_samples": N,
        "n_eigvecs": int(d),
        "eigenvalue_min": float(eig_vals.min()),
        "eigenvalue_max": float(eig_vals.max()),
        "mean_state_error": float(err.mean()),
        "corr_eigval_meanerr": rho_lambda_err,
        "corr_eigval_projvar": rho_lambda_var,
        "linear_fit_slope": slope,
        "linear_fit_intercept": intercept,
        "linear_fit_r_squared": r_squared,
        "eigenvalues": eig_vals.tolist(),
        "mean_proj_err_per_eigvec": mean_proj_err.tolist(),
        "proj_variance_per_eigvec": proj_variance.tolist(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[se1] wrote {args.output}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.scatter(eig_vals, mean_proj_err, s=12, alpha=0.5, color="#4C72B0")
        ax.plot(eig_vals, fit_vals, color="black", linewidth=1.0,
                label=f"linear fit (slope={slope:+.4f}, $R^2$={r_squared:.3f})")
        ax.set_xlabel(r"P\textsubscript{lyap} eigenvalue $\lambda_k$")
        ax.set_ylabel(r"$\mathbb{E}[\,|v_k|\cdot\|x-\hat{x}\|\,]$")
        ax.set_title("SE-1: probe error vs P_lyap eigenvalue direction")
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_out, dpi=160)
        fig.savefig(Path(args.fig_out).with_suffix(".png"), dpi=160)
        print(f"[se1] wrote figure {args.fig_out}")
    except Exception as e:
        print(f"[se1] plotting failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

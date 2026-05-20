"""Offline fit of the local linear model A, B at a touchdown equilibrium.

Loads a trained DINOWM checkpoint, encodes a reference "landed pad" image to
get the equilibrium latent z_*, computes:

  A = ∂f_θ/∂z(z_*, a_*)                 (d, d) latent-Jacobian via autograd
  B = ∂f_θ/∂a(z_*, a_*)                 (d, m) action-Jacobian via autograd
  spectral decomposition of A           via stable_worldmodel.spectral
  P_lyap                                discrete Lyapunov on E^s

Saves everything to a single .pt file consumed by
stable_worldmodel.stable_mpc.StableManifoldConstrainedCost.

Usage (always via srun on OSCAR — no Python on login node):

  srun --pty --gres=gpu:1 --time=00:30:00 python scripts/fit_local_linear_model.py \\
      --checkpoint /oscar/scratch/aiyer40/rocket_lejepa_wm_union_vhead_rolloutprobe \\
      --reference-image data/landed_pad_reference.png \\
      --action-dim 7 \\
      --output models/local_linear_model.pt

The pooled latent convention follows clf_cost.CLFAugmentedCost._pooled_latents:
the (B, T, P, d) DINOWM embedding is mean-pooled over the patch dim to give
(B, T, d).  The Jacobian is computed at the "all-patches-equal-to-z_*" point.

Diagnostics printed at the end:
  - eigenvalue magnitudes histogram (sanity check on what the predictor learned)
  - hyperbolicity gap (target > 0.05)
  - n_s / d fraction (target > 0.85)
  - condition number of P_lyap (target < 1e6)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repo root on sys.path for stable_worldmodel imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stable_worldmodel.spectral import (
    decompose_stable_unstable,
    solve_discrete_lyapunov_on_stable_subspace,
)


def _load_image_tensor(path: Path, size: int = 224) -> torch.Tensor:
    """Load an RGB image into a (3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _encode_reference(
    dinowm,
    image: torch.Tensor,
    device: str,
    proprio_dim: int,
    action_dim: int,
) -> tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """Run dinowm.encode on a single image and return the pooled latent z_*.

    The predictor was trained on a concatenated embedding (pixel + proprio + action).
    To match dimensions at inference time, we feed placeholder proprio and action
    tensors (zeros) to the encoder.  At the equilibrium these are exactly correct:
    proprio is the 17-D landed state (zeros for target-relative position when on pad),
    action is the idle vector.
    """
    pixels = image.unsqueeze(0).unsqueeze(0).to(device)               # (1, 1, 3, H, W)
    proprio = torch.zeros(1, 1, proprio_dim, device=device)            # (1, 1, P_dim)
    action  = torch.zeros(1, 1, action_dim, device=device)             # (1, 1, A_dim)
    info = {"pixels": pixels, "proprio": proprio, "action": action}
    info = dinowm.encode(
        info, pixels_key="pixels", target="embed",
        proprio_key="proprio", action_key="action",
    )
    embed = info["embed"]                                              # (1, 1, P, D)
    assert embed.dim() == 4 and embed.shape[:2] == (1, 1), (
        f"Unexpected embed shape after encode: {tuple(embed.shape)}"
    )
    action_dim_emb = info.get("action_embed", torch.zeros(0)).shape[-1] if "action_embed" in info else 0
    proprio_dim_emb = info.get("proprio_embed", torch.zeros(0)).shape[-1] if "proprio_embed" in info else 0
    pixel_dim = embed.shape[-1] - action_dim_emb - proprio_dim_emb
    pixel_embed = embed[..., :pixel_dim]                                # (1, 1, P, pixel_dim)
    z_star_pooled = pixel_embed.mean(dim=2).squeeze(0).squeeze(0)       # (pixel_dim,)
    return (
        z_star_pooled.detach(),
        int(action_dim_emb),
        int(proprio_dim_emb),
        int(pixel_dim),
        embed,
    )


def _jacobian_at_equilibrium(
    dinowm,
    embed_ref: torch.Tensor,
    z_star: torch.Tensor,
    a_star: torch.Tensor,
    action_dim_emb: int,
    pixel_dim: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute A = ∂f/∂z and B = ∂f/∂a at (z_*, a_*) via autograd.

    f(z_pool, a) := pool(predict(broadcast(z_pool, action_encoder(a))))
        broadcasting: every patch in (1, hist, P, pixel_dim) := z_pool;
        action embed appended to dim;
        proprio embed kept as in embed_ref (zeros if absent).
    """
    embed_ref = embed_ref.to(device)
    B_, T, P, D = embed_ref.shape
    proprio_dim_emb = D - pixel_dim - action_dim_emb

    history_size = getattr(dinowm, "history_size", 1)
    # Build the "frozen" non-pixel chunks (proprio + action embed) from the
    # reference encode; for the equilibrium we will overwrite the action embed
    # but keep the proprio embed.
    proprio_chunk = (
        embed_ref[..., pixel_dim:pixel_dim + proprio_dim_emb].detach()
        if proprio_dim_emb > 0 else None
    )

    def f(z_pool: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """One-step pooled prediction at z_*, a_*."""
        # broadcast pooled latent to every patch and every history step
        z_patches = z_pool.view(1, 1, 1, pixel_dim).expand(1, history_size, P, pixel_dim)
        chunks = [z_patches]
        if proprio_dim_emb > 0:
            # Reuse the reference's proprio embed (assumed valid at equilibrium)
            proprio_replicated = proprio_chunk.expand(1, history_size, P, proprio_dim_emb).to(device)
            chunks.append(proprio_replicated)
        if action_dim_emb > 0:
            # action_encoder takes (1, T=1, action_dim) → (1, 1, action_emb_dim)
            a_emb = dinowm.action_encoder(a.view(1, 1, -1))  # (1, 1, action_emb_dim)
            a_patches = a_emb.unsqueeze(2).expand(1, history_size, P, action_dim_emb)
            chunks.append(a_patches)
        embedding = torch.cat(chunks, dim=-1)              # (1, hist, P, D)
        predicted = dinowm.predict(embedding)               # (1, hist, P, D)
        # Take the last history step's predicted pixel embed; pool over patches.
        pixel_pred = predicted[:, -1:, :, :pixel_dim]       # (1, 1, P, pixel_dim)
        pooled = pixel_pred.mean(dim=2).squeeze(0).squeeze(0)  # (pixel_dim,)
        return pooled

    # Compute Jacobians.  Avoid functorch path issues: use a wrapper that
    # accepts each input separately, then ask jacobian for both.
    A = torch.autograd.functional.jacobian(
        lambda z: f(z, a_star), z_star, vectorize=True, create_graph=False,
    )
    B = torch.autograd.functional.jacobian(
        lambda a: f(z_star, a), a_star, vectorize=True, create_graph=False,
    )
    return A.detach(), B.detach()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit local linear model at the touchdown equilibrium.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Directory containing the DINOWM checkpoint.")
    parser.add_argument("--reference-image", type=str, required=True,
                        help="Path to landed-pad reference image (PNG/JPG).")
    parser.add_argument("--action-dim", type=int, required=True,
                        help="Physical action dim (7 for rocket).")
    parser.add_argument("--action-block", type=int, default=2,
                        help="Frame-skip action stacking (2 for default training).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file (caller may want models/ subdir).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gap-tol", type=float, default=0.05,
                        help="Hyperbolicity gap below which to warn.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[fit] device={device}")

    # ------------------------------------------------------------
    # Load DINOWM via the same utilities the eval harness uses.  The training
    # pickle references __main__.ObsProbe so inject the symbol before load
    # (mirrors evaluate_rocket_clf.py:472-486).
    # ------------------------------------------------------------
    import importlib as _importlib
    _train_dir = str(REPO_ROOT / "scripts" / "train")
    if _train_dir not in sys.path:
        sys.path.insert(0, _train_dir)
    try:
        _lejepa_mod = _importlib.import_module("lejepa_wm")
        for _name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
            if hasattr(_lejepa_mod, _name):
                setattr(sys.modules["__main__"], _name, getattr(_lejepa_mod, _name))
    except Exception as _e:
        print(f"[warn] could not preload ObsProbe symbol: {_e}")

    from stable_worldmodel.policy import AutoCostModel
    print(f"[fit] loading checkpoint from {args.checkpoint}")
    auto = AutoCostModel(args.checkpoint).to(device).eval()
    # The cost wrapper unwraps to the underlying DINOWM via the world_model attr.
    dinowm = getattr(auto, "world_model", auto)
    print(f"[fit] dinowm class: {type(dinowm).__name__}")
    print(f"[fit] history_size: {getattr(dinowm, 'history_size', '?')}")

    # ------------------------------------------------------------
    # Encode the reference image to get z_*.
    # ------------------------------------------------------------
    img = _load_image_tensor(Path(args.reference_image), size=args.image_size)
    print(f"[fit] reference image: shape={tuple(img.shape)}")

    PROPRIO_DIM = 17
    action_input_dim = args.action_dim * args.action_block   # Conv1d-stacked actions
    with torch.no_grad():
        z_star, action_dim_emb, proprio_dim_emb, pixel_dim, embed_ref = _encode_reference(
            dinowm, img, device,
            proprio_dim=PROPRIO_DIM, action_dim=action_input_dim,
        )
    print(f"[fit] pixel_dim={pixel_dim}, proprio_dim_emb={proprio_dim_emb}, "
          f"action_dim_emb={action_dim_emb}, total D={embed_ref.shape[-1]}")
    print(f"[fit] z_star norm: {float(z_star.norm()):.4f}")

    # ------------------------------------------------------------
    # Pick the equilibrium action a_*.
    # ------------------------------------------------------------
    # Rocket idle: throttle=0 (no burn), gimbal=0, finlets=neutral.  Caller can
    # override by editing this script if the action layout differs.  For now,
    # use a zero vector of size (action_dim * action_block) — the Conv1d
    # action encoder consumes stacked actions, so the input dim is widened.
    a_star = torch.zeros(action_input_dim, device=device)
    print(f"[fit] a_star (idle, size {action_input_dim}): {a_star.cpu().numpy()}")

    # ------------------------------------------------------------
    # Compute A, B.  Both via autograd.functional.jacobian; CPU-friendly even
    # for a ~384-dim latent since we evaluate at a single point.
    # ------------------------------------------------------------
    print("[fit] computing A = ∂f/∂z and B = ∂f/∂a ...")
    A_torch, B_torch = _jacobian_at_equilibrium(
        dinowm,
        embed_ref=embed_ref.detach(),
        z_star=z_star.requires_grad_(False),
        a_star=a_star,
        action_dim_emb=action_dim_emb,
        pixel_dim=pixel_dim,
        device=device,
    )
    print(f"[fit] A: shape={tuple(A_torch.shape)}  ‖A‖_F={float(A_torch.norm()):.4f}")
    print(f"[fit] B: shape={tuple(B_torch.shape)}  ‖B‖_F={float(B_torch.norm()):.4f}")

    A_np = A_torch.cpu().double().numpy()
    B_np = B_torch.cpu().double().numpy()

    # ------------------------------------------------------------
    # Spectral decomposition + Lyapunov solve.
    # ------------------------------------------------------------
    print("[fit] decomposing A ...")
    decomp = decompose_stable_unstable(A_np, gap_tol=args.gap_tol)
    print(f"[fit] eigvalue magnitudes: min={float(np.min(np.abs(decomp.eigvals))):.4f}, "
          f"max={float(np.max(np.abs(decomp.eigvals))):.4f}")
    print(f"[fit] n_s={decomp.n_s}, n_u={decomp.n_u} (of d={A_np.shape[0]})")
    print(f"[fit] hyperbolicity gap: {decomp.gap:.4f} (target > {args.gap_tol})")
    print(f"[fit] hyperbolic: {decomp.hyperbolic}")
    print(f"[fit] spectral_radius_stable: {decomp.spectral_radius_stable:.4f}")
    if not decomp.hyperbolic:
        print("[fit][warn] gap below threshold — center-manifold formulation needed for the proof.")

    P_lyap_np = solve_discrete_lyapunov_on_stable_subspace(decomp)
    cond_P = float(np.linalg.cond(P_lyap_np + 1e-12 * np.eye(P_lyap_np.shape[0])))
    print(f"[fit] cond(P_lyap): {cond_P:.4e}")
    if cond_P > 1e6:
        print("[fit][warn] P_lyap is poorly conditioned.")

    # ------------------------------------------------------------
    # Save artifact.  Format must match StableManifoldConstrainedCost.__init__.
    # ------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "z_star": z_star.detach().cpu(),
        "a_star": a_star.detach().cpu(),
        "A": A_torch.detach().cpu(),
        "B": B_torch.detach().cpu(),
        "P_s": torch.from_numpy(decomp.P_s).float(),
        "P_u": torch.from_numpy(decomp.P_u).float(),
        "P_lyap": torch.from_numpy(P_lyap_np).float(),
        "V_s_basis": torch.from_numpy(decomp.V_s_basis).float(),
        "V_u_basis": torch.from_numpy(decomp.V_u_basis).float(),
        "A_s_basis": torch.from_numpy(decomp.A_s_basis).float(),
        "eigvals": torch.from_numpy(np.stack(
            [decomp.eigvals.real, decomp.eigvals.imag]
        ).astype(np.float32)),
        "n_s": decomp.n_s,
        "n_u": decomp.n_u,
        "gap": decomp.gap,
        "hyperbolic": decomp.hyperbolic,
        "spectral_radius_stable": decomp.spectral_radius_stable,
        "cond_P_lyap": cond_P,
        "checkpoint_path": args.checkpoint,
        "reference_image_path": args.reference_image,
    }
    torch.save(artifact, out_path)
    print(f"[fit] saved artifact to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

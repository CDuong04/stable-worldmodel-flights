"""Evaluate LeJEPA-WM on rocket landing with decoded Lyapunov monitoring.

Implements the eval pipeline described in rocket_jepa.tex:
  1. Encode current camera image  z_t  via the world model.
  2. Run CEM in latent space toward an encoded goal image.
  3. Execute the chosen action in the env, observe next image.
  4. Encode next image  z_{t+1}, decode both via state decoder, evaluate
     Lyapunov V and descent rate delta_t.
  5. Aggregate per-episode mean descent + violation rate.

Reports:
  - success_rate (terminateds via env)
  - mean_descent_rate, violation_rate (from LyapunovMonitor)
  - mean episode length, return
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401  (registers env)
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel, WorldModelPolicy, PlanConfig
from stable_worldmodel.lyapunov import LyapunovMonitor, LyapunovConfig
from scripts.train_state_decoder import StateDecoder


def load_decoder(path: str, embed_dim: int, state_dim: int = 17,
                 architecture: str = "mlp_2", device: str = "cpu") -> StateDecoder:
    decoder = StateDecoder(embed_dim=embed_dim, state_dim=state_dim, architecture=architecture)
    sd = torch.load(path, map_location=device)
    decoder.load_state_dict((sd.get("state_dict") or sd.get("decoder") or sd) if isinstance(sd, dict) else sd)
    return decoder.to(device).eval()


def encode_pixels(world_model, pixels: np.ndarray, device: str) -> torch.Tensor:
    """Encode (H, W, 3) uint8 pixels -> (embed_dim,) DINO patch-mean embedding."""
    if pixels.ndim == 3:
        pixels = pixels[None]
    x = torch.as_tensor(pixels, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    x = (x - 0.5) / 0.5
    info = {"pixels": x.unsqueeze(1)}  # (B, T=1, 3, H, W)
    with torch.no_grad():
        info = world_model.model.encode(info, target="embed", pixels_key="pixels")
    z = info["pixels_embed"].mean(dim=2).squeeze(1)  # average over patches -> (B, embed_dim)
    return z.squeeze(0)


def run_lejepa_eval(
    model_name: str,
    decoder_path: str,
    decoder_arch: str,
    horizon: int,
    episodes: int,
    seed: int,
    levels: list[str],
    num_samples: int = 200,
    n_steps: int = 20,
    device: str = "cuda",
    pad_pos=(0.0, 0.0, 0.0),
    alpha: float = 0.5,
    beta: float = 0.1,
):
    print(f"Loading world model: {model_name}")
    base_model = AutoCostModel(model_name)
    # base_model.backbone is wrapped with stable_pretraining's EvalOnly during
    # training; the underlying HF DINOv2 model (with .config) sits one level deeper.
    backbone = base_model.backbone
    if hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    embed_dim = backbone.config.hidden_size

    print(f"Loading decoder: {decoder_path} (arch={decoder_arch})")
    decoder = load_decoder(decoder_path, embed_dim=embed_dim, architecture=decoder_arch, device=device)

    solver = swm.solver.CEMSolver(
        model=base_model, num_samples=num_samples, var_scale=0.5,
        n_steps=n_steps, topk=max(num_samples // 10, 5), device=device,
    )
    plan_cfg = PlanConfig(horizon=horizon, receding_horizon=max(horizon // 3, 1),
                          history_len=1, action_block=2, warm_start=True)

    results = {}
    for level in levels:
        print(f"\n--- LeJEPA-WM + Lyapunov: {level} ---")
        world = swm.World(
            "swm/PFRocketLandingExt-v0",
            num_envs=1, image_shape=(224, 224),
            max_episode_steps=1200, render_mode="rgb_array",
        )
        policy = WorldModelPolicy(solver=solver, config=plan_cfg)
        world.set_policy(policy)

        ep_records = []
        for ep in range(episodes):
            mon = LyapunovMonitor(
                decoder=decoder, device=device,
                cfg=LyapunovConfig(alpha=alpha, beta=beta, pad_pos=tuple(pad_pos)),
            )
            obs, info = world.reset(seed=seed + ep, options={"perturbation_level": level})
            z_prev = encode_pixels(base_model.world_model, obs["pixels"][0], device)

            done = False; ep_return = 0.0; t = 0
            while not done and t < 1200:
                action = policy.get_action({"pixels": obs["pixels"], "goal": obs["goal"]})
                obs, reward, terminated, truncated, info = world.envs.step(action)
                z_curr = encode_pixels(base_model.world_model, obs["pixels"][0], device)
                mon.step(z_prev, z_curr)
                z_prev = z_curr
                ep_return += float(np.asarray(reward).sum())
                t += 1
                done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())

            summary = mon.summary()
            summary.update({"success": bool(np.asarray(terminated).any()), "return": ep_return, "T": t})
            ep_records.append(summary)
            print(f"  ep {ep}: success={summary['success']} return={ep_return:.1f} "
                  f"mean_descent={summary['mean_descent']:+.3f} viol={summary['violation_rate']:.2f}")

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
        print(f"[{level}] success={success*100:.1f}% mean_desc={agg['mean_descent_rate']:+.3f} "
              f"viol={agg['violation_rate']:.2f}")
        results[level] = agg
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="WM object name (without _object.ckpt)")
    p.add_argument("--decoder", required=True, help="State-decoder .pt path")
    p.add_argument("--decoder-arch", default="mlp_2", choices=["linear", "mlp_2", "mlp_3"])
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard", "extreme"])
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/lejepa_wm_lyapunov.json")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()

    results = run_lejepa_eval(
        args.model, args.decoder, args.decoder_arch, args.horizon, args.episodes,
        args.seed, args.levels, args.num_samples, args.n_steps, args.device,
        alpha=args.alpha, beta=args.beta,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

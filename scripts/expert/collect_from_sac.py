"""Collect expert trajectories using a pretrained SB3 SAC checkpoint.

This is the same pattern as scripts/expert/train_policies.py, but used purely
for rollout collection (no further training).  Loads the SAC policy + VecNormalize
stats and runs N rollouts, saving (pixels, proprio, action, reward) to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import stable_worldmodel  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["cartpole", "pendulum"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True, help="path to SB3 .zip")
    ap.add_argument("--vec-normalize", default=None, help="path to vec_normalize .pkl (optional)")
    ap.add_argument("--n-rollouts", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from stable_baselines3 import SAC

    print(f"[expert] loading SAC from {args.checkpoint}")
    model = SAC.load(args.checkpoint, device="cpu")

    env_id = f"swm/{args.domain.capitalize()}DMControl-v0"
    env = gym.make(env_id)
    print(f"[expert] env={env_id}")

    # Optionally load VecNormalize stats for obs normalization
    obs_normalizer = None
    if args.vec_normalize:
        from stable_worldmodel.envs.dmcontrol.expert_policy import SB3Normalizer
        obs_normalizer = SB3Normalizer(args.vec_normalize)
        print(f"[expert] vec_normalize loaded")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_success = 0
    total_reward = 0.0

    for r in range(args.n_rollouts):
        obs, _info = env.reset(seed=args.seed + r)
        pixels, proprios, actions, rewards = [], [], [], []
        for t in range(args.max_steps):
            try:
                img = env.render()
            except Exception:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            pixels.append(np.asarray(img, dtype=np.uint8))
            proprios.append(np.asarray(obs, dtype=np.float32))

            obs_for_policy = obs_normalizer(obs) if obs_normalizer is not None else obs
            a, _ = model.predict(obs_for_policy, deterministic=True)
            if args.noise > 0:
                a = a + rng.normal(0, args.noise, size=a.shape).astype(np.float32)
                a = np.clip(a, env.action_space.low, env.action_space.high)
            actions.append(a.copy())
            obs, reward, terminated, truncated, _info = env.step(a)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        if not pixels:
            continue
        pixels = np.stack(pixels, axis=0)
        proprios = np.stack(proprios, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.asarray(rewards, dtype=np.float32)

        ep_return = float(rewards.sum())
        total_reward += ep_return
        if ep_return > 100:  # reasonable swing-up reward
            n_success += 1

        np.savez_compressed(
            out / f"rollout_{r:04d}.npz",
            pixels=pixels, proprio=proprios, action=actions, reward=rewards,
        )
        if (r + 1) % 20 == 0:
            print(f"[expert] {r+1}/{args.n_rollouts} rollouts: mean_return={total_reward/(r+1):.1f}, "
                  f"successes={n_success}")

    env.close()
    print(f"[expert] done. mean_return={total_reward/args.n_rollouts:.1f}, "
          f"{n_success}/{args.n_rollouts} successful")


if __name__ == "__main__":
    main()

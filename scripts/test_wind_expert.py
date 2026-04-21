"""Smoke test: does the PDG expert still land under OU wind disturbance?

For each wind preset, runs N episodes with the convex SOCP expert and reports
success rate + mean episode length + wind statistics. This is the baseline
number used in the paper's Table B.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401  (registers env)
import stable_worldmodel as swm
from stable_worldmodel.envs.pdg_expert import PDGExpertPolicy


def run_wind_sweep(episodes: int, seed: int, wind_levels: list[str]):
    results = {}
    for level in wind_levels:
        print(f"\n--- wind_level = {level!r} ---")
        world = swm.World(
            "swm/PFRocketLandingExt-v0",
            num_envs=1,
            image_shape=(224, 224),
            max_episode_steps=1200,
            render_mode="rgb_array",
        )
        policy = PDGExpertPolicy()
        world.set_policy(policy)

        options = {}
        if level != "none":
            options["wind_level"] = level

        t0 = time.time()
        metrics = world.evaluate(
            episodes=episodes,
            seed=seed,
            options=options,
            eval_keys=["reward_total"] if False else None,
        )
        elapsed = time.time() - t0
        success = float(metrics["success_rate"])
        print(f"  success_rate = {success*100:.1f}%  ({episodes} eps, {elapsed:.1f}s)")
        results[level] = {
            "success_rate": success,
            "n_episodes": int(episodes),
            "elapsed_s": float(elapsed),
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels", nargs="+",
                   default=["none", "calm", "light", "gust", "storm"])
    p.add_argument("--out", default="results/wind_expert_sweep.json")
    args = p.parse_args()

    results = run_wind_sweep(args.episodes, args.seed, args.levels)

    print("\n" + "=" * 54)
    print(f"{'wind':>10} | {'success rate':>14} | {'n':>5} | {'elapsed':>8}")
    print("-" * 54)
    for level, r in results.items():
        print(f"{level:>10} | {r['success_rate']*100:>13.1f}% | "
              f"{r['n_episodes']:>5} | {r['elapsed_s']:>7.1f}s")
    print("=" * 54)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

"""Collect expert trajectories for rocket landing world model training.

Collects data across all perturbation levels (easy/medium/hard/extreme)
and optionally with moving platform, using the PDG expert policy.

Usage:
    # On interactive GPU node:
    srun --partition=gpu --gres=gpu:1 --mem=32G --time=6:00:00 --pty bash
    source .venv_rocket/bin/activate

    # Collect all conditions:
    python scripts/collect_rocket_data.py

    # Collect specific condition:
    python scripts/collect_rocket_data.py --levels medium hard --episodes 200

    # With moving pad:
    python scripts/collect_rocket_data.py --levels medium --moving-pad --episodes 200
"""

import argparse
import sys
from pathlib import Path

# Register extended env before importing World
import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401 (registers swm/PFRocketLandingExt-v0)
import stable_worldmodel as swm
from stable_worldmodel.envs.pdg_expert import PDGExpertPolicy


def collect_level(
    level: str,
    episodes: int,
    seed: int,
    moving_pad: bool = False,
    num_envs: int = 2,
    pad_sigma_xy: float = 1.0,
):
    """Collect expert data for one perturbation level."""
    suffix = f"_movingpad" if moving_pad else ""
    dataset_name = f"rocket_expert_{level}{suffix}"

    print(f"\n{'='*60}")
    print(f"Collecting {episodes} episodes | level={level} | moving_pad={moving_pad}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    world = swm.World(
        "swm/PFRocketLandingExt-v0",
        num_envs=num_envs,
        image_shape=(224, 224),
        max_episode_steps=1200,  # 30s at 40Hz
        render_mode="rgb_array",
    )

    policy = PDGExpertPolicy()
    world.set_policy(policy)

    reset_options = {
        "perturbation_level": level,
        "moving_pad": moving_pad,
        "pad_sigma_xy": pad_sigma_xy,
        "variation": ["environment"],
    }

    world.record_dataset(
        dataset_name,
        episodes=episodes,
        seed=seed,
        options=reset_options,
    )

    print(f"Done: {dataset_name}")

    # Quick evaluation
    results = world.evaluate(episodes=min(20, episodes), seed=seed + 10000)
    print(f"  Success rate: {results['success_rate']:.1f}%")

    return dataset_name


def main():
    parser = argparse.ArgumentParser(description="Collect rocket expert data")
    parser.add_argument("--levels", nargs="+", default=["easy", "medium", "hard", "extreme"],
                        help="Perturbation levels to collect")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--num-envs", type=int, default=2,
                        help="Number of parallel environments")
    parser.add_argument("--moving-pad", action="store_true",
                        help="Enable moving landing pad")
    parser.add_argument("--pad-sigma", type=float, default=1.0,
                        help="Moving pad OU sigma_xy")
    args = parser.parse_args()

    collected = []
    for i, level in enumerate(args.levels):
        seed = args.seed + i * 10000
        name = collect_level(
            level=level,
            episodes=args.episodes,
            seed=seed,
            moving_pad=args.moving_pad,
            num_envs=args.num_envs,
            pad_sigma_xy=args.pad_sigma,
        )
        collected.append(name)

    print(f"\n{'='*60}")
    print("Collection complete. Datasets:")
    for name in collected:
        print(f"  - {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Orchestrate all ablation studies for RocketJEPA paper.

Runs the 8 ablation studies by invoking evaluate_rocket.py with different configs.

Usage:
    python scripts/run_ablations.py --ablation all
    python scripts/run_ablations.py --ablation constraints
    python scripts/run_ablations.py --ablation planner
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ABLATIONS = {
    # Ablation 1: Expert quality (data source)
    "expert_quality": [
        {"mode": "expert", "levels": ["easy", "medium", "hard", "extreme"]},
    ],

    # Ablation 2: Representation (pixels-only vs state-only vs pixels+state)
    # Requires separate training runs with different configs
    "representation": [
        {"mode": "wm", "model": "rocket_wm_pixels_only", "solver": "cem", "constraints": "physics"},
        {"mode": "wm", "model": "rocket_wm_state_only", "solver": "cem", "constraints": "physics"},
        {"mode": "wm", "model": "rocket_wm_pixels_state", "solver": "cem", "constraints": "physics"},
    ],

    # Ablation 3: Prediction horizon (open-loop)
    # This requires a separate script that measures latent MSE at different horizons
    "prediction_horizon": [
        # Placeholder: this ablation is evaluated offline, not via closed-loop
    ],

    # Ablation 4: Planner comparison
    "planner": [
        {"mode": "wm", "solver": "cem", "horizon": 5},
        {"mode": "wm", "solver": "cem", "horizon": 10},
        {"mode": "wm", "solver": "cem", "horizon": 15},
        {"mode": "wm", "solver": "cem", "horizon": 20},
        {"mode": "wm", "solver": "mppi", "horizon": 5},
        {"mode": "wm", "solver": "mppi", "horizon": 10},
        {"mode": "wm", "solver": "mppi", "horizon": 15},
        {"mode": "wm", "solver": "mppi", "horizon": 20},
    ],

    # Ablation 5: Constraint mode
    "constraints": [
        {"mode": "wm", "constraints": "none"},
        {"mode": "wm", "constraints": "action"},
        {"mode": "wm", "constraints": "physics", "lambda_constraint": 0.5},
        {"mode": "wm", "constraints": "physics", "lambda_constraint": 1.0},
        {"mode": "wm", "constraints": "physics", "lambda_constraint": 2.0},
        {"mode": "wm", "constraints": "physics_only"},
    ],

    # Ablation 6: Perturbation robustness
    "robustness": [
        {"mode": "wm", "levels": ["easy", "medium", "hard", "extreme"]},
        {"mode": "wm", "levels": ["medium", "hard"], "moving_pad": True},
    ],

    # Ablation 7: Data quantity
    # Requires separate training runs with subsets of data
    "data_quantity": [
        {"mode": "wm", "model": "rocket_wm_100ep"},
        {"mode": "wm", "model": "rocket_wm_250ep"},
        {"mode": "wm", "model": "rocket_wm_500ep"},
        {"mode": "wm", "model": "rocket_wm_1000ep"},
        {"mode": "wm", "model": "rocket_wm_2000ep"},
    ],

    # Ablation 8: Encoder comparison
    "encoder": [
        {"mode": "wm", "model": "rocket_wm_dino_frozen"},
        {"mode": "wm", "model": "rocket_wm_random_frozen"},
        {"mode": "wm", "model": "rocket_wm_dino_finetuned"},
    ],
}


def build_cmd(config: dict, base_args: dict) -> list[str]:
    """Build evaluate_rocket.py command from config dict."""
    cmd = [sys.executable, "scripts/evaluate_rocket.py"]

    merged = {**base_args, **config}

    for key, val in merged.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif isinstance(val, list):
            cmd.extend([flag] + [str(v) for v in val])
        else:
            cmd.extend([flag, str(val)])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablation", default="all",
                        choices=list(ABLATIONS.keys()) + ["all"])
    parser.add_argument("--model", type=str, default="rocket_world_model_object_epoch_200",
                        help="Default world model name")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="ablation_results")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    ablations_to_run = ABLATIONS if args.ablation == "all" else {args.ablation: ABLATIONS[args.ablation]}

    base_args = {
        "episodes": args.episodes,
        "seed": args.seed,
        "device": args.device,
    }

    for abl_name, configs in ablations_to_run.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {abl_name}")
        print(f"{'='*60}")

        for i, config in enumerate(configs):
            # Fill in default model if not specified
            if config.get("mode") == "wm" and "model" not in config:
                config["model"] = args.model

            output_file = output_dir / f"{abl_name}_{i}.json"
            config["output"] = str(output_file)

            cmd = build_cmd(config, base_args)
            print(f"  [{i}] {' '.join(cmd)}")

            if not args.dry_run:
                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    print(f"  WARNING: command returned {result.returncode}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()

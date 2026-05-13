"""Compare baseline PDG against the 4 robust variants on each disturbance bucket.

Runs N_EPS episodes per (variant x bucket) cell, prints landing rate, mean
final position error, mean final velocity. Uses the Ext env so disturbances
are real.
"""
import argparse
from collections import defaultdict

import numpy as np
import gymnasium as gym

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401  (registers env)
from stable_worldmodel.envs.pdg_expert import PDGController, parse_obs
from stable_worldmodel.envs.pdg_expert_robust import (
    OracleWindPDG, ESOWindPDG, L1AdaptivePDG, SCPLikePDG,
)


def run_one(env_id, expert_factory, expert_kwargs, opts, seed, max_steps=1200):
    """Run one episode, return (landed, final_pos_err, final_vel_norm, steps)."""
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed, options=opts)
    expert = expert_factory(**expert_kwargs)
    expert.reset()

    done = False
    steps = 0
    last_state = parse_obs(obs)
    while not done and steps < max_steps:
        state = parse_obs(obs)
        try:
            action = expert.compute_control(state, env=env)
        except TypeError:
            action = expert.compute_control(state)
        action = np.asarray(action, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_state = state
        steps += 1
    env.close()

    landed = bool(info.get("env_complete"))
    pos = last_state["position"]
    pad = np.zeros(3)
    pos_err = float(np.linalg.norm(pos[:2] - pad[:2]))
    vel_norm = float(np.linalg.norm(last_state["velocity"]))
    return landed, pos_err, vel_norm, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eps", type=int, default=10)
    parser.add_argument("--buckets", nargs="+",
                        default=["nominal", "calm", "light", "gust", "movpad"])
    parser.add_argument("--variants", nargs="+",
                        default=["pdg", "oracle_wind", "eso", "l1", "scp"])
    args = parser.parse_args()

    factories = {
        "pdg":         (PDGController, {}),
        "oracle_wind": (OracleWindPDG, {}),
        "eso":         (ESOWindPDG, {}),
        "l1":          (L1AdaptivePDG, {}),
        "scp":         (SCPLikePDG, {}),
    }
    bucket_opts = {
        "nominal": {},
        "calm":    {"wind_level": "calm"},
        "light":   {"wind_level": "light"},
        "gust":    {"wind_level": "gust"},
        "movpad":  {"moving_pad": True},
    }

    print(f"{'variant':<14} {'bucket':<10} {'land/N':>8} {'rate':>6} "
          f"{'pos_err_m':>10} {'vel_m/s':>9} {'steps':>6}")
    print("-" * 72)
    rows = []
    for v in args.variants:
        if v not in factories:
            print(f"  unknown variant {v}, skipping")
            continue
        cls, kw = factories[v]
        for b in args.buckets:
            opts = bucket_opts.get(b, {})
            landings = 0
            pos_errs, vel_norms, step_counts = [], [], []
            for ep in range(args.n_eps):
                seed = 9000 + 100 * args.buckets.index(b) + ep
                landed, pe, vn, st = run_one(
                    "swm/PFRocketLandingExt-v0", cls, kw, opts, seed,
                )
                landings += int(landed)
                pos_errs.append(pe)
                vel_norms.append(vn)
                step_counts.append(st)
            rate = landings / max(args.n_eps, 1)
            row = (v, b, f"{landings}/{args.n_eps}", f"{rate:.0%}",
                   f"{np.mean(pos_errs):.2f}", f"{np.mean(vel_norms):.2f}",
                   f"{int(np.mean(step_counts))}")
            rows.append(row)
            print(f"{row[0]:<14} {row[1]:<10} {row[2]:>8} {row[3]:>6} "
                  f"{row[4]:>10} {row[5]:>9} {row[6]:>6}")
    print("-" * 72)


if __name__ == "__main__":
    main()

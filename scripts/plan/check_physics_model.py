"""Validate RocketPhysicsModel against the dataset's expert trajectory.

If physics.rollout(proprio_0, expert_actions) ≈ expert_proprio_trajectory,
the physics model is correct. Otherwise CEM is optimizing on a model that
disagrees with reality, and the MPC can never work.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from stable_worldmodel.solver.rocket_physics import RocketPhysicsModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/oscar/data/jpober/cduong5/swm_cache/datasets/rocket_expert.h5")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=50)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        ep_offset = int(f["ep_offset"][args.episode])
        ep_len = int(f["ep_len"][args.episode])
        n = min(args.n_steps, ep_len - 1)
        actions = f["action"][ep_offset : ep_offset + n]           # (n, 7)
        proprio = f["proprio"][ep_offset : ep_offset + n + 1]      # (n+1, 17)

    # Replace NaN actions (sequence-boundary markers) with zeros
    actions = np.nan_to_num(actions, nan=0.0)

    # Run physics model
    physics = RocketPhysicsModel()
    state_0 = torch.from_numpy(proprio[0:1]).float()                # (1, 17)
    actions_t = torch.from_numpy(actions[None]).float()             # (1, n, 7)
    traj = physics.rollout(state_0, actions_t)[0].numpy()           # (n+1, 17)

    # Compare
    print(f"episode {args.episode} length {ep_len}, comparing first {n+1} steps")
    print(f"{'step':>5}  {'pos_err(m)':>12}  {'vel_err(m/s)':>14}  {'angvel_err':>12}  {'fuel_real':>10}  {'fuel_phys':>10}")
    for t in [0, 1, 5, 10, 25, min(n, 49)]:
        if t > n:
            continue
        r = proprio[t]
        p = traj[t]
        pos_err = np.linalg.norm(r[0:3] - p[0:3])
        vel_err = np.linalg.norm(r[3:6] - p[3:6])
        angvel_err = np.linalg.norm(r[10:13] - p[10:13])
        print(f"{t:>5}  {pos_err:>12.3f}  {vel_err:>14.3f}  {angvel_err:>12.3f}  {r[13]:>10.4f}  {p[13]:>10.4f}")

    # Show first-step debug
    print("\n--- state_0 (from HDF5) ---")
    print(f"pos={proprio[0, 0:3]}  vel={proprio[0, 3:6]}  quat={proprio[0, 6:10]}")
    print(f"angvel={proprio[0, 10:13]}  fuel={proprio[0, 13]}  target_rel={proprio[0, 14:17]}")
    print("\n--- action_0 ---")
    print(f"{actions[0]}")
    print("\n--- state_1 real vs physics ---")
    print(f"real pos: {proprio[1, 0:3]}   vel: {proprio[1, 3:6]}")
    print(f"phys pos: {traj[1, 0:3]}   vel: {traj[1, 3:6]}")
    print(f"\ndelta per-axis (real - phys) state_1:")
    print(f"  pos: {proprio[1, 0:3] - traj[1, 0:3]}")
    print(f"  vel: {proprio[1, 3:6] - traj[1, 3:6]}")


if __name__ == "__main__":
    main()

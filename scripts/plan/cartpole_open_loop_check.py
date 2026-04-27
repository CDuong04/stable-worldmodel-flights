"""Open-loop sanity check for the trained cart-pole world model.

Pulls a handful of real trajectories from the HDF5 dataset, encodes the first
frame, then rolls the predictor forward using the expert action sequence
recorded for that episode. Decodes the predicted latents and compares to the
ground-truth state at every horizon step.

Outputs:
  - per-horizon-step error curves (mean abs error per dim, plus mean dV(true)
    vs dV(predicted))
  - a JSON summary with aggregate numbers
  - a PNG with predicted vs actual trajectories for one episode

Tells us how many steps of horizon the world model can be trusted for. If the
error blows up after a handful of steps, the planner can't see far enough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Reuse the planner's helpers so the sanity check matches the eval pipeline.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cartpole_mpc import (  # noqa: E402
    NormStats,
    compute_norm_stats,
    decode_states,
    encode_images,
    load_model,
    make_image_preprocessor,
    rollout_latent,
    compute_lyapunov_np,
)

from stable_worldmodel.data import get_cache_dir  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--dataset-name', type=str, default='cartpole_expert_worldmodel')
    p.add_argument('--cache-dir', type=str, default=None)
    p.add_argument('--episodes', type=int, default=8,
                   help='How many episodes to replay.')
    p.add_argument('--horizon', type=int, default=80,
                   help='How many open-loop prediction steps from the initial frame.')
    p.add_argument('--start-step', type=int, default=0,
                   help='Step within each episode to start the rollout from.')
    p.add_argument('--seed', type=int, default=10_000_000,
                   help='Seed for choosing which episodes to sample.')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', type=str, default=None)
    return p.parse_args()


def pick_episodes(h5_path: Path, n: int, seed: int) -> list[tuple[int, int]]:
    """Return list of (episode_offset, episode_length)."""
    with h5py.File(h5_path, 'r') as f:
        ep_offset = f['ep_offset'][:]
        ep_len = f['ep_len'][:]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ep_len), size=min(n, len(ep_len)), replace=False)
    return [(int(ep_offset[i]), int(ep_len[i])) for i in idx]


def load_episode(h5_path: Path, offset: int, length: int) -> dict:
    """Load pixels / action / state for a contiguous chunk."""
    with h5py.File(h5_path, 'r') as f:
        pixels = f['pixels'][offset:offset + length]   # (T, H, W, 3) uint8
        action = f['action'][offset:offset + length]   # (T, 1) float
        state = f['state'][offset:offset + length]     # (T, 4) float
    return {'pixels': pixels, 'action': action, 'state': state}


@torch.no_grad()
def open_loop_rollout(
    model,
    stats: NormStats,
    pixels_init: np.ndarray,       # (HS, H, W, 3) uint8 — first HS frames
    action_seq: np.ndarray,        # (H_eff, 1) raw actions to replay
    img_size: int,
    history_size: int,
    device: str,
) -> np.ndarray:
    """Return predicted physical state trajectory of shape (HS + H_eff, state_dim).

    The leading HS frames correspond to the encoded real history; the rest are
    autoregressive predictions seeded from that history.
    """
    prep = make_image_preprocessor(img_size)
    imgs = prep(pixels_init).to(device)             # (HS, 3, h, w)
    z_hist = encode_images(model, imgs)             # (HS, D)
    z_init = z_hist.unsqueeze(0)                    # (1, HS, D)
    a = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)  # (1, H, A)
    z_traj = rollout_latent(
        model, z_init, a, stats, history_size=history_size,
    )                                               # (1, HS + H, D)
    s_pred = decode_states(model, z_traj, stats)    # (1, HS + H, state_dim)
    return s_pred[0].cpu().numpy()


def cossin_to_raw(s_cs: np.ndarray) -> np.ndarray:
    """Convert (T, 5) cos/sin state to (T, 4) raw [x, theta, x_dot, theta_dot]
    via theta = atan2(sin, cos)."""
    x, x_dot = s_cs[..., 0], s_cs[..., 1]
    cos_t, sin_t = s_cs[..., 2], s_cs[..., 3]
    theta = np.arctan2(sin_t, cos_t)
    theta_dot = s_cs[..., 4]
    return np.stack([x, theta, x_dot, theta_dot], axis=-1)


def per_dim_abs_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred, true: (T, 4). Returns (T, 4) absolute error per dim."""
    err = np.abs(pred - true).copy()
    # angle wraparound: bring θ-error into [0, π]
    dtheta = pred[:, 1] - true[:, 1]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    err[:, 1] = np.abs(dtheta)
    return err


def main():
    args = parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    weights_path = Path(args.weights).expanduser().resolve()
    out_dir = Path(args.out_dir) if args.out_dir else weights_path.parent / 'open_loop_check'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model from {weights_path}')
    model, cfg = load_model(weights_path)
    model = model.to(args.device).eval()

    datasets_dir = get_cache_dir(args.cache_dir, sub_folder='datasets')
    dataset_path = Path(datasets_dir, f'{args.dataset_name}.h5')
    state_repr = str(cfg.get('wm', {}).get('state_repr', 'raw')).lower()
    if state_repr not in ('raw', 'cossin'):
        state_repr = 'cossin' if cfg['wm']['state_dim'] == 5 else 'raw'
    print(f'state_repr={state_repr}')
    stats = compute_norm_stats(dataset_path, state_repr=state_repr).to(args.device)

    history_size = int(cfg.get('wm', {}).get('history_size', 1))
    eps = pick_episodes(dataset_path, args.episodes, args.seed)
    print(f'Selected {len(eps)} episodes from {dataset_path.name}; '
          f'history_size={history_size}')

    all_errors = []
    all_pred = []
    all_true = []
    horizon = args.horizon
    for ep_idx, (offset, length) in enumerate(eps):
        # Need history_size warmup frames + horizon predictions.
        needed = args.start_step + history_size + horizon
        if length <= needed:
            print(f'  ep {ep_idx}: too short ({length}); skipping')
            continue
        ep = load_episode(
            dataset_path, offset + args.start_step, history_size + horizon,
        )
        pixels_init = ep['pixels'][:history_size]                     # (HS, H, W, 3)
        action_seq = ep['action'][history_size - 1:history_size - 1 + horizon]
        # action[t] is the action taken at time t; we need actions from the LAST history
        # position onwards to drive the rollout
        true_state = ep['state'][history_size - 1:history_size - 1 + horizon + 1]

        action_seq = np.nan_to_num(action_seq, 0.0).astype(np.float32)

        pred_full = open_loop_rollout(
            model, stats, pixels_init, action_seq,
            img_size=cfg['img_size'],
            history_size=history_size,
            device=args.device,
        )                                                              # (HS + horizon, S)
        # Compare from the last history frame onwards (current state + predictions).
        pred_state = pred_full[history_size - 1:]                      # (horizon + 1, S)

        if pred_state.shape[-1] == 5:
            pred_raw = cossin_to_raw(pred_state)
        else:
            pred_raw = pred_state

        err = per_dim_abs_error(pred_raw, true_state)
        all_errors.append(err)
        all_pred.append(pred_raw)
        all_true.append(true_state)

    if not all_errors:
        raise SystemExit('no episodes survived; reduce --horizon or --start-step')

    errors = np.stack(all_errors, axis=0)                 # (E, H+1, 4)
    mean_err = errors.mean(axis=0)                        # (H+1, 4)
    median_err = np.median(errors, axis=0)
    p90_err = np.percentile(errors, 90, axis=0)

    # Single-step error: predicted z_1 vs decoded z_1 from the next frame
    print('\nPer-step open-loop error (mean abs across episodes):')
    print(f'{"step":>4} | {"x":>7} {"theta":>7} {"x_dot":>7} {"theta_dot":>9}')
    for k in [0, 1, 2, 5, 10, 20, 40, min(80, horizon)]:
        if k > horizon:
            continue
        e = mean_err[k]
        print(f'{k:>4} | {e[0]:>7.3f} {e[1]:>7.3f} {e[2]:>7.3f} {e[3]:>9.3f}')

    # Also report Lyapunov-value error vs horizon
    pred_arr = np.stack(all_pred, axis=0)                 # (E, H+1, 4)
    true_arr = np.stack(all_true, axis=0)                 # (E, H+1, 4)
    V_pred = compute_lyapunov_np(pred_arr)                # (E, H+1)
    V_true = compute_lyapunov_np(true_arr)                # (E, H+1)
    V_err = np.abs(V_pred - V_true).mean(axis=0)          # (H+1,)
    print('\nLyapunov-value error vs horizon (mean across episodes):')
    for k in [0, 1, 2, 5, 10, 20, 40, min(80, horizon)]:
        if k > horizon:
            continue
        print(f'  step {k:>3}: |V_pred - V_true| = {V_err[k]:.4f}'
              f'  (V_true = {V_true[:, k].mean():.3f})')

    # Save JSON summary
    summary = {
        'config': {
            'weights': str(weights_path),
            'episodes': len(all_errors),
            'horizon': horizon,
            'start_step': args.start_step,
        },
        'mean_abs_error_per_step': mean_err.tolist(),
        'median_abs_error_per_step': median_err.tolist(),
        'p90_abs_error_per_step': p90_err.tolist(),
        'mean_abs_V_error_per_step': V_err.tolist(),
        'V_true_mean_per_step': V_true.mean(0).tolist(),
        'state_dim_names': ['x', 'theta', 'x_dot', 'theta_dot'],
    }
    out_path = out_dir / 'open_loop_check.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {out_path}')

    # Save a quick plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        steps = np.arange(horizon + 1)

        for j, name in enumerate(['x', 'theta', 'x_dot', 'theta_dot']):
            ax = axes[j // 2, j % 2]
            ax.plot(steps, mean_err[:, j], label='mean')
            ax.plot(steps, p90_err[:, j], linestyle='--', label='p90')
            ax.set_title(f'|err| in {name}')
            ax.set_xlabel('horizon step')
            ax.set_ylabel('abs error (physical units)')
            ax.grid(True, alpha=0.3)
            ax.legend()

        ax = axes[1, 2]
        ax.plot(steps, V_err, label='|V_pred - V_true|')
        ax.plot(steps, V_true.mean(0), linestyle='--', label='V_true (mean)')
        ax.set_title('Lyapunov value error')
        ax.set_xlabel('horizon step')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-left: predicted vs true theta for episode 0
        ax = axes[0, 2]
        ax.plot(steps, true_arr[0, :, 1], label='theta_true')
        ax.plot(steps, pred_arr[0, :, 1], label='theta_pred', linestyle='--')
        ax.set_title('Episode 0: theta')
        ax.set_xlabel('horizon step')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = out_dir / 'open_loop_check.png'
        fig.savefig(png_path, dpi=120)
        print(f'Wrote {png_path}')
    except Exception as e:
        print(f'(plot skipped: {e})')


if __name__ == '__main__':
    main()

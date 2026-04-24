"""Collect cart-pole expert trajectories in native Stable World-Model HDF5 format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset, get_cache_dir
from stable_worldmodel.envs.dmcontrol import (
    CartpoleExpertPolicy,
    ExpertPolicy as SB3ExpertPolicy,
)
from stable_worldmodel.utils import record_video_from_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect cart-pole expert trajectories as a native .h5 dataset.'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='cartpole_expert_perturbed',
        help='Dataset name without extension.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1600,
        help='Total number of episodes to record.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=64,
        help='Episodes per collection chunk; each chunk gets new perturbations.',
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=8,
        help='Number of vectorized environments to use during collection.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=('HEIGHT', 'WIDTH'),
        help='Rendered image size stored in the dataset.',
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=500,
        help='Episode horizon used during collection.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=7,
        help='Base random seed.',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Optional override for STABLEWM_HOME.',
    )
    parser.add_argument(
        '--expert',
        choices=('auto', 'analytic', 'sb3'),
        default='auto',
        help='Expert backend to use.',
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='SB3 expert checkpoint path (needed for --expert sb3).',
    )
    parser.add_argument(
        '--vec-normalize-path',
        type=str,
        default=None,
        help='VecNormalize stats path (needed for --expert sb3).',
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.06,
        help='Small Gaussian action noise added every step.',
    )
    parser.add_argument(
        '--burst-prob',
        type=float,
        default=0.04,
        help='Probability of starting a short perturbation burst in a step.',
    )
    parser.add_argument(
        '--burst-noise-std',
        type=float,
        default=0.45,
        help='Noise scale used during perturbation bursts.',
    )
    parser.add_argument(
        '--burst-steps',
        type=int,
        nargs=2,
        default=(2, 5),
        metavar=('MIN', 'MAX'),
        help='Inclusive range for perturbation burst duration.',
    )
    parser.add_argument(
        '--vary-visuals',
        action='store_true',
        help='Apply mild visual domain randomization across chunks.',
    )
    parser.add_argument(
        '--vary-dynamics',
        action='store_true',
        help='Apply mild dynamics randomization across chunks.',
    )
    parser.add_argument(
        '--video-episodes',
        type=int,
        default=4,
        help='Number of replay videos to export after collection. Set 0 to skip.',
    )
    return parser.parse_args()


def build_policy(args):
    use_sb3 = args.expert == 'sb3' or (
        args.expert == 'auto' and args.ckpt_path and args.vec_normalize_path
    )

    if use_sb3:
        if not args.ckpt_path or not args.vec_normalize_path:
            raise ValueError(
                '--ckpt-path and --vec-normalize-path are required for SB3 expert collection.'
            )
        return SB3ExpertPolicy(
            ckpt_path=args.ckpt_path,
            vec_normalize_path=args.vec_normalize_path,
            noise_std=args.noise_std,
            seed=args.seed,
        )

    return CartpoleExpertPolicy(
        noise_std=args.noise_std,
        burst_prob=args.burst_prob,
        burst_noise_std=args.burst_noise_std,
        burst_steps_range=tuple(args.burst_steps),
        seed=args.seed,
    )


def sample_rgb_pair(rng: np.random.Generator) -> list[list[float]]:
    base = rng.uniform(0.05, 0.5, size=(2, 3))
    contrast = rng.uniform(0.05, 0.2, size=(2, 3))
    colors = np.clip(base + np.array([[0.0], [1.0]]) * contrast, 0.0, 1.0)
    return colors.tolist()


def sample_variation_values(
    rng: np.random.Generator,
    vary_visuals: bool,
    vary_dynamics: bool,
):
    values = {}
    watch_keys = []

    if vary_visuals:
        values.update(
            {
                'agent.color': rng.uniform(0.15, 0.95, size=3).tolist(),
                'floor.color': sample_rgb_pair(rng),
                'light.intensity': [float(rng.uniform(0.5, 0.95))],
                'agent.cart_shape': int(rng.integers(0, 2)),
            }
        )
        watch_keys.extend(
            [
                'agent.color',
                'floor.color',
                'light.intensity',
                'agent.cart_shape',
            ]
        )

    if vary_dynamics:
        values.update(
            {
                'agent.cart_mass': [float(rng.uniform(0.85, 1.15))],
                'agent.pole_density': [float(rng.uniform(850.0, 1150.0))],
                'gravity.x': [float(rng.uniform(-0.5, 0.5))],
                'gravity.y': [float(rng.uniform(-0.25, 0.25))],
                'gravity.z': [float(rng.uniform(-10.8, -8.8))],
            }
        )
        watch_keys.extend(
            [
                'agent.cart_mass',
                'agent.pole_density',
                'gravity.x',
                'gravity.y',
                'gravity.z',
            ]
        )

    return values, watch_keys


def export_videos(args):
    if args.video_episodes <= 0:
        return

    dataset = HDF5Dataset(
        name=args.dataset_name,
        num_steps=1,
        cache_dir=args.cache_dir,
    )
    num_episodes = min(args.video_episodes, len(dataset.lengths))
    out_dir = (
        get_cache_dir(args.cache_dir, sub_folder='videos')
        / args.dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    record_video_from_dataset(
        video_path=out_dir,
        dataset=dataset,
        episode_idx=list(range(num_episodes)),
        max_steps=args.max_episode_steps,
    )
    print(f'Exported {num_episodes} replay videos to {out_dir}')


def main():
    args = parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')

    episodes_recorded = 0
    datasets_dir = get_cache_dir(args.cache_dir, sub_folder='datasets')
    dataset_path = Path(datasets_dir, f'{args.dataset_name}.h5')
    if dataset_path.exists():
        with h5py.File(dataset_path, 'r') as handle:
            if 'ep_len' in handle:
                episodes_recorded = int(handle['ep_len'].shape[0])
        print(
            f'Resuming existing dataset at {dataset_path} '
            f'({episodes_recorded} episodes)'
        )

    world = swm.World(
        'swm/CartpoleDMControl-v0',
        num_envs=args.num_envs,
        image_shape=tuple(args.image_size),
        max_episode_steps=args.max_episode_steps,
        render_mode='rgb_array',
    )
    world.set_policy(build_policy(args))

    rng = np.random.default_rng(args.seed)

    try:
        while episodes_recorded < args.episodes:
            next_total = min(args.episodes, episodes_recorded + args.chunk_size)
            variation_values, watch_keys = sample_variation_values(
                rng,
                vary_visuals=args.vary_visuals,
                vary_dynamics=args.vary_dynamics,
            )

            options = None
            if variation_values:
                options = {
                    'variation': watch_keys,
                    'variation_values': variation_values,
                }
                print(
                    f'Collecting episodes {episodes_recorded}:{next_total} '
                    f'with perturbations: {variation_values}'
                )
            else:
                print(f'Collecting episodes {episodes_recorded}:{next_total}')

            world.record_dataset(
                dataset_name=args.dataset_name,
                episodes=next_total,
                seed=args.seed + episodes_recorded,
                cache_dir=args.cache_dir,
                options=options,
            )
            episodes_recorded = next_total
    finally:
        world.close()

    print(f'Dataset written to {dataset_path}')

    dataset = HDF5Dataset(
        name=args.dataset_name,
        num_steps=1,
        cache_dir=args.cache_dir,
    )
    print(f'Episodes: {len(dataset.lengths)}')
    print(f'Columns: {dataset.column_names}')
    export_videos(args)


if __name__ == '__main__':
    main()

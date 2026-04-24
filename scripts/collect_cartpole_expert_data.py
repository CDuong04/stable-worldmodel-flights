"""Collect cart-pole expert trajectories in native Stable World-Model HDF5 format."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset, get_cache_dir
from stable_worldmodel.envs.dmcontrol import (
    CartpoleExpertPolicy,
    ExpertPolicy as SB3ExpertPolicy,
)
from stable_worldmodel.utils import record_video_from_dataset


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class CollectionProfile:
    idx: int
    name: str
    weight: float


PROFILES = (
    CollectionProfile(0, 'default_swingup', 0.22),
    CollectionProfile(1, 'wide_swingup', 0.24),
    CollectionProfile(2, 'midarc_recovery', 0.20),
    CollectionProfile(3, 'near_upright_recovery', 0.22),
    CollectionProfile(4, 'rail_recovery', 0.12),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect cart-pole expert trajectories as a native .h5 dataset.'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='cartpole_expert_worldmodel',
        help='Dataset name without extension.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1800,
        help='Total number of episodes to record.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=64,
        help='Print a profile-mix progress update every N completed episodes.',
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
        default=0.02,
        help='Small Gaussian action noise added every step.',
    )
    parser.add_argument(
        '--burst-prob',
        type=float,
        default=0.01,
        help='Probability of starting a short perturbation burst in a step.',
    )
    parser.add_argument(
        '--burst-noise-std',
        type=float,
        default=0.15,
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
        default=8,
        help='Number of replay videos to export after collection. Set 0 to skip.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete an existing dataset/video export with the same name before collecting.',
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


def sample_from_segments(
    rng: np.random.Generator,
    segments: list[tuple[float, float]],
) -> float:
    lengths = np.array([high - low for low, high in segments], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    low, high = segments[idx]
    return float(rng.uniform(low, high))


def sample_rgb_pair(rng: np.random.Generator) -> list[list[float]]:
    base = rng.uniform(0.05, 0.5, size=(2, 3))
    contrast = rng.uniform(0.05, 0.2, size=(2, 3))
    colors = np.clip(base + np.array([[0.0], [1.0]]) * contrast, 0.0, 1.0)
    return colors.tolist()


def sample_profile(rng: np.random.Generator) -> CollectionProfile:
    weights = np.array([profile.weight for profile in PROFILES], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(PROFILES), p=weights))
    return PROFILES[idx]


def sample_state(profile: CollectionProfile, rng: np.random.Generator):
    if profile.name == 'default_swingup':
        return None

    if profile.name == 'wide_swingup':
        x = rng.uniform(-1.2, 1.2)
        theta = _wrap_to_pi(np.pi + rng.uniform(-0.65, 0.65))
        x_dot = rng.uniform(-1.2, 1.2)
        theta_dot = rng.uniform(-3.0, 3.0)
    elif profile.name == 'midarc_recovery':
        x = rng.uniform(-1.35, 1.35)
        theta = sample_from_segments(
            rng,
            [(-2.6, -0.6), (0.6, 2.6)],
        )
        x_dot = rng.uniform(-1.8, 1.8)
        theta_dot = rng.uniform(-6.0, 6.0)
    elif profile.name == 'near_upright_recovery':
        x = rng.uniform(-1.45, 1.45)
        theta = rng.uniform(-0.9, 0.9)
        x_dot = rng.uniform(-2.2, 2.2)
        theta_dot = rng.uniform(-8.0, 8.0)
    elif profile.name == 'rail_recovery':
        x = sample_from_segments(
            rng,
            [(-1.65, -1.15), (1.15, 1.65)],
        )
        theta = rng.uniform(-1.25, 1.25)
        x_dot = rng.uniform(-1.2, 1.2)
        theta_dot = rng.uniform(-5.0, 5.0)
    else:
        raise ValueError(f'Unknown profile: {profile.name}')

    return [float(x), float(theta), float(x_dot), float(theta_dot)]


def interpolate(low: float, high: float, strength: float) -> float:
    return low + (high - low) * strength


def sample_variation_values(
    rng: np.random.Generator,
    vary_visuals: bool,
    vary_dynamics: bool,
    profile: CollectionProfile,
):
    values = {}
    watch_keys = []
    strength = {
        'default_swingup': 0.30,
        'wide_swingup': 0.55,
        'midarc_recovery': 0.70,
        'near_upright_recovery': 0.45,
        'rail_recovery': 0.60,
    }[profile.name]

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
        cart_mass_delta = interpolate(0.10, 0.22, strength)
        pole_density_delta = interpolate(120.0, 260.0, strength)
        lateral_gravity = interpolate(0.20, 0.80, strength)
        vertical_gravity_delta = interpolate(0.45, 1.20, strength)
        values.update(
            {
                'agent.cart_mass': [
                    float(rng.uniform(1.0 - cart_mass_delta, 1.0 + cart_mass_delta))
                ],
                'agent.pole_density': [
                    float(
                        rng.uniform(
                            1000.0 - pole_density_delta,
                            1000.0 + pole_density_delta,
                        )
                    )
                ],
                'gravity.x': [float(rng.uniform(-lateral_gravity, lateral_gravity))],
                'gravity.y': [
                    float(rng.uniform(-0.5 * lateral_gravity, 0.5 * lateral_gravity))
                ],
                'gravity.z': [
                    float(
                        rng.uniform(
                            -9.81 - vertical_gravity_delta,
                            -9.81 + vertical_gravity_delta,
                        )
                    )
                ],
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


def sample_episode_spec(
    args,
    rng: np.random.Generator,
):
    profile = sample_profile(rng)
    state = sample_state(profile, rng)
    variation_values, watch_keys = sample_variation_values(
        rng,
        vary_visuals=args.vary_visuals,
        vary_dynamics=args.vary_dynamics,
        profile=profile,
    )

    options = None
    if state is not None or variation_values:
        options = {}
        if state is not None:
            options['state'] = state
        if variation_values:
            options['variation'] = watch_keys
            options['variation_values'] = variation_values

    return profile, options


def attach_profile_info(world: swm.World, profile_ids: np.ndarray) -> None:
    world.infos['profile_idx'] = profile_ids.astype(np.int32, copy=False)


def episode_seed(base_seed: int | None, episode_idx: int, env_idx: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed + 1009 * episode_idx + env_idx)


def collect_dataset(
    world: swm.World,
    args,
):
    rng = np.random.default_rng(args.seed)
    datasets_dir = get_cache_dir(args.cache_dir, sub_folder='datasets')
    dataset_path = Path(datasets_dir, f'{args.dataset_name}.h5')

    world.terminateds = np.zeros(world.num_envs, dtype=bool)
    world.truncateds = np.zeros(world.num_envs, dtype=bool)
    episode_buffers = [defaultdict(list) for _ in range(world.num_envs)]
    profile_counts = np.zeros(len(PROFILES), dtype=np.int64)

    h5_kwargs = {
        'name': str(dataset_path),
        'mode': 'a' if dataset_path.exists() else 'w',
        'libver': 'latest',
    }
    if not dataset_path.exists():
        h5_kwargs.update(
            {'fs_strategy': 'page', 'fs_page_size': 4 * 1024 * 1024}
        )

    with h5py.File(**h5_kwargs) as handle:
        handle.swmr_mode = True

        if 'ep_len' in handle:
            episodes_recorded = int(handle['ep_len'].shape[0])
            global_step_ptr = (
                int(handle['ep_offset'][-1] + handle['ep_len'][-1])
                if episodes_recorded > 0
                else 0
            )
            initialized = True
            print(f'Resuming existing dataset at {dataset_path} ({episodes_recorded} episodes)')
        else:
            episodes_recorded = 0
            global_step_ptr = 0
            initialized = False

        profile_ids = np.zeros(world.num_envs, dtype=np.int32)
        initial_options = []
        initial_seeds = []
        for env_idx in range(world.num_envs):
            profile, options = sample_episode_spec(args, rng)
            profile_ids[env_idx] = profile.idx
            initial_options.append(options)
            initial_seeds.append(
                episode_seed(args.seed, episodes_recorded + env_idx, env_idx)
            )

        world.reset(seed=initial_seeds, options=initial_options)
        attach_profile_info(world, profile_ids)
        world._dump_step_data(episode_buffers)

        with tqdm(
            total=args.episodes,
            initial=episodes_recorded,
            desc='Recording',
        ) as pbar:
            while episodes_recorded < args.episodes:
                world.step()
                attach_profile_info(world, profile_ids)
                world._dump_step_data(episode_buffers)

                for env_idx in range(world.num_envs):
                    if not (
                        world.terminateds[env_idx] or world.truncateds[env_idx]
                    ):
                        continue

                    finished_ep = world._handle_done_ep(
                        episode_buffers,
                        env_idx,
                        episodes_recorded,
                    )

                    if not initialized:
                        world._init_h5_datasets(handle, finished_ep)
                        initialized = True

                    steps_written = world._write_episode(
                        handle,
                        finished_ep,
                        global_step_ptr,
                    )
                    global_step_ptr += steps_written
                    profile_counts[profile_ids[env_idx]] += 1
                    episodes_recorded += 1
                    pbar.update(1)
                    handle.flush()

                    if episodes_recorded >= args.episodes:
                        break

                    profile, options = sample_episode_spec(args, rng)
                    profile_ids[env_idx] = profile.idx
                    reset_seed = episode_seed(args.seed, episodes_recorded, env_idx)
                    world._reset_single_env(env_idx, reset_seed, options)
                    attach_profile_info(world, profile_ids)
                    world._dump_step_data(episode_buffers, env_idx=env_idx)

                    if (
                        args.chunk_size > 0
                        and episodes_recorded % args.chunk_size == 0
                    ):
                        pretty_counts = {
                            profile.name: int(profile_counts[profile.idx])
                            for profile in PROFILES
                            if profile_counts[profile.idx] > 0
                        }
                        print(
                            f'Collected {episodes_recorded}/{args.episodes} episodes '
                            f'| profile mix: {pretty_counts}'
                        )

    return dataset_path


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


def summarize_dataset(dataset_path: Path):
    summary_path = dataset_path.with_suffix('.summary.json')
    with h5py.File(dataset_path, 'r') as handle:
        qpos = handle['qpos'][:]
        qvel = handle['qvel'][:]
        ep_len = handle['ep_len'][:]
        ep_offset = handle['ep_offset'][:]
        profile_counts = {}
        if 'profile_idx' in handle:
            profile_idx = handle['profile_idx'][ep_offset]
            profile_counts = {
                profile.name: int(np.sum(profile_idx == profile.idx))
                for profile in PROFILES
                if np.any(profile_idx == profile.idx)
            }

    theta = _wrap_to_pi(qpos[:, 1])
    cart_pos = qpos[:, 0]
    cart_vel = qvel[:, 0]
    pole_vel = qvel[:, 1]

    reached_top = []
    reached_upright = []
    for start, length in zip(ep_offset, ep_len, strict=True):
        end = int(start + length)
        ep_theta = theta[int(start):end]
        reached_top.append(bool(np.any(np.abs(ep_theta) < 0.60)))
        reached_upright.append(bool(np.any(np.abs(ep_theta) < 0.35)))

    theta_bins = np.linspace(-np.pi, np.pi, 33)
    theta_dot_bins = np.linspace(-12.0, 12.0, 33)
    occupancy, _, _ = np.histogram2d(theta, pole_vel, bins=(theta_bins, theta_dot_bins))
    occupancy_fraction = float(np.count_nonzero(occupancy) / occupancy.size)

    summary = {
        'dataset_path': str(dataset_path),
        'episodes': int(len(ep_len)),
        'total_steps': int(qpos.shape[0]),
        'cart_position_range': [float(cart_pos.min()), float(cart_pos.max())],
        'cart_velocity_range': [float(cart_vel.min()), float(cart_vel.max())],
        'pole_angle_range': [float(theta.min()), float(theta.max())],
        'pole_velocity_range': [float(pole_vel.min()), float(pole_vel.max())],
        'episodes_reaching_upper_half': float(np.mean(reached_top)),
        'episodes_entering_upright_band': float(np.mean(reached_upright)),
        'theta_theta_dot_occupancy_fraction': occupancy_fraction,
        'profile_counts': profile_counts,
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Saved coverage summary to {summary_path}')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')

    datasets_dir = get_cache_dir(args.cache_dir, sub_folder='datasets')
    dataset_path = Path(datasets_dir, f'{args.dataset_name}.h5')
    videos_dir = get_cache_dir(args.cache_dir, sub_folder='videos')
    video_path = Path(videos_dir, args.dataset_name)

    if args.overwrite:
        if dataset_path.exists():
            dataset_path.unlink()
            print(f'Deleted existing dataset at {dataset_path}')
        if video_path.exists():
            shutil.rmtree(video_path)
            print(f'Deleted existing videos at {video_path}')

    world = swm.World(
        'swm/CartpoleDMControl-v0',
        num_envs=args.num_envs,
        image_shape=tuple(args.image_size),
        max_episode_steps=args.max_episode_steps,
    )
    world.set_policy(build_policy(args))

    try:
        dataset_path = collect_dataset(world, args)
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
    summarize_dataset(dataset_path)
    export_videos(args)


if __name__ == '__main__':
    main()

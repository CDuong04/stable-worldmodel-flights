"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP
from transformers import ViTConfig, ViTModel

_VIT_SIZES = {
    'tiny':  {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3,  'intermediate_size': 768},
    'small': {'hidden_size': 384, 'num_hidden_layers': 12, 'num_attention_heads': 6,  'intermediate_size': 1536},
    'base':  {'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'intermediate_size': 3072},
}


def _build_lewm_from_full_cfg(cfg: dict) -> LeWM:
    """Rebuild a LeWM from the full training cfg (as written by save_pretrained)."""
    vit_cfg = ViTConfig(
        image_size=cfg['img_size'], patch_size=cfg['patch_size'],
        num_channels=3, **_VIT_SIZES[cfg['encoder_scale']],
    )
    encoder = ViTModel(vit_cfg, add_pooling_layer=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg['wm'].get('embed_dim', hidden_dim)
    effective_act_dim = cfg['data']['dataset']['frameskip'] * cfg['wm']['action_dim']
    predictor = Predictor(
        num_frames=cfg['wm']['history_size'], input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg['predictor'],
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return LeWM(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)


def _load_lewm_checkpoint(path: str) -> LeWM:
    import json
    path = Path(path)
    with open(path.parent / 'config.json') as f:
        cfg = json.load(f)
    model = _build_lewm_from_full_cfg(cfg)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    image_shape = tuple(cfg.get('image_shape', (224, 224)))
    world = swm.World(**cfg.world, image_shape=image_shape)

    # create the transform
    transform = {
        'pixels': img_transform(cfg),
        'goal': img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy == 'random':
        policy = swm.policy.RandomPolicy()
    elif policy == 'physics':
        print('[eval_wm] pure-physics mode: skipping WM load, using solver-config model')
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        try:
            model = swm.wm.utils.load_pretrained(cfg.policy)
        except Exception as e:
            print(f'[eval_wm] load_pretrained failed ({e!s}); falling back to manual LeWM rebuild')
            model = _load_lewm_checkpoint(cfg.policy)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        # Optional: wrap the raw WM in a PhysicsConstrainedCost (hybrid mode).
        cost_wrapper = cfg.get('cost_wrapper', None)
        if cost_wrapper == 'physics_constrained':
            from stable_worldmodel.solver.rocket_physics import PhysicsConstrainedCost
            lam = float(cfg.get('hybrid_lambda', 1.0))
            print(f'[eval_wm] hybrid mode: wrapping WM in PhysicsConstrainedCost (lambda={lam})')
            model = PhysicsConstrainedCost(
                world_model=model,
                lambda_constraint=lam,
                project_actions=True,
                device='cuda',
            )

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    # Project actions to rocket-feasible ranges (throttle [0,1], gimbal ±5°, fire boolean).
    # CEM samples unbounded Gaussians; PyFlyt crashes on out-of-bound pwm.
    env_name = cfg.world.get('env_name', '')
    if 'Rocket' in env_name and not isinstance(policy, swm.policy.RandomPolicy):
        from stable_worldmodel.solver.rocket_physics import project_rocket_actions
        _orig_get_action = policy.get_action

        def _projected_get_action(info_dict, **kwargs):
            action = _orig_get_action(info_dict, **kwargs)
            action_t = torch.from_numpy(np.asarray(action)).float()
            action_t = project_rocket_actions(action_t)
            return action_t.numpy().astype(action.dtype if hasattr(action, 'dtype') else np.float32)

        policy.get_action = _projected_get_action

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    # Optional: warm-start CEM from recorded expert actions.
    if cfg.get('warm_start_expert', False) and hasattr(policy, '_next_init'):
        print('[eval_wm] warm-start from expert actions enabled')
        horizon = int(cfg.plan_config.horizon)
        n_envs = int(cfg.eval.num_eval)

        import h5py
        h5_path = Path(
            cfg.cache_dir or swm.data.utils.get_cache_dir()
        ) / 'datasets' / f'{cfg.eval.dataset_name}.h5'
        with h5py.File(h5_path, 'r') as f:
            ep_offset = f['ep_offset'][:]
            ep_len = f['ep_len'][:]
            expert_actions = [None] * n_envs
            for env_idx, (ep, st) in enumerate(
                zip(eval_episodes.tolist(), eval_start_idx.tolist())
            ):
                g0 = int(ep_offset[ep]) + int(st)
                g1 = int(ep_offset[ep]) + int(ep_len[ep])
                traj = f['action'][g0:g1]
                traj = np.nan_to_num(traj, nan=0.0)
                expert_actions[env_idx] = traj

        step_counter = [0]
        _warm_orig_get_action = policy.get_action

        def _warm_get_action(info_dict, **kwargs):
            t = step_counter[0]
            init = np.zeros((n_envs, horizon, 7), dtype=np.float32)
            for ei, traj in enumerate(expert_actions):
                remaining = traj.shape[0] - t
                take = min(horizon, max(remaining, 0))
                if take > 0:
                    init[ei, :take] = traj[t : t + take]
            policy._next_init = torch.from_numpy(init).float()
            step_counter[0] += 1
            return _warm_orig_get_action(info_dict, **kwargs)

        policy.get_action = _warm_get_action

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video_path=results_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()

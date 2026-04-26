"""MPC evaluation for the trained cart-pole world model.

Compares two cost functions over the same rollouts and the same candidate
action sequences:

  - standard MPC : terminal cost only          C = V(s_H)
  - Lyapunov MPC : terminal + monotonic penalty C = V(s_H) + lambda * sum_k max(0, V(s_{k+1}) - V(s_k))

Both methods plan in latent space using the trained encoder + LeWM dynamics
predictor + state decoder, then decode predicted latents to physical state and
score them with the Lyapunov function

  V(s) = 0.1*x^2 + (1 - cos(theta)) + 0.05*x_dot^2 + 0.1*theta_dot^2

Rollouts are vectorised over (num_envs x N candidates) on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as tv_v2

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset, get_cache_dir
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

# Re-use the exact model class used during training so checkpoint keys match.
import sys
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'train'
sys.path.insert(0, str(SCRIPTS_DIR))
from cartpole_wm import CartpoleLeWM, make_vit_encoder  # noqa: E402


# ----------------------------------------------------------------------------- #
#  Stats: state / action normalization were computed at runtime from the HDF5  #
#  dataset during training. Recompute them here so the decoder output can be   #
#  inverse-transformed to physical state, and so we can normalize candidate    #
#  actions before feeding them to the action_encoder.                          #
# --------------------------------------------------------------------------- #


@dataclass
class NormStats:
    state_mean: torch.Tensor   # (1, 4)
    state_std: torch.Tensor    # (1, 4)
    action_mean: torch.Tensor  # (1, 1)
    action_std: torch.Tensor   # (1, 1)

    def to(self, device):
        return NormStats(
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
        )


def _column_stats(arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(np.asarray(arr))
    t = t[~torch.isnan(t).any(dim=tuple(range(1, t.ndim)))]
    mean = t.mean(0, keepdim=True).float()
    std = t.std(0, keepdim=True).float().clamp_min(1e-6)
    return mean, std


def compute_norm_stats(dataset_path: Path, cache_path: Path | None = None) -> NormStats:
    """Compute (or load cached) z-score stats for `state` and `action`.

    Cached as JSON next to the .h5 so we don't re-stream the whole file.
    """
    if cache_path is None:
        cache_path = dataset_path.with_suffix('.norm_stats.json')

    if cache_path.exists():
        with open(cache_path) as f:
            d = json.load(f)
        return NormStats(
            state_mean=torch.tensor(d['state_mean']),
            state_std=torch.tensor(d['state_std']),
            action_mean=torch.tensor(d['action_mean']),
            action_std=torch.tensor(d['action_std']),
        )

    print(f'Computing normalization stats from {dataset_path} ...')
    with h5py.File(dataset_path, 'r') as f:
        state = f['state'][:]
        action = f['action'][:]

    s_mean, s_std = _column_stats(state)
    a_mean, a_std = _column_stats(action)

    cache = {
        'state_mean': s_mean.tolist(),
        'state_std': s_std.tolist(),
        'action_mean': a_mean.tolist(),
        'action_std': a_std.tolist(),
    }
    cache_path.write_text(json.dumps(cache, indent=2))
    print(f'Wrote {cache_path}')
    return NormStats(s_mean, s_std, a_mean, a_std)


# --------------------------------------------------------------------------- #
#  Model loading                                                              #
# --------------------------------------------------------------------------- #


def build_model_from_config(cfg: dict) -> CartpoleLeWM:
    """Reconstruct CartpoleLeWM with the same architecture used at train time."""
    encoder = make_vit_encoder(
        scale=cfg['encoder_scale'],
        image_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg['wm'].get('embed_dim', hidden_dim)
    effective_act_dim = cfg['data']['dataset']['frameskip'] * cfg['wm']['action_dim']

    predictor = Predictor(
        num_frames=cfg['wm']['history_size'],
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg['predictor'],
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    proj_norm = str(cfg.get('projector', {}).get('norm', 'bn')).lower()
    proj_norm_fn = {'bn': nn.BatchNorm1d, 'ln': nn.LayerNorm, 'none': None}[proj_norm]

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=proj_norm_fn,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=proj_norm_fn,
    )
    state_decoder = nn.Sequential(
        nn.Linear(hidden_dim, cfg['decoder']['hidden_dim']),
        nn.GELU(),
        nn.LayerNorm(cfg['decoder']['hidden_dim']),
        nn.Linear(cfg['decoder']['hidden_dim'], cfg['decoder']['hidden_dim']),
        nn.GELU(),
        nn.LayerNorm(cfg['decoder']['hidden_dim']),
        nn.Linear(cfg['decoder']['hidden_dim'], cfg['wm']['state_dim']),
    )
    return CartpoleLeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        state_decoder=state_decoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def load_model(weights_path: Path) -> tuple[CartpoleLeWM, dict]:
    """Load checkpoint (weights_epoch_*.pt) + the config.json next to it."""
    config_path = weights_path.parent / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found next to {weights_path}')
    with open(config_path) as f:
        cfg = json.load(f)
    model = build_model_from_config(cfg)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- #
#  Image preprocessing — must match training pipeline                          #
# --------------------------------------------------------------------------- #


def make_image_preprocessor(img_size: int):
    """Returns a callable: (np.uint8 (..., H, W, 3)) -> torch.float32 (..., 3, img, img).

    Mirrors the training transform: ToImage with ImageNet stats + Resize.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    resize = tv_v2.Resize(img_size, antialias=True)

    def _prep(img: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        # img: (..., H, W, 3) -> (..., 3, H, W)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # (1, H, W, 3)
        x = img.permute(0, 3, 1, 2).contiguous().float() / 255.0
        x = (x - mean) / std
        x = resize(x)
        return x

    return _prep


# --------------------------------------------------------------------------- #
#  Lyapunov function                                                          #
# --------------------------------------------------------------------------- #


def compute_lyapunov(states: torch.Tensor) -> torch.Tensor:
    """V(s) = 0.1*x^2 + (1 - cos(theta)) + 0.05*x_dot^2 + 0.1*theta_dot^2.

    states: (..., 4) physical [x, theta, x_dot, theta_dot]
    """
    x = states[..., 0]
    theta = states[..., 1]
    x_dot = states[..., 2]
    theta_dot = states[..., 3]
    cos_theta = torch.cos(theta)
    return 0.1 * x ** 2 + (1.0 - cos_theta) + 0.05 * x_dot ** 2 + 0.1 * theta_dot ** 2


def compute_lyapunov_np(states: np.ndarray) -> np.ndarray:
    x, theta, x_dot, theta_dot = states[..., 0], states[..., 1], states[..., 2], states[..., 3]
    return 0.1 * x ** 2 + (1.0 - np.cos(theta)) + 0.05 * x_dot ** 2 + 0.1 * theta_dot ** 2


# --------------------------------------------------------------------------- #
#  Latent rollout (vectorised over N candidates)                              #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_images(model: CartpoleLeWM, imgs: torch.Tensor) -> torch.Tensor:
    """imgs: (B, 3, H, W) -> z: (B, D)"""
    out = model.encoder(imgs, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]  # (B, hidden)
    return model.projector(cls)         # (B, embed)


@torch.no_grad()
def rollout_latent(
    model: CartpoleLeWM,
    z0: torch.Tensor,           # (B, D)
    action_seq: torch.Tensor,   # (B, H, A)  -- raw env-space actions
    stats: NormStats,
) -> torch.Tensor:
    """Return latent trajectory (B, H+1, D) where index 0 is z0 and 1..H are predicted."""
    B, H, A = action_seq.shape
    a_norm = (action_seq - stats.action_mean) / stats.action_std
    a_emb = model.action_encoder(a_norm)  # (B, H, D)

    z_curr = z0.unsqueeze(1)              # (B, 1, D), history_size=1
    out = [z0]
    for k in range(H):
        z_next = model.predict(z_curr, a_emb[:, k:k + 1])  # (B, 1, D)
        out.append(z_next.squeeze(1))
        z_curr = z_next                                    # history_size=1
    return torch.stack(out, dim=1)         # (B, H+1, D)


@torch.no_grad()
def decode_states(model: CartpoleLeWM, z_seq: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """z_seq: (B, T, D) -> physical states (B, T, 4)."""
    B, T, D = z_seq.shape
    s_norm = model.state_decoder(z_seq.reshape(B * T, D))
    s_phys = s_norm * stats.state_std + stats.state_mean
    return s_phys.reshape(B, T, -1)


# --------------------------------------------------------------------------- #
#  Cost functions / planners                                                  #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def plan_standard(
    model: CartpoleLeWM,
    z0: torch.Tensor,           # (E, D)
    candidates: torch.Tensor,   # (E, N, H, A)
    stats: NormStats,
) -> tuple[torch.Tensor, dict]:
    """Pick the action sequence with the lowest V(s_H). Vectorised over E envs."""
    E, N, H, A = candidates.shape
    D = z0.shape[-1]
    z0_flat = z0.unsqueeze(1).expand(E, N, D).reshape(E * N, D)
    cand_flat = candidates.reshape(E * N, H, A)
    z_traj = rollout_latent(model, z0_flat, cand_flat, stats)        # (EN, H+1, D)
    s_traj = decode_states(model, z_traj, stats)                     # (EN, H+1, 4)
    V_traj = compute_lyapunov(s_traj).reshape(E, N, H + 1)
    V_final = V_traj[:, :, -1]                                       # (E, N)
    best = V_final.argmin(dim=1)                                     # (E,)
    best_actions = candidates[torch.arange(E), best, 0]              # (E, A)
    return best_actions, {
        'V_traj': V_traj,
        'V_final': V_final,
        'best_idx': best,
    }


@torch.no_grad()
def plan_lyapunov(
    model: CartpoleLeWM,
    z0: torch.Tensor,
    candidates: torch.Tensor,
    stats: NormStats,
    lambda_v: float,
) -> tuple[torch.Tensor, dict]:
    """Same as plan_standard but adds sum_k max(0, V_{k+1} - V_k) penalty."""
    E, N, H, A = candidates.shape
    D = z0.shape[-1]
    z0_flat = z0.unsqueeze(1).expand(E, N, D).reshape(E * N, D)
    cand_flat = candidates.reshape(E * N, H, A)
    z_traj = rollout_latent(model, z0_flat, cand_flat, stats)
    s_traj = decode_states(model, z_traj, stats)
    V_traj = compute_lyapunov(s_traj).reshape(E, N, H + 1)
    V_final = V_traj[:, :, -1]
    dV = V_traj[:, :, 1:] - V_traj[:, :, :-1]                        # (E, N, H)
    penalty = torch.clamp(dV, min=0.0).sum(dim=-1)                   # (E, N)
    cost = V_final + lambda_v * penalty
    best = cost.argmin(dim=1)
    best_actions = candidates[torch.arange(E), best, 0]
    return best_actions, {
        'V_traj': V_traj,
        'V_final': V_final,
        'penalty': penalty,
        'cost': cost,
        'best_idx': best,
    }


# --------------------------------------------------------------------------- #
#  Policy wrapper for swm.World                                               #
# --------------------------------------------------------------------------- #


class MPCPolicy(BasePolicy):
    """Wraps the planner so we can reuse swm.World step/reset machinery."""

    def __init__(
        self,
        model: CartpoleLeWM,
        stats: NormStats,
        method: str,
        n_candidates: int,
        horizon: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        img_size: int,
        lambda_v: float = 1.0,
        device: str = 'cuda',
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert method in ('standard', 'lyapunov')
        self.type = 'mpc'
        self.method = method
        self.model = model.to(device).eval()
        self.stats = stats.to(device)
        self.N = n_candidates
        self.H = horizon
        self.A = action_dim
        self.alow, self.ahigh = action_low, action_high
        self.lambda_v = lambda_v
        self.device = device
        self.seed = seed
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.preprocess = make_image_preprocessor(img_size)
        # diagnostics buffers — set per evaluate() call
        self.last_diag: dict | None = None

    def set_seed(self, seed):
        self.seed = seed
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        pixels = info_dict['pixels']
        if isinstance(pixels, np.ndarray):
            pixels_np = pixels
        else:
            pixels_np = np.asarray(pixels)
        # collapse history dim if present (B, T, H, W, 3) -> use last frame
        if pixels_np.ndim == 5:
            pixels_np = pixels_np[:, -1]
        # to (E, 3, H, W)
        imgs = self.preprocess(pixels_np).to(self.device)
        z0 = encode_images(self.model, imgs)                         # (E, D)
        E = z0.shape[0]

        cand = torch.empty(
            (E, self.N, self.H, self.A), device=self.device,
        ).uniform_(self.alow, self.ahigh, generator=self.gen)

        if self.method == 'standard':
            best_a, diag = plan_standard(self.model, z0, cand, self.stats)
        else:
            best_a, diag = plan_lyapunov(self.model, z0, cand, self.stats, self.lambda_v)

        self.last_diag = {k: v.cpu() if torch.is_tensor(v) else v for k, v in diag.items()}
        return best_a.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
#  Evaluation harness                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeResult:
    method: str
    seed: int
    length: int
    succeeded: bool
    upright_fraction: float       # fraction of last 100 steps with |theta|<0.3
    final_abs_theta: float
    V_violation_rate: float       # fraction of steps with V_{t+1} > V_t
    avg_dV: float                 # mean (V_{t+1} - V_t) across steps
    final_V: float


def evaluate_method(
    method: str,
    model: CartpoleLeWM,
    stats: NormStats,
    cfg: dict,
    n_episodes: int,
    eval_seeds: list[int],
    horizon: int,
    n_candidates: int,
    lambda_v: float,
    max_steps: int,
    num_envs: int,
    device: str,
) -> list[EpisodeResult]:
    img_size = cfg['img_size']
    world = swm.World(
        'swm/CartpoleDMControl-v0',
        num_envs=num_envs,
        image_shape=(img_size, img_size),
        max_episode_steps=max_steps,
        verbose=0,
    )
    policy = MPCPolicy(
        model=model,
        stats=stats,
        method=method,
        n_candidates=n_candidates,
        horizon=horizon,
        action_dim=cfg['wm']['action_dim'],
        action_low=-1.0,
        action_high=1.0,
        img_size=img_size,
        lambda_v=lambda_v,
        device=device,
        seed=cfg.get('seed', 0) + 1000,
    )
    world.set_policy(policy)

    results: list[EpisodeResult] = []
    seed_iter = iter(eval_seeds)
    completed = 0

    while completed < n_episodes:
        # Batch: launch num_envs episodes at once.
        seeds_batch = []
        for _ in range(num_envs):
            try:
                seeds_batch.append(next(seed_iter))
            except StopIteration:
                seeds_batch.append(None)
        active_seeds = [s for s in seeds_batch if s is not None]
        if not active_seeds:
            break
        # World expects a flat list of seeds
        world.reset(seed=seeds_batch)

        # per-env trajectories of physical state
        traj = [[] for _ in range(num_envs)]
        steps_taken = np.zeros(num_envs, dtype=int)
        finished = np.array([s is None for s in seeds_batch])

        # record initial state
        for i in range(num_envs):
            if finished[i]:
                continue
            qpos = np.asarray(world.infos['qpos'][i]).reshape(-1)
            qvel = np.asarray(world.infos['qvel'][i]).reshape(-1)
            traj[i].append(np.concatenate([qpos, qvel]).astype(np.float32))

        for t in range(max_steps):
            world.step()
            for i in range(num_envs):
                if finished[i]:
                    continue
                qpos = np.asarray(world.infos['qpos'][i]).reshape(-1)
                qvel = np.asarray(world.infos['qvel'][i]).reshape(-1)
                traj[i].append(np.concatenate([qpos, qvel]).astype(np.float32))
                steps_taken[i] += 1
                if world.terminateds[i] or world.truncateds[i]:
                    finished[i] = True
            if finished.all():
                break

        for i, seed in enumerate(seeds_batch):
            if seed is None:
                continue
            ts = np.stack(traj[i], axis=0)                    # (T+1, 4)
            theta = ts[:, 1]
            V = compute_lyapunov_np(ts)
            dV = V[1:] - V[:-1]
            tail = max(1, len(theta) - 100)
            upright_frac = float(np.mean(np.abs(theta[tail:]) < 0.3))
            results.append(EpisodeResult(
                method=method,
                seed=int(seed),
                length=int(steps_taken[i]),
                succeeded=bool(upright_frac >= 0.8),
                upright_fraction=upright_frac,
                final_abs_theta=float(np.abs(theta[-1])),
                V_violation_rate=float(np.mean(dV > 0.0)),
                avg_dV=float(np.mean(dV)),
                final_V=float(V[-1]),
            ))
            completed += 1
            if completed >= n_episodes:
                break
        print(
            f'[{method}] {completed}/{n_episodes} '
            f'success_so_far={np.mean([r.succeeded for r in results]):.2f}'
        )

    world.close()
    return results


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True,
                   help='Path to weights_epoch_<N>.pt (config.json must be next to it).')
    p.add_argument('--dataset-name', type=str, default='cartpole_expert_worldmodel',
                   help='HDF5 dataset name (used for normalization stats).')
    p.add_argument('--cache-dir', type=str, default=None)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--horizon', type=int, default=15)
    p.add_argument('--candidates', type=int, default=512)
    p.add_argument('--lambda-v', type=float, default=1.0)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # Eval seeds chosen far from any seed used during data collection.
    # Collection used: base=7 + 1009*ep + env_idx, max ≈ 1.8M for 1800 episodes.
    p.add_argument('--eval-seed-base', type=int, default=10_000_000,
                   help='First eval seed (must be disjoint from training seeds).')
    p.add_argument('--methods', type=str, default='standard,lyapunov')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Where to write results.json. Defaults to <weights>/../mpc_eval/.')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    weights_path = Path(args.weights).expanduser().resolve()
    out_dir = Path(args.out_dir) if args.out_dir else weights_path.parent / 'mpc_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model from {weights_path}')
    model, cfg = load_model(weights_path)
    print(
        f'Model: encoder={cfg["encoder_scale"]} img={cfg["img_size"]}'
        f' patch={cfg["patch_size"]} action_dim={cfg["wm"]["action_dim"]}'
        f' state_dim={cfg["wm"]["state_dim"]}'
    )

    datasets_dir = get_cache_dir(args.cache_dir, sub_folder='datasets')
    dataset_path = Path(datasets_dir, f'{args.dataset_name}.h5')
    stats = compute_norm_stats(dataset_path)
    print(
        f'Stats — state_mean={stats.state_mean.flatten().tolist()}'
        f' state_std={stats.state_std.flatten().tolist()}'
    )

    eval_seeds = [args.eval_seed_base + i for i in range(args.episodes)]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    all_results: dict[str, list[EpisodeResult]] = {}
    for method in methods:
        print(f'\n=== Evaluating method: {method} ===')
        results = evaluate_method(
            method=method,
            model=model,
            stats=stats,
            cfg=cfg,
            n_episodes=args.episodes,
            eval_seeds=eval_seeds,
            horizon=args.horizon,
            n_candidates=args.candidates,
            lambda_v=args.lambda_v,
            max_steps=args.max_steps,
            num_envs=args.num_envs,
            device=args.device,
        )
        all_results[method] = results

    summary = {}
    for method, rs in all_results.items():
        ep_lens = [r.length for r in rs]
        succ = [r.succeeded for r in rs]
        viol = [r.V_violation_rate for r in rs]
        dV = [r.avg_dV for r in rs]
        upright = [r.upright_fraction for r in rs]
        finalV = [r.final_V for r in rs]
        summary[method] = {
            'n_episodes': len(rs),
            'success_rate': float(np.mean(succ)),
            'avg_episode_length': float(np.mean(ep_lens)),
            'avg_lyapunov_violation_rate': float(np.mean(viol)),
            'avg_dV': float(np.mean(dV)),
            'avg_upright_fraction': float(np.mean(upright)),
            'avg_final_V': float(np.mean(finalV)),
        }

    out_summary = {
        'config': {
            'weights': str(weights_path),
            'episodes': args.episodes,
            'horizon': args.horizon,
            'candidates': args.candidates,
            'lambda_v': args.lambda_v,
            'max_steps': args.max_steps,
            'eval_seed_base': args.eval_seed_base,
        },
        'summary': summary,
        'episodes': {
            method: [asdict(r) for r in rs]
            for method, rs in all_results.items()
        },
    }
    out_path = out_dir / 'mpc_eval_results.json'
    out_path.write_text(json.dumps(out_summary, indent=2))
    print(f'\nWrote {out_path}')

    print('\n=========================================')
    print('METHOD                     succ%   avg-len  V-violation%   avg-dV   final-V')
    print('-----------------------------------------')
    for method, s in summary.items():
        print(
            f'{method:25s} {s["success_rate"]*100:6.1f}'
            f' {s["avg_episode_length"]:8.1f}'
            f' {s["avg_lyapunov_violation_rate"]*100:13.2f}'
            f' {s["avg_dV"]:9.4f}'
            f' {s["avg_final_V"]:8.4f}'
        )


if __name__ == '__main__':
    main()

"""Latent-rollout observation probe + Lyapunov monitor.

During CEM/MPC rollout the predictor produces a sequence of predicted latents
{ẑ_1, ẑ_2, ...}. This module:
  - mean-pools the pixel slice of each latent,
  - passes it through the observation probe to get an estimated obs x̂,
  - evaluates a fixed Lyapunov candidate V(x̂) at each step,
  - computes the per-step descent rate δ_t = (V_t - V_{t+1}) / (V_t + eps).

Env-agnostic. The Lyapunov candidate is selected by cfg.lyapunov.mode:
  - "quadratic_goal":   V = w_pos * ||target_rel||^2 + w_vel * ||vel||^2
                            (+ w_ang_vel * ||ang_vel||^2 if ang_vel_idx given)
  - "pendulum_swingup": V = w_theta * (1 - cos_theta) + w_ang_vel * ang_vel^2
  - "cartpole_swingup": V = w_pos * pos^2 + w_theta * (1 - cos_theta)
                            + w_vel * vel^2 + w_ang_vel * ang_vel^2
  - "rocket":           V = ||p - p_pad||^2 + alpha * ||v||^2 + beta * ||omega||^2
"""
from __future__ import annotations

import torch


# -- Lyapunov candidates ------------------------------------------------------


def _quadratic_goal(obs, cfg):
    """V = w_pos * ||obs[target_rel_idx]||^2 + w_vel * ||obs[vel_idx]||^2
          (+ w_ang_vel * ||obs[ang_vel_idx]||^2 if given)"""
    target_rel = obs[..., cfg.target_rel_idx]
    vel = obs[..., cfg.vel_idx]
    V = cfg.get("w_pos", 1.0) * (target_rel ** 2).sum(-1) \
        + cfg.get("w_vel", 0.0) * (vel ** 2).sum(-1)
    if cfg.get("ang_vel_idx", None) is not None:
        ang_vel = obs[..., cfg.ang_vel_idx]
        V = V + cfg.get("w_ang_vel", 0.0) * (ang_vel ** 2).sum(-1)
    return V


def _pendulum_swingup(obs, cfg):
    cos_theta = obs[..., cfg.cos_theta_idx]
    ang_vel = obs[..., cfg.ang_vel_idx]
    return cfg.get("w_theta", 1.0) * (1.0 - cos_theta) \
        + cfg.get("w_ang_vel", 0.1) * ang_vel ** 2


def _cartpole_swingup(obs, cfg):
    pos = obs[..., cfg.pos_idx]
    vel = obs[..., cfg.vel_idx]
    cos_theta = obs[..., cfg.cos_theta_idx]
    ang_vel = obs[..., cfg.ang_vel_idx]
    return cfg.get("w_pos", 0.1) * pos ** 2 \
        + cfg.get("w_theta", 1.0) * (1.0 - cos_theta) \
        + cfg.get("w_vel", 0.05) * vel ** 2 \
        + cfg.get("w_ang_vel", 0.1) * ang_vel ** 2


def _rocket(obs, cfg):
    target_rel = obs[..., cfg.target_rel_idx]
    vel = obs[..., cfg.vel_idx]
    ang_vel = obs[..., cfg.ang_vel_idx]
    return (target_rel ** 2).sum(-1) \
        + cfg.get("alpha", 0.5) * (vel ** 2).sum(-1) \
        + cfg.get("beta", 0.1) * (ang_vel ** 2).sum(-1)


_LYAPUNOV = {
    "quadratic_goal":   _quadratic_goal,
    "pendulum_swingup": _pendulum_swingup,
    "cartpole_swingup": _cartpole_swingup,
    "rocket":           _rocket,
}


def lyapunov_value(obs, cfg):
    """Evaluate V(obs) given a config dict. obs: (..., obs_dim) tensor."""
    mode = cfg.get("mode", "rocket")
    if mode not in _LYAPUNOV:
        raise ValueError(f"Unknown Lyapunov mode: {mode}")
    return _LYAPUNOV[mode](obs, cfg)


# -- Rollout monitor ----------------------------------------------------------


@torch.no_grad()
def rollout_monitor(world_model, obs_probe, latents, lyap_cfg, eps: float = 1e-6):
    """Apply probe + Lyapunov to a rollout of predicted latents.

    Args:
      world_model : DINOWM-like object. Used only for its `pixels_embed` slicing
                    convention; here we assume `latents` is already the
                    pixel-dim slice after mean-pooling across patches.
                    Shape: (B, T, D_pix).
      obs_probe   : trained ObsProbe module (latent -> obs).
      latents     : (B, T, D_pix) mean-pooled pixel latents along the rollout.
      lyap_cfg    : OmegaConf / dict with Lyapunov params.

    Returns dict with:
      obs_hat         : (B, T, obs_dim)
      V               : (B, T)
      delta           : (B, T-1)        per-step descent rate
      violation_mask  : (B, T-1) bool   delta < 0
      alarm_mask      : (B, T-1) bool   delta < -descent_thresh for k consecutive
    """
    obs_probe.eval()
    obs_hat = obs_probe(latents)                       # (B, T, obs_dim)
    V = lyapunov_value(obs_hat, lyap_cfg)              # (B, T)
    delta = (V[:, :-1] - V[:, 1:]) / (V[:, :-1] + eps) # (B, T-1)
    violation_mask = delta < 0

    thresh = lyap_cfg.get("descent_thresh", 0.05)
    k = int(lyap_cfg.get("k_consecutive", 3))
    below = (delta < -thresh).float()
    # Alarm fires at t if below[t-k+1:t+1] are all 1.
    if below.shape[1] >= k:
        kernel = torch.ones(k, device=below.device)
        conv = torch.nn.functional.conv1d(
            below.unsqueeze(1), kernel.view(1, 1, k),
        ).squeeze(1)
        alarm = (conv >= k).bool()
        pad = torch.zeros(below.shape[0], k - 1, dtype=torch.bool, device=below.device)
        alarm_mask = torch.cat([pad, alarm], dim=1)
    else:
        alarm_mask = torch.zeros_like(below, dtype=torch.bool)

    return {
        "obs_hat": obs_hat,
        "V": V,
        "delta": delta,
        "violation_mask": violation_mask,
        "alarm_mask": alarm_mask,
    }


def probe_rollout_from_predictor(world_model, obs_probe, init_embedding,
                                 actions, lyap_cfg):
    """Roll the predictor forward and return probe+Lyapunov along the rollout.

    Args:
      world_model     : DINOWM instance (has .predict(), .encode()).
      obs_probe       : trained ObsProbe.
      init_embedding  : (B, T_hist, P, D) history of encoded embeddings.
      actions         : (B, H, action_embed_dim) action sequence to apply.
      lyap_cfg        : Lyapunov config.

    Returns the same dict as rollout_monitor, plus the raw predicted latents.
    """
    B = init_embedding.shape[0]
    H = actions.shape[1]
    pred_pooled = []
    embedding = init_embedding
    # DINOWM.predict outputs a full prediction; we extract the last frame's
    # pixel slice each step and append the *predicted* token as next history.
    for t in range(H):
        pred = world_model.predict(embedding)          # (B, T_hist, P, D)
        next_latent = pred[:, -1:, :, :]                # (B, 1, P, D)
        # Slide history forward.
        embedding = torch.cat([embedding[:, 1:], next_latent], dim=1)
        # Mean-pool patches, take pixel slice (assume pixels_dim is leading).
        pixels_dim = world_model.backbone.model.config.hidden_size \
            if hasattr(world_model.backbone, "model") \
            else next_latent.shape[-1]
        pooled = next_latent[..., :pixels_dim].mean(dim=2)  # (B, 1, D_pix)
        pred_pooled.append(pooled)
    latents = torch.cat(pred_pooled, dim=1)             # (B, H, D_pix)

    monitor = rollout_monitor(world_model, obs_probe, latents, lyap_cfg)
    monitor["latents"] = latents
    return monitor

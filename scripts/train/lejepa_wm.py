"""Rocket LeWorldModel-JEPA training with a parallel reconstruction probe.

The world-model part follows the LeWorldModel objective:

  (1) World model predictor (LeWM-style; stable_worldmodel/scripts/train/lewm.py):
         emb      = encode(pixels, proprio, action)     # full context-tiled
         ctx_emb  = emb[:, :-n_preds]
         tgt_emb  = emb[:, n_preds:]                    # encoder output shifted
         pred_emb = predict(ctx_emb)
         L_wm     = (pred_emb - tgt_emb).pow(2).mean() + lambda * SIGReg(pred_emb)

      The reference LeWorldModel implementation applies SIGReg to learned
      embeddings. Here DINOv2 is frozen and detached inside DINOWM.encode, so
      applying SIGReg to encoder outputs would have no trainable path. We
      apply it to the predicted pixel-latent slice instead, preserving the
      LeWM anti-collapse pressure on the trainable predictor.

  (2) Reconstruction probe (separate MLP, separate optimiser):
         L_probe = MSE( MLP( mean-pool( encode(I_t)_pix ).detach() ), obs_t )
      The probe's gradient cannot reach the world model: its input is
      detached and its parameters live in a different optimiser group.

The probe is used at rollout time as a control-Lyapunov monitor: for each
predicted next latent ẑ_{t+1} the MLP produces an observation estimate x̂_{t+1}
and a fixed Lyapunov candidate V(x̂) is evaluated to check descent.

Env-agnostic. Configure obs_dim, action_dim, and dataset_name per env
(rocket, cartpole, reacher, ...); see rocket_lejepa_config.yaml or
dmcontrol_{pendulum,cartpole,reacher}_config.yaml.
"""
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from datasets import load_from_disk
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm
from stable_worldmodel.sigreg import SIGReg
import stable_worldmodel.wm.dinowm as _swm_dinowm  # not re-exported at package top level
from _stepsdataset import StepsDataset as _LocalStepsDataset  # local shim; see _stepsdataset.py


DINO_PATCH_SIZE = 14


class ObsProbe(nn.Module):
    """MLP: mean-pooled DINO patch embedding -> environment observation.

    Trained independently from the world model on detached encoded features.
    """

    def __init__(self, embed_dim: int, obs_dim: int,
                 architecture: str = "mlp_2", hidden_dim: int = 256):
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(embed_dim, obs_dim)
        elif architecture == "mlp_2":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )
        elif architecture == "mlp_3":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, z):
        return self.net(z)


class ScalarProbe(nn.Module):
    """Small MLP head from pooled pixel latent to one scalar."""

    def __init__(self, embed_dim: int, architecture: str = "mlp_2", hidden_dim: int = 256):
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(embed_dim, 1)
        elif architecture == "mlp_2":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif architecture == "mlp_3":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DeltaProbe(nn.Module):
    """Predict normalized Lyapunov descent from consecutive pooled latents."""

    def __init__(self, embed_dim: int, architecture: str = "mlp_2", hidden_dim: int = 256):
        super().__init__()
        in_dim = 2 * embed_dim
        if architecture == "linear":
            self.net = nn.Linear(in_dim, 1)
        elif architecture == "mlp_2":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif architecture == "mlp_3":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, z_t, z_tp1):
        return self.net(torch.cat([z_t, z_tp1], dim=-1)).squeeze(-1)


def _dataset_path(cfg) -> Path:
    name = Path(str(cfg.dataset_name))
    if name.exists():
        return name
    cache = Path(cfg.get("cache_dir", None) or swm.data.get_cache_dir())
    return cache / str(cfg.dataset_name)


def _proprio_stats(cfg):
    ds = load_from_disk(str(_dataset_path(cfg)))
    proprio = np.asarray(ds["proprio"], dtype=np.float32)
    mean = torch.from_numpy(proprio.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(np.maximum(proprio.std(axis=0), 1e-6).astype(np.float32))
    return mean, std


def _raw_proprio(module, proprio: torch.Tensor) -> torch.Tensor:
    mean = getattr(module, "proprio_mean", None)
    std = getattr(module, "proprio_std", None)
    if mean is None or std is None:
        return proprio.float()
    return proprio.float() * std.to(proprio.device, proprio.dtype) + mean.to(proprio.device, proprio.dtype)


def _rocket_V(module, raw_state: torch.Tensor) -> torch.Tensor:
    pos_idx = list(getattr(module, "_v_pos_idx", [14, 15, 16]))
    vel_idx = list(getattr(module, "_v_vel_idx", [3, 4, 5]))
    omega_idx = list(getattr(module, "_v_omega_idx", [10, 11, 12]))
    alpha = float(getattr(module, "_v_alpha", 0.5))
    beta = float(getattr(module, "_v_beta", 0.1))
    p = raw_state[..., pos_idx]
    v = raw_state[..., vel_idx]
    omega = raw_state[..., omega_idx]
    return (p ** 2).sum(-1) + alpha * (v ** 2).sum(-1) + beta * (omega ** 2).sum(-1)


def _normalized_delta_from_V(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (V[..., :-1] - V[..., 1:]) / (V[..., :-1].abs() + eps)


def get_data(cfg):
    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], source=key, target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )

    def norm_col_transform(dataset, col="pixels"):
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)

    def norm_action_transform(dataset):
        data = dataset["action"][:]
        mean = data.mean(0).repeat(cfg.frameskip).unsqueeze(0)
        std = data.std(0).clamp_min(1e-6).repeat(cfg.frameskip).unsqueeze(0)
        return lambda x: (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)

    # swm.data.StepsDataset was removed upstream in favour of HDF5Dataset,
    # but our rocket_expert_all dataset is HuggingFace Arrow on disk, not
    # HDF5. The local shim in _stepsdataset.py restores the old class so
    # the training script works without upstream surgery or re-conversion.
    dataset = _LocalStepsDataset(
        cfg.dataset_name, num_steps=cfg.n_steps, frameskip=cfg.frameskip,
        transform=None, cache_dir=cfg.get("cache_dir", None),
    )
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE
    norm_action_transform = norm_action_transform(dataset.dataset)
    norm_proprio_transform = norm_col_transform(dataset.dataset, "proprio")

    transform = spt.data.transforms.Compose(
        *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size)
          for col in ["pixels"] for i in range(cfg.n_steps)],
        spt.data.transforms.WrapTorchTransform(norm_action_transform, source="action", target="action"),
        spt.data.transforms.WrapTorchTransform(norm_proprio_transform, source="proprio", target="proprio"),
    )
    dataset.transform = transform

    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split])
    logging.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
    train = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                       shuffle=True, pin_memory=True, drop_last=True)
    val = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                     shuffle=False, pin_memory=True)
    return spt.data.DataModule(train=train, val=val)


def forward(self, batch, stage):
    """LeWM-style next-latent prediction, SIGReg, and a decoupled probe.

    World-model predictor loss follows stable_worldmodel's LeWM recipe
    (scripts/train/lewm.py:96-99):

        tgt_emb   = emb[:, n_preds:]               # encoder output shifted
        pred_emb  = predict(ctx_emb, ctx_act)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg    = SIGReg(pred_emb)

    Single MSE over the full embedding (no per-slice sub-losses).
    The observation probe trains in parallel on detached features with a
    separate optimiser group, so its gradient cannot reach the predictor.
    """
    proprio_key = "proprio" if "proprio" in batch else None
    if proprio_key is not None:
        batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Encoder (frozen DINOv2) + proprio/action context -> embedding.
    # Optional train-time image noise: loosens the encoder's reliance on a
    # single visual distribution and absorbs sensor jitter. Off at eval.
    noise_std = float(getattr(self, "_image_noise_std", 0.0) or 0.0)
    if self.training and noise_std > 0.0 and "pixels" in batch:
        batch["pixels"] = batch["pixels"] + noise_std * torch.randn_like(batch["pixels"])

    batch = self.model.encode(
        batch, target="embed", pixels_key="pixels",
        proprio_key=proprio_key, action_key="action",
    )

    # --- (1) LeWM predictor loss: single MSE over full embedding --------------
    n_preds = int(getattr(self.model, "num_pred", 1))
    hist = int(getattr(self.model, "history_size", batch["embed"].shape[1]))
    pred_chunks = []
    tgt_chunks = []
    for target_idx in range(n_preds, int(batch["embed"].shape[1])):
        ctx_start = max(0, target_idx - hist)
        ctx_emb = batch["embed"][:, ctx_start:target_idx, :, :]
        pred_chunks.append(self.model.predict(ctx_emb)[:, -1:, :, :])
        tgt_chunks.append(batch["embed"][:, target_idx:target_idx + 1, :, :])
    pred_emb = torch.cat(pred_chunks, dim=1)
    tgt_emb = torch.cat(tgt_chunks, dim=1)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    batch["pred_loss"] = pred_loss

    # LeWorldModel anti-collapse regularizer. The reference implementation
    # regularizes encoded embeddings, but those are detached in DINO-WM. Using
    # the predicted pixel slice keeps the SIGReg gradient on trainable modules.
    sigreg_loss = torch.zeros((), device=pred_loss.device)
    sigreg = getattr(self, "sigreg", None)
    sigreg_weight = float(getattr(self, "_sigreg_weight", 0.0) or 0.0)
    if sigreg is not None and sigreg_weight > 0.0:
        pixels_dim = batch["pixels_embed"].shape[-1]
        pred_pixels = pred_emb[..., :pixels_dim]
        sigreg_loss = sigreg(pred_pixels)
        batch["sigreg_loss"] = sigreg_loss

    wm_loss = pred_loss + sigreg_weight * sigreg_loss

    # --- (2) Observation probe: detached latent -> observation ----------------
    probe = getattr(self.model, "obs_probe", None)
    v_head = getattr(self.model, "v_head", None)
    delta_head = getattr(self.model, "delta_head", None)
    probe_loss = torch.zeros((), device=wm_loss.device)
    if probe is not None and proprio_key is not None:
        pooled = batch["pixels_embed"].mean(dim=2).detach()   # (B, T, D_pix)
        obs_pred = probe(pooled)                              # (B, T, obs_dim)
        obs_target = batch[proprio_key].float()               # (B, T, obs_dim)
        probe_loss = F.mse_loss(obs_pred, obs_target)
        batch["probe_loss"] = probe_loss

    raw_state = _raw_proprio(self, batch[proprio_key]) if proprio_key is not None else None
    true_V = _rocket_V(self, raw_state) if raw_state is not None else None

    # --- (3) Direct V / DeltaV heads on teacher-forced encoded latents ---------
    # V is the safety quantity the shield actually needs. Predicting it directly
    # avoids asking a full state decoder to be uniformly accurate in dimensions
    # that do not affect the CLF.
    v_direct_loss = torch.zeros((), device=wm_loss.device)
    delta_direct_loss = torch.zeros((), device=wm_loss.device)
    v_direct_weight = float(getattr(self, "_v_direct_loss_weight", 0.0) or 0.0)
    delta_direct_weight = float(getattr(self, "_delta_direct_loss_weight", 0.0) or 0.0)
    if v_head is not None and true_V is not None and v_direct_weight > 0.0:
        pooled_detached = batch["pixels_embed"].mean(dim=2).detach()
        pred_logV = v_head(pooled_detached)
        target_logV = torch.log1p(true_V.clamp_min(0.0))
        v_direct_loss = F.mse_loss(pred_logV, target_logV)
        batch["v_direct_loss"] = v_direct_loss
    if (
        delta_head is not None
        and true_V is not None
        and delta_direct_weight > 0.0
        and batch["pixels_embed"].shape[1] > 1
    ):
        pooled_detached = batch["pixels_embed"].mean(dim=2).detach()
        target_delta = _normalized_delta_from_V(true_V, eps=float(getattr(self, "_v_eps", 1e-6)))
        pred_delta = delta_head(pooled_detached[:, :-1], pooled_detached[:, 1:])
        delta_direct_loss = F.mse_loss(pred_delta, target_delta)
        batch["delta_direct_loss"] = delta_direct_loss

    # Multi-step probe supervision on predicted rollouts. The planning failure
    # mode is that CEM optimizes decoded predicted latents whose V trajectory
    # does not match the executed rollout. This loss lets gradients reach the
    # predictor through the probe on z_{t+h}^{pred}, while the direct probe loss
    # above remains a detached teacher-forced decoder fit.
    probe_rollout_loss = torch.zeros((), device=wm_loss.device)
    v_rollout_loss = torch.zeros((), device=wm_loss.device)
    delta_rollout_loss = torch.zeros((), device=wm_loss.device)
    rollout_weight = float(getattr(self, "_probe_rollout_loss_weight", 0.0) or 0.0)
    v_rollout_weight = float(getattr(self, "_v_rollout_loss_weight", 0.0) or 0.0)
    delta_rollout_weight = float(getattr(self, "_delta_rollout_loss_weight", 0.0) or 0.0)
    rollout_horizons = list(getattr(self, "_probe_rollout_horizons", []) or [])
    if (
        (probe is not None or v_head is not None or delta_head is not None)
        and proprio_key is not None
        and (rollout_weight > 0.0 or v_rollout_weight > 0.0 or delta_rollout_weight > 0.0)
        and rollout_horizons
        and "pixels" in batch
        and "action" in batch
    ):
        rollout_batch_size = int(getattr(self, "_probe_rollout_batch_size", 0) or 0)
        hist = int(getattr(self.model, "history_size", 1))
        T = int(batch["pixels"].shape[1])
        valid_horizons = [int(h) for h in rollout_horizons if hist - 1 + int(h) < T]
        if valid_horizons:
            if rollout_batch_size > 0:
                n_roll = min(rollout_batch_size, int(batch["pixels"].shape[0]))
                rollout_pixels = batch["pixels"][:n_roll]
                rollout_proprio = batch[proprio_key][:n_roll]
                rollout_action = batch["action"][:n_roll]
            else:
                rollout_pixels = batch["pixels"]
                rollout_proprio = batch[proprio_key]
                rollout_action = batch["action"]
            rollout_info = self.model.rollout(
                {
                    "pixels": rollout_pixels[:, :hist],
                    proprio_key: rollout_proprio[:, :hist],
                },
                rollout_action[:, :T],
            )
            pred_pixels = rollout_info["predicted_pixels_embed"]
            state_horizon_losses = []
            v_horizon_losses = []
            delta_horizon_losses = []
            for h in valid_horizons:
                idx = hist - 1 + h
                pred_z = pred_pixels[:, idx].mean(dim=1)
                target_state = rollout_proprio[:, idx].float()
                target_raw = _raw_proprio(self, target_state)
                target_logV = torch.log1p(_rocket_V(self, target_raw).clamp_min(0.0))
                if probe is not None and rollout_weight > 0.0:
                    pred_state = probe(pred_z)
                    state_horizon_losses.append(F.mse_loss(pred_state, target_state))
                if v_head is not None and v_rollout_weight > 0.0:
                    v_horizon_losses.append(F.mse_loss(v_head(pred_z), target_logV))
                if (
                    delta_head is not None
                    and delta_rollout_weight > 0.0
                    and idx > 0
                ):
                    pred_z_prev = pred_pixels[:, idx - 1].mean(dim=1)
                    prev_raw = _raw_proprio(self, rollout_proprio[:, idx - 1].float())
                    V_pair = torch.stack([
                        _rocket_V(self, prev_raw),
                        _rocket_V(self, target_raw),
                    ], dim=-1)
                    target_delta = _normalized_delta_from_V(
                        V_pair, eps=float(getattr(self, "_v_eps", 1e-6))
                    ).squeeze(-1)
                    delta_horizon_losses.append(
                        F.mse_loss(delta_head(pred_z_prev, pred_z), target_delta)
                    )
            if state_horizon_losses:
                probe_rollout_loss = torch.stack(state_horizon_losses).mean()
                batch["probe_rollout_loss"] = probe_rollout_loss
            if v_horizon_losses:
                v_rollout_loss = torch.stack(v_horizon_losses).mean()
                batch["v_rollout_loss"] = v_rollout_loss
            if delta_horizon_losses:
                delta_rollout_loss = torch.stack(delta_horizon_losses).mean()
                batch["delta_rollout_loss"] = delta_rollout_loss

    # Sum is only for Lightning's autograd bookkeeping. The two summands touch
    # disjoint parameter groups for the direct probe loss. The optional rollout
    # probe term intentionally also updates the predictor, aligning planned
    # decoded state trajectories with future observations.
    batch["wm_loss"] = wm_loss
    batch["probe_total_loss"] = probe_loss + rollout_weight * probe_rollout_loss
    batch["v_total_loss"] = (
        v_direct_weight * v_direct_loss
        + v_rollout_weight * v_rollout_loss
        + delta_direct_weight * delta_direct_loss
        + delta_rollout_weight * delta_rollout_loss
    )
    batch["loss"] = wm_loss + batch["probe_total_loss"] + batch["v_total_loss"]

    prefix = "train/" if self.training else "val/"
    losses_dict = {f"{prefix}{k}": v.detach()
                   for k, v in batch.items() if "_loss" in k or k == "loss"}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return batch


def get_world_model(cfg):
    encoder = AutoModel.from_pretrained(cfg.get("encoder_id", "facebook/dinov2-small"))
    embedding_dim = encoder.config.hidden_size

    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    embedding_dim_total = (
        embedding_dim + cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim
    )
    logging.info(f"Patches: {num_patches}, Embedding dim total: {embedding_dim_total}")

    predictor = _swm_dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim_total,
        **cfg.predictor,
    )
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = _swm_dinowm.Embedder(
        in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim,
    )
    proprio_encoder = _swm_dinowm.Embedder(
        in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim,
    )

    world_model = _swm_dinowm.DINOWM(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor, action_encoder=action_encoder, proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size, num_pred=cfg.dinowm.num_preds, device="cuda",
    )

    # Parallel observation probe. Separate module, separate optimiser.
    probe_cfg = cfg.get("obs_probe", {}) or {}
    if probe_cfg.get("enabled", True):
        obs_dim = probe_cfg.get("obs_dim", cfg.dinowm.proprio_dim)
        world_model.obs_probe = ObsProbe(
            embed_dim=embedding_dim,
            obs_dim=obs_dim,
            architecture=probe_cfg.get("architecture", "mlp_2"),
            hidden_dim=probe_cfg.get("hidden_dim", 256),
        )
        logging.info(
            f"ObsProbe: arch={probe_cfg.get('architecture', 'mlp_2')} "
            f"embed_dim={embedding_dim} obs_dim={obs_dim} "
            f"hidden={probe_cfg.get('hidden_dim', 256)}"
        )

    v_cfg = cfg.get("v_head", {}) or {}
    if v_cfg.get("enabled", False):
        arch = v_cfg.get("architecture", "mlp_2")
        hidden = int(v_cfg.get("hidden_dim", 256))
        world_model.v_head = ScalarProbe(
            embed_dim=embedding_dim,
            architecture=arch,
            hidden_dim=hidden,
        )
        world_model.v_head_target = "log1p"
        logging.info(f"VHead: arch={arch} embed_dim={embedding_dim} hidden={hidden} target=log1p(V)")
        if v_cfg.get("delta_enabled", True):
            world_model.delta_head = DeltaProbe(
                embed_dim=embedding_dim,
                architecture=arch,
                hidden_dim=hidden,
            )
            world_model.delta_head_target = "normalized_descent"
            logging.info(
                f"DeltaHead: arch={arch} embed_dim={embedding_dim} hidden={hidden} "
                "target=normalized_descent"
            )

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    # World-model optimiser group: predictor + context projectors only.
    optim = {
        "wm_predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
        "wm_proprio_opt":   add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
        "wm_action_opt":    add_opt("model.action_encoder", cfg.action_encoder_lr),
    }
    # Probe optimiser group: disjoint from the world-model group.
    if getattr(world_model, "obs_probe", None) is not None:
        optim["probe_opt"] = add_opt(
            "model.obs_probe", probe_cfg.get("lr", cfg.predictor_lr),
        )
    if getattr(world_model, "v_head", None) is not None:
        optim["v_head_opt"] = add_opt(
            "model.v_head", v_cfg.get("lr", cfg.predictor_lr),
        )
    if getattr(world_model, "delta_head", None) is not None:
        optim["delta_head_opt"] = add_opt(
            "model.delta_head", v_cfg.get("lr", cfg.predictor_lr),
        )

    sigreg_module = None
    sigreg_weight = 0.0
    loss_cfg = cfg.get("loss", {}) or {}
    sigreg_cfg = loss_cfg.get("sigreg", {}) or cfg.get("lejepa_wm", {}) or {}
    if sigreg_cfg.get("enabled", True):
        sigreg_weight = float(
            sigreg_cfg.get("weight", sigreg_cfg.get("lambda_sigreg", 0.0)) or 0.0
        )
        if sigreg_weight > 0.0:
            kwargs = dict(sigreg_cfg.get("kwargs", {}) or {})
            # Backwards-compatible aliases from the previous rocket config.
            if "n_projections" not in kwargs and "n_projections" in sigreg_cfg:
                kwargs["n_projections"] = sigreg_cfg["n_projections"]
            if "n_frequencies" not in kwargs and "n_frequencies" in sigreg_cfg:
                kwargs["n_frequencies"] = sigreg_cfg["n_frequencies"]
            if "freq_min" not in kwargs and "freq_min" in sigreg_cfg:
                kwargs["freq_min"] = sigreg_cfg["freq_min"]
            if "freq_max" not in kwargs and "freq_max" in sigreg_cfg:
                kwargs["freq_max"] = sigreg_cfg["freq_max"]
            sigreg_module = SIGReg(latent_dim=embedding_dim, **kwargs)
            logging.info(
                f"SIGReg: weight={sigreg_weight} latent_dim={embedding_dim} "
                f"kwargs={kwargs}"
            )

    world_model = spt.Module(
        model=world_model,
        sigreg=sigreg_module,
        forward=forward,
        optim=optim,
    )
    p_mean, p_std = _proprio_stats(cfg)
    world_model.register_buffer("proprio_mean", p_mean, persistent=True)
    world_model.register_buffer("proprio_std", p_std, persistent=True)
    lyap_cfg = cfg.get("lyapunov", {}) or {}
    world_model._v_pos_idx = list(lyap_cfg.get("target_rel_idx", lyap_cfg.get("pos_idx", [14, 15, 16])))
    world_model._v_vel_idx = list(lyap_cfg.get("vel_idx", [3, 4, 5]))
    world_model._v_omega_idx = list(lyap_cfg.get("ang_vel_idx", lyap_cfg.get("omega_idx", [10, 11, 12])))
    world_model._v_alpha = float(lyap_cfg.get("alpha", 0.5))
    world_model._v_beta = float(lyap_cfg.get("beta", 0.1))
    world_model._v_eps = 1e-6
    world_model._sigreg_weight = sigreg_weight
    world_model._probe_rollout_horizons = list(probe_cfg.get("rollout_horizons", []) or [])
    world_model._probe_rollout_loss_weight = float(probe_cfg.get("rollout_loss_weight", 0.0) or 0.0)
    world_model._probe_rollout_batch_size = int(probe_cfg.get("rollout_batch_size", 0) or 0)
    if world_model._probe_rollout_loss_weight > 0.0 and world_model._probe_rollout_horizons:
        logging.info(
            f"ObsProbe rollout loss: horizons={world_model._probe_rollout_horizons} "
            f"weight={world_model._probe_rollout_loss_weight} "
            f"microbatch={world_model._probe_rollout_batch_size or 'full'}"
        )
    world_model._v_direct_loss_weight = float(v_cfg.get("direct_loss_weight", 0.0) or 0.0)
    world_model._v_rollout_loss_weight = float(v_cfg.get("rollout_loss_weight", 0.0) or 0.0)
    world_model._delta_direct_loss_weight = float(v_cfg.get("delta_direct_loss_weight", 0.0) or 0.0)
    world_model._delta_rollout_loss_weight = float(v_cfg.get("delta_rollout_loss_weight", 0.0) or 0.0)
    if getattr(world_model.model, "v_head", None) is not None:
        logging.info(
            "V/Delta losses: "
            f"v_direct={world_model._v_direct_loss_weight} "
            f"v_rollout={world_model._v_rollout_loss_weight} "
            f"delta_direct={world_model._delta_direct_loss_weight} "
            f"delta_rollout={world_model._delta_rollout_loss_weight}"
        )

    # Thread the image-noise augmentation std through to the forward pass.
    aug_cfg = cfg.get("augment", {}) or {}
    world_model._image_noise_std = float(aug_cfg.get("image_noise_std", 0.0) or 0.0)
    if world_model._image_noise_std > 0:
        logging.info(f"Train-time image noise std: {world_model._image_noise_std}")
    return world_model


def setup_pl_logger(cfg):
    if not cfg.wandb.get("enable", False):
        return None
    wandb_logger = WandbLogger(
        name=cfg.get("output_model_name", "dinowm_probe"),
        project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
        log_model=False,
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath, self.filename, self.epoch_interval = dirpath, filename, epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero and (trainer.current_epoch + 1) % self.epoch_interval == 0:
            output_path = Path(self.dirpath, f"{self.filename}_epoch_{trainer.current_epoch + 1}.ckpt")
            torch.save(pl_module, output_path)
            logging.info(f"Saved world model object to {output_path}")


@hydra.main(version_base=None, config_path="./", config_name="rocket_lejepa_config")
def run(cfg):
    from lightning.pytorch.callbacks import ModelCheckpoint
    import os
    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    ckpt_dir = os.environ.get(
        "WM_CKPT_DIR",
        f"/oscar/scratch/aiyer40/{cfg.output_model_name}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    dump_object_callback = ModelObjectCallBack(
        dirpath=ckpt_dir,
        filename=f"{cfg.output_model_name}_object",
        epoch_interval=int(os.environ.get("WM_OBJECT_EPOCH_INTERVAL", "10")),
    )
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=f"{cfg.output_model_name}_weights")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback, dump_object_callback],
        num_sanity_val_steps=1, logger=wandb_logger, enable_checkpointing=True,
    )
    resume_ckpt = os.environ.get("WM_RESUME_CKPT")
    if resume_ckpt and os.environ.get("WM_WARMSTART_NONSTRICT", "0") == "1":
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        current = world_model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in current and tuple(current[k].shape) == tuple(v.shape)
        }
        skipped = len(state) - len(compatible)
        world_model.load_state_dict(compatible, strict=False)
        logging.info(
            f"Warm-started non-strict from {resume_ckpt}: "
            f"loaded {len(compatible)} tensors, skipped {skipped}"
        )
        resume_ckpt = None

    manager = spt.Manager(trainer=trainer, module=world_model, data=data,
                          ckpt_path=resume_ckpt)
    manager()


if __name__ == "__main__":
    run()

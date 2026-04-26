"""Cart-pole world-model trainer.

Encoder (HF ViT) + LeWM latent dynamics + MLP state decoder, trained jointly:

    L = ||F(z_t, a_t) - z_{t+1}||^2 + lambda * ||D(z_t) - s_t||^2

Built on the LeWM scaffold from stable_worldmodel/wm/lewm. The existing
scripts/train/lewm.py imports a few names that aren't actually defined in
the installed package (ARPredictor, JEPA, SIGReg from lewm.module); this
script avoids them and uses what's actually there: LeWM + Predictor.
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from torch import nn
from torchvision.transforms import v2 as tv_v2
from transformers import ViTConfig, ViTModel

from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from stable_worldmodel.wm.utils import save_pretrained


# --------------------------------------------------------------------------- #
#  Encoder / model construction                                               #
# --------------------------------------------------------------------------- #

VIT_SIZES = {
    'tiny': dict(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    ),
    'small': dict(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
    ),
    'base': dict(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ),
}


def make_vit_encoder(scale: str, image_size: int, patch_size: int) -> ViTModel:
    """Random-init HF ViT used as the visual encoder."""
    if scale not in VIT_SIZES:
        raise ValueError(
            f'unknown encoder_scale={scale!r} (choose from {list(VIT_SIZES)})'
        )
    cfg = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        **VIT_SIZES[scale],
    )
    return ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)


class CartpoleLeWM(LeWM):
    """LeWM augmented with an MLP head that decodes latent -> physical state."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        state_decoder: nn.Module,
        projector=None,
        pred_proj=None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.state_decoder = state_decoder

    def decode_state(self, emb: torch.Tensor) -> torch.Tensor:
        """Apply the MLP head per-timestep. emb: (B, T, D) -> (B, T, state_dim)."""
        b, t, d = emb.shape
        out = self.state_decoder(emb.reshape(b * t, d))
        return out.reshape(b, t, -1)


# --------------------------------------------------------------------------- #
#  Data pipeline                                                              #
# --------------------------------------------------------------------------- #


def get_img_preprocessor(source: str, target: str, img_size: int):
    """Build a (B, T, C, H, W) image transform pipeline.

    spt 0.1.2's `dt.transforms.Resize` is broken against the installed
    torchvision (calls `self.transform(...)` which v2.Resize doesn't expose).
    We bypass it by wrapping `v2.Resize` directly with `WrapTorchTransform`,
    which routes through `v2.Lambda.__call__` and works correctly.
    """
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.WrapTorchTransform(
        tv_v2.Resize(img_size, antialias=True),
        source=source,
        target=target,
    )
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone().clamp_min(1e-6)

    def norm_fn(x):
        return ((x - mean) / std).float()

    return dt.transforms.WrapTorchTransform(
        norm_fn, source=source, target=target
    )


def state_4_to_5(s: torch.Tensor) -> torch.Tensor:
    """Map raw cart-pole state (..., [x, theta, x_dot, theta_dot]) to the
    5-dim cos/sin parameterisation (..., [x, x_dot, cos theta, sin theta, theta_dot]).

    Eliminates the ±pi wraparound discontinuity that prevents the decoder
    from learning theta under plain MSE.
    """
    if s.shape[-1] != 4:
        raise ValueError(f'expected last dim = 4, got {tuple(s.shape)}')
    x, theta, x_dot, theta_dot = s.unbind(-1)
    return torch.stack(
        [x, x_dot, torch.cos(theta), torch.sin(theta), theta_dot], dim=-1,
    )


def get_state_cossin_transform(dataset, source: str = 'state', target: str = 'state'):
    """Compose: raw 4-dim state -> 5-dim cos/sin -> z-score (5-dim stats)."""
    raw = torch.from_numpy(np.array(dataset.get_col_data(source))).float()
    raw = raw[~torch.isnan(raw).any(dim=1)]
    cs = state_4_to_5(raw)                                      # (N, 5)
    mean = cs.mean(0, keepdim=True).clone()
    std = cs.std(0, keepdim=True).clone().clamp_min(1e-6)

    def fn(x):
        x = torch.as_tensor(x).float()
        return ((state_4_to_5(x) - mean) / std).float()

    return dt.transforms.WrapTorchTransform(fn, source=source, target=target)


# --------------------------------------------------------------------------- #
#  Lightning callback for periodic checkpointing                              #
# --------------------------------------------------------------------------- #


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            self._save(pl_module.model, trainer.current_epoch + 1)
        if (trainer.current_epoch + 1) == trainer.max_epochs:
            self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


# --------------------------------------------------------------------------- #
#  Forward pass: encode, predict next latent, decode current state            #
# --------------------------------------------------------------------------- #


def cartpole_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.state_weight

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)
    emb = output['emb']            # (B, T, D)
    act_emb = output['act_emb']    # (B, T, D)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    dyn_loss = (pred_emb - tgt_emb).pow(2).mean()

    pred_state = self.model.decode_state(ctx_emb)
    tgt_state = batch['state'][:, :ctx_len].float()
    state_loss = (pred_state - tgt_state).pow(2).mean()

    total = dyn_loss + lambd * state_loss
    output['dyn_loss'] = dyn_loss
    output['state_loss'] = state_loss
    output['loss'] = total

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


# --------------------------------------------------------------------------- #
#  Hydra entry point                                                          #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path='./config', config_name='cartpole_wm')
def run(cfg):
    # ------- dataset -------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    state_repr = str(cfg.wm.get('state_repr', 'cossin')).lower()
    if state_repr not in ('raw', 'cossin'):
        raise ValueError(f"wm.state_repr must be 'raw' or 'cossin'; got {state_repr!r}")

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            if col == 'state' and state_repr == 'cossin':
                transforms.append(
                    get_state_cossin_transform(dataset, source=col, target=col)
                )
                cfg.wm.state_dim = 5
            else:
                transforms.append(get_column_normalizer(dataset, col, col))
                setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    # ------- model -------
    encoder = make_vit_encoder(
        scale=cfg.encoder_scale,
        image_size=cfg.img_size,
        patch_size=cfg.patch_size,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get('embed_dim', hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = Predictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    proj_norm = str(cfg.projector.norm).lower()
    proj_norm_map = {'bn': nn.BatchNorm1d, 'ln': nn.LayerNorm, 'none': None}
    if proj_norm not in proj_norm_map:
        raise ValueError(
            f"projector.norm must be one of {list(proj_norm_map)}; got {proj_norm!r}"
        )
    proj_norm_fn = proj_norm_map[proj_norm]

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=proj_norm_fn,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=proj_norm_fn,
    )

    state_decoder = nn.Sequential(
        nn.Linear(hidden_dim, cfg.decoder.hidden_dim),
        nn.GELU(),
        nn.LayerNorm(cfg.decoder.hidden_dim),
        nn.Linear(cfg.decoder.hidden_dim, cfg.decoder.hidden_dim),
        nn.GELU(),
        nn.LayerNorm(cfg.decoder.hidden_dim),
        nn.Linear(cfg.decoder.hidden_dim, cfg.wm.state_dim),
    )

    world_model = CartpoleLeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        state_decoder=state_decoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        forward=partial(cartpole_forward, cfg=cfg),
        optim=optimizers,
    )

    # ------- training -------
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name, cfg=cfg, epoch_interval=1
            )
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()

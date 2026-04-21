"""LeWorldModel-JEPA training: DINO-WM + SIGReg loss in latent space.

Identical pipeline to scripts/train/dinowm.py except the loss adds a
characteristic-function (SIGReg) regularizer on the predicted next-frame
embeddings. This is the architecture for the rocket_jepa CoRL submission.

Loss = MSE(pred pixels-embed, target pixels-embed) +
       MSE(pred proprio-embed, target proprio-embed) +
       lambda_sigreg * SIGReg(pred pixels-embed)
"""
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm
from stable_worldmodel.sigreg import SIGReg


DINO_PATCH_SIZE = 14


# Module-level handle to the SIGReg loss; populated in get_world_model().
# Kept module-level because spt.Module.forward is a static-style function that
# receives only (self, batch, stage) and self refers to the wrapped Module.
_SIGREG: SIGReg | None = None
_LAMBDA_SIGREG: float = 0.02


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
        return lambda x: (x - mean) / std

    dataset = swm.data.StepsDataset(
        cfg.dataset_name, num_steps=cfg.n_steps, frameskip=cfg.frameskip,
        transform=None, cache_dir=cfg.get("cache_dir", None),
    )
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE
    norm_action_transform = norm_col_transform(dataset.dataset, "action")
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
    """LeJEPA-WM forward: invariance MSE + proprio MSE + SIGReg on predicted latents."""
    proprio_key = "proprio" if "proprio" in batch else None
    if proprio_key is not None:
        batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    batch = self.model.encode(
        batch, target="embed", pixels_key="pixels",
        proprio_key=proprio_key, action_key="action",
    )

    embedding = batch["embed"][:, :-1, :, :]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch["embed"][:, 1:, :, :]

    pixels_dim = batch["pixels_embed"].shape[-1]
    pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
    loss = pixels_loss
    batch["pixels_loss"] = pixels_loss

    if proprio_key is not None:
        proprio_dim = batch["proprio_embed"].shape[-1]
        proprio_loss = F.mse_loss(
            pred_embedding[..., pixels_dim:pixels_dim + proprio_dim],
            target_embedding[..., pixels_dim:pixels_dim + proprio_dim].detach(),
        )
        loss = loss + proprio_loss
        batch["proprio_loss"] = proprio_loss

    # SIGReg on predicted latent pixels-embeddings (flatten patches into batch dim)
    sigreg_loss = _SIGREG(pred_embedding[..., :pixels_dim])
    batch["sigreg_loss"] = sigreg_loss
    loss = loss + _LAMBDA_SIGREG * sigreg_loss

    batch["loss"] = loss
    prefix = "train/" if self.training else "val/"
    losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return batch


def get_world_model(cfg):
    global _SIGREG, _LAMBDA_SIGREG
    encoder = AutoModel.from_pretrained("facebook/dinov2-small")
    embedding_dim = encoder.config.hidden_size

    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    embedding_dim_total = (
        embedding_dim + cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim
    )
    logging.info(f"Patches: {num_patches}, Embedding dim total: {embedding_dim_total}")

    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim_total,
        **cfg.predictor,
    )
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)

    world_model = swm.wm.DINOWM(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor, action_encoder=action_encoder, proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size, num_pred=cfg.dinowm.num_preds, device="cuda",
    )

    # SIGReg dimension is the DINO pixels embed dim only (excludes proprio/action concat).
    _SIGREG = SIGReg(
        latent_dim=encoder.config.hidden_size,
        n_projections=cfg.lejepa_wm.get("n_projections", 1024),
        n_frequencies=cfg.lejepa_wm.get("n_frequencies", 17),
    ).to("cuda")
    _LAMBDA_SIGREG = float(cfg.lejepa_wm.get("lambda_sigreg", 0.02))
    logging.info(f"SIGReg: dim={encoder.config.hidden_size} lambda={_LAMBDA_SIGREG}")

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    world_model = spt.Module(
        model=world_model, forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    return world_model


def setup_pl_logger(cfg):
    if not cfg.wandb.get("enable", False):
        return None
    wandb_logger = WandbLogger(
        name="lejepa_wm", project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
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

    # Save checkpoints to /oscar/scratch (quota plentiful) not $STABLEWM_HOME
    # (in /users which has tight quota and crashed previous runs).
    ckpt_dir = os.environ.get(
        "WM_CKPT_DIR",
        f"/oscar/scratch/aiyer40/{cfg.output_model_name}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    dump_object_callback = ModelObjectCallBack(
        dirpath=ckpt_dir, filename=f"{cfg.output_model_name}_object", epoch_interval=10,
    )
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=f"{cfg.output_model_name}_weights")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback, dump_object_callback],
        num_sanity_val_steps=1, logger=wandb_logger, enable_checkpointing=True,
    )
    resume_ckpt = os.environ.get("WM_RESUME_CKPT")
    manager = spt.Manager(trainer=trainer, module=world_model, data=data,
                          ckpt_path=resume_ckpt)
    manager()


if __name__ == "__main__":
    run()

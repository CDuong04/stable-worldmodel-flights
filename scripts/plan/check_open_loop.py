"""Open-loop WM rollout sanity check.

Feed a recorded expert trajectory through LeWM.rollout and compare the
predicted latent at each future step to the true encoder(obs) latent.

Usage:
    python scripts/plan/check_open_loop.py \
        --ckpt /oscar/data/jpober/cduong5/models/lewm-rocket/weights_epoch_100.pt \
        --h5 /oscar/data/jpober/cduong5/swm_cache/datasets/rocket_expert.h5 \
        --episode 0 --start 0 --horizon 10
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import stable_worldmodel as swm
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP
from transformers import ViTConfig, ViTModel

VIT_SIZES = {
    "tiny":  {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3,  "intermediate_size": 768},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6,  "intermediate_size": 1536},
    "base":  {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
}


def build_model(cfg: dict) -> LeWM:
    vit_cfg = ViTConfig(
        image_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        num_channels=3,
        **VIT_SIZES[cfg["encoder_scale"]],
    )
    encoder = ViTModel(vit_cfg, add_pooling_layer=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg["wm"].get("embed_dim", hidden_dim)
    effective_act_dim = cfg["data"]["dataset"]["frameskip"] * cfg["wm"]["action_dim"]
    predictor = Predictor(
        num_frames=cfg["wm"]["history_size"],
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg["predictor"],
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    return LeWM(
        encoder=encoder, predictor=predictor,
        action_encoder=action_encoder, projector=projector, pred_proj=pred_proj,
    )


def load_model(ckpt_path: Path) -> LeWM:
    cfg_path = ckpt_path.parent / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing: {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")
    return model


def load_window(h5_path: Path, ep_idx: int, start: int, total_len: int):
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        assert start + total_len <= ep_len[ep_idx], (
            f"window overflows episode (ep_len={ep_len[ep_idx]}, need {start + total_len})"
        )
        g0 = int(ep_offset[ep_idx]) + start
        g1 = g0 + total_len
        pixels = f["pixels"][g0:g1]
        actions = f["action"][g0:g1]
    return pixels, actions


def preprocess_pixels(pixels_np, img_size=224):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    stats = spt.data.dataset_stats.ImageNet
    t = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**stats),
        transforms.Resize(size=img_size),
    ])
    out = torch.stack([t(p) for p in pixels_np])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    total_len = args.history + args.horizon
    pixels_np, actions_np = load_window(
        Path(args.h5), args.episode, args.start, total_len
    )
    print(f"loaded pixels {pixels_np.shape} actions {actions_np.shape}")

    pixels = preprocess_pixels(pixels_np, args.img_size).to(device)  # (T, 3, H, W)
    actions = torch.from_numpy(actions_np).float().to(device)  # (T, A)

    model = load_model(Path(args.ckpt))
    model = model.to(device).eval()
    model.requires_grad_(False)

    # Encode all frames (true latents for comparison)
    with torch.no_grad():
        enc_out = model.encoder(pixels, interpolate_pos_encoding=True)
        true_emb_raw = enc_out.last_hidden_state[:, 0]  # (T, hidden)
        true_emb = model.projector(true_emb_raw)  # (T, D) — match rollout output space

    # Build rollout inputs: pixels (B=1, S=1, HS, C, H, W), actions (B=1, S=1, T, A)
    B, S = 1, 1
    HS = args.history
    pix_in = pixels[:HS].unsqueeze(0).unsqueeze(0)  # (1, 1, HS, C, H, W)
    act_in = actions.unsqueeze(0).unsqueeze(0)  # (1, 1, T, A)
    info = {"pixels": pix_in, "action": act_in}

    with torch.no_grad():
        out = model.rollout(info, act_in, history_size=HS)
    pred_emb = out["predicted_emb"].squeeze(0).squeeze(0)  # (T+1, D) or (T, D)
    print(f"predicted_emb shape: {tuple(pred_emb.shape)}")
    print(f"true_emb shape: {tuple(true_emb.shape)}")

    # Compare predicted vs true at horizon steps (after the context window)
    # Slice aligned tail: last `horizon` steps of both
    T = pred_emb.shape[0]
    T_true = true_emb.shape[0]
    n = min(T - HS, T_true - HS, args.horizon)
    preds = pred_emb[HS : HS + n]
    trues = true_emb[HS : HS + n]

    cos = F.cosine_similarity(preds, trues, dim=-1).cpu().numpy()
    l2 = torch.linalg.norm(preds - trues, dim=-1).cpu().numpy()
    print("\nper-step metrics (t = HS .. HS+n-1):")
    print(f"{'step':>5}  {'cos_sim':>10}  {'l2':>10}")
    for i, (c, l) in enumerate(zip(cos, l2)):
        print(f"{i+1:>5}  {c:>10.4f}  {l:>10.4f}")
    print(f"\nmean cosine: {cos.mean():.4f}   mean L2: {l2.mean():.4f}")


if __name__ == "__main__":
    main()

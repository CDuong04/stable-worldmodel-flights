"""Diagnostic for the rocket_expert_union action-fold mismatch.

Both `lejepa-wm-union` (1889416) and `dinowm-union` (1889417) die at
dinowm.py:81 with `expected input[32, 14, 1] to have 14 channels, but got
7 channels instead`. That means the dataset emits raw 7-D actions when
the model expects frameskip*action_dim=14 per macro-step. This script
isolates which link in the chain is broken.
"""
from __future__ import annotations
import sys
import importlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import hydra
from omegaconf import OmegaConf

# Make the local shim importable regardless of cwd.
SCRIPTS_TRAIN = Path(__file__).resolve().parent
if str(SCRIPTS_TRAIN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TRAIN))

import _stepsdataset as shim_module
StepsDataset = shim_module.StepsDataset


@hydra.main(config_path="../train", config_name="rocket_lejepa_config", version_base="1.3")
def main(cfg):
    print("=" * 70)
    print("[1] Hydra config — what reached runtime?")
    print("=" * 70)
    print(f"  cfg.frameskip            = {cfg.frameskip!r}  (type {type(cfg.frameskip).__name__})")
    print(f"  cfg.n_steps              = {cfg.n_steps!r}")
    print(f"  cfg.dinowm.action_dim    = {cfg.dinowm.action_dim!r}")
    print(f"  cfg.dataset_name         = {cfg.dataset_name!r}")
    print(f"  cfg.get('cache_dir')     = {cfg.get('cache_dir', None)!r}")

    print("\n" + "=" * 70)
    print("[2] Imported shim module — is the local one winning?")
    print("=" * 70)
    print(f"  StepsDataset.__module__  = {StepsDataset.__module__}")
    print(f"  StepsDataset.__file__    = {shim_module.__file__}")
    # Check if upstream also defines one
    try:
        from stable_worldmodel.data import StepsDataset as UpstreamSD
        print(f"  upstream StepsDataset    = {UpstreamSD.__module__}.{UpstreamSD.__name__}")
        print(f"  same object as shim?     = {StepsDataset is UpstreamSD}")
    except Exception as exc:
        print(f"  upstream StepsDataset    = NOT IMPORTABLE ({type(exc).__name__})")

    print("\n" + "=" * 70)
    print("[3] Build the shim and inspect raw HF dataset row")
    print("=" * 70)
    sd = StepsDataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        cache_dir=cfg.get("cache_dir", None),
        transform=None,
    )
    raw = sd.dataset
    print(f"  raw HF dataset class     = {type(raw).__name__}")
    print(f"  raw len(dataset)         = {len(raw)}")
    print(f"  raw column_names         = {raw.column_names}")
    row0 = raw[0]
    a0 = row0["action"]
    print(f"  row0['action'] type      = {type(a0).__name__}")
    if hasattr(a0, "shape"):
        print(f"  row0['action'].shape     = {tuple(a0.shape)}")
    arr = np.asarray(a0)
    print(f"  np.asarray(...).shape    = {arr.shape}")
    print(f"  flatten len              = {arr.reshape(-1).shape[0]}")
    print(f"  shim self.frameskip      = {sd.frameskip!r}  (type {type(sd.frameskip).__name__})")
    print(f"  shim self.num_steps      = {sd.num_steps!r}")

    print("\n" + "=" * 70)
    print("[4] Shim __getitem__(0) — does the fold produce 14-D macro-actions?")
    print("=" * 70)
    item = sd[0]
    print(f"  item.keys()              = {sorted(item.keys())}")
    if "action" in item:
        act = item["action"]
        print(f"  item['action'] type      = {type(act).__name__}")
        if hasattr(act, "shape"):
            print(f"  item['action'].shape     = {tuple(act.shape)}")
        expected = (cfg.n_steps, cfg.frameskip * cfg.dinowm.action_dim)
        ok = tuple(act.shape) == expected if hasattr(act, "shape") else False
        print(f"  expected shape           = {expected}")
        print(f"  match?                   = {ok}")

    print("\n" + "=" * 70)
    print("[5] DataLoader over full shim, batch_size=4")
    print("=" * 70)
    loader = DataLoader(sd, batch_size=4, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    if "action" in batch:
        b_act = batch["action"]
        print(f"  batch['action'].shape    = {tuple(b_act.shape)}")
        expected_b = (4, cfg.n_steps, cfg.frameskip * cfg.dinowm.action_dim)
        print(f"  expected batch shape     = {expected_b}")
        print(f"  match?                   = {tuple(b_act.shape) == expected_b}")

    print("\n" + "=" * 70)
    print("[6] DataLoader over Subset(shim, [0..7]) — does Subset preserve fold?")
    print("=" * 70)
    sub = Subset(sd, list(range(8)))
    sub_loader = DataLoader(sub, batch_size=4, num_workers=0, shuffle=False)
    sub_batch = next(iter(sub_loader))
    if "action" in sub_batch:
        sb_act = sub_batch["action"]
        print(f"  subset batch shape       = {tuple(sb_act.shape)}")
        print(f"  match?                   = {tuple(sb_act.shape) == (4, cfg.n_steps, cfg.frameskip * cfg.dinowm.action_dim)}")

    print("\n" + "=" * 70)
    print("[7] Inspect the fold-branch source lines once more (sanity)")
    print("=" * 70)
    src = Path(shim_module.__file__).read_text().splitlines()
    for i, line in enumerate(src[100:125], start=101):
        print(f"  {i:>3}: {line}")

    print("\nDone.")


if __name__ == "__main__":
    main()

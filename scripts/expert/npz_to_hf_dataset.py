"""Convert per-rollout .npz files (from lqr_experts.py / collect_from_sac.py)
into the HuggingFace Arrow dataset format that lejepa_wm.py expects.

Expected output schema (matches rocket_expert_union):
    pixels:      string         (path to image file on disk)
    proprio:     Sequence(float64)
    action:      Sequence(float64)
    episode_idx: int64
    step_idx:    int64

Image files are saved to <out_dir>/images/<episode>_<step>.png.
The Arrow dataset is saved via dataset.save_to_disk(<out_dir>/<name>).

Usage:
    python scripts/expert/npz_to_hf_dataset.py \\
        --in-dir data/pendulum_swingup_expert \\
        --out-dir data/expert_trajectories \\
        --name pendulum_swingup_expert
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with rollout_*.npz files")
    ap.add_argument("--out-dir", default="data/expert_trajectories",
                    help="parent dir for HF arrow datasets")
    ap.add_argument("--name", required=True,
                    help="dataset name (e.g., pendulum_swingup_expert)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / "images" / args.name
    img_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("rollout_*.npz"))
    print(f"[convert] found {len(files)} rollouts in {in_dir}")
    if not files:
        print("[convert] nothing to convert; aborting.")
        return

    rows_pixels = []
    rows_proprio = []
    rows_action = []
    rows_episode = []
    rows_step = []

    total_steps = 0
    for ep_idx, f in enumerate(files):
        d = np.load(f)
        pix = d["pixels"]      # (T, H, W, 3) uint8
        pro = d["proprio"]     # (T, S)
        act = d["action"]      # (T-1, A) (or T, depending on collection script)

        # Align lengths: each row needs (image_t, proprio_t, action_t).  We use
        # action[t] = action taken AFTER image_t (length T-1).  Drop the final
        # image (no action follows it) to keep arrays aligned.
        T = len(pix)
        if len(act) < T:
            T = len(act)
        for t in range(T):
            img_path = img_dir / f"ep{ep_idx:04d}_st{t:04d}.png"
            if not img_path.exists():
                Image.fromarray(pix[t]).save(img_path, optimize=True)
            # Use ABSOLUTE path so the dataset works regardless of CWD at train time.
            rows_pixels.append(str(img_path.resolve()))
            rows_proprio.append(pro[t].astype(np.float64).tolist())
            rows_action.append(act[t].astype(np.float64).tolist())
            rows_episode.append(int(ep_idx))
            rows_step.append(int(t))
        total_steps += T
        if (ep_idx + 1) % 50 == 0:
            print(f"[convert] {ep_idx+1}/{len(files)} episodes, {total_steps} steps")

    print(f"[convert] writing {total_steps} rows to HF dataset...")
    features = Features({
        "pixels":      Value("string"),
        "proprio":     Sequence(feature=Value("float64")),
        "action":      Sequence(feature=Value("float64")),
        "episode_idx": Value("int64"),
        "step_idx":    Value("int64"),
    })
    ds = Dataset.from_dict(
        {
            "pixels":      rows_pixels,
            "proprio":     rows_proprio,
            "action":      rows_action,
            "episode_idx": rows_episode,
            "step_idx":    rows_step,
        },
        features=features,
    )
    out_path = out_root / args.name
    ds.save_to_disk(str(out_path))
    print(f"[convert] wrote dataset to {out_path}")
    print(f"[convert] n_rows = {len(ds)}, n_episodes = {len(files)}")


if __name__ == "__main__":
    main()

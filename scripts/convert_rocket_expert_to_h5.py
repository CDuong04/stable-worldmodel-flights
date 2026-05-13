"""Convert HuggingFace Arrow rocket-expert dataset to HDF5 for HDF5Dataset."""

import argparse
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/oscar/data/jpober/cduong5/expert_trajectories/rocket_expert_default_hf",
    )
    parser.add_argument(
        "--dst",
        default="/oscar/data/jpober/cduong5/swm_cache/datasets/rocket_expert.h5",
    )
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Arrow dataset from {args.src}")
    ds = load_from_disk(args.src)
    print(f"Loaded {len(ds)} rows with features: {list(ds.features)}")

    print("Sorting by (episode, step)")
    ds = ds.sort(["episode", "step"])

    n = len(ds)
    eps = np.asarray(ds["episode"])
    change = np.concatenate([[True], eps[1:] != eps[:-1]])
    ep_offset = np.flatnonzero(change).astype(np.int64)
    ep_len = np.diff(np.append(ep_offset, n)).astype(np.int32)
    print(f"Episodes: {ep_len.shape[0]}, total steps: {n}, mean steps/ep: {ep_len.mean():.1f}")

    sample = ds[0]
    img0 = np.asarray(sample["pixels"], dtype=np.uint8)
    h, w = img0.shape[:2]
    c = 3  # always drop alpha if present
    proprio_dim = len(sample["proprio"])
    action_dim = len(sample["action"])
    print(f"pixels: ({n},{h},{w},{c}) uint8 (src channels={img0.shape[-1]}), proprio: ({n},{proprio_dim}), action: ({n},{action_dim})")

    blosc = hdf5plugin.Blosc(
        cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
    )

    with h5py.File(dst, "w") as f:
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_len", data=ep_len)
        pix = f.create_dataset(
            "pixels",
            shape=(n, h, w, c),
            dtype="uint8",
            chunks=(min(100, n), h, w, c),
            **blosc,
        )
        pro = f.create_dataset(
            "proprio",
            shape=(n, proprio_dim),
            dtype="float64",
            chunks=(min(100, n), proprio_dim),
            **blosc,
        )
        act = f.create_dataset(
            "action",
            shape=(n, action_dim),
            dtype="float64",
            chunks=(min(100, n), action_dim),
            **blosc,
        )

        for i in tqdm(range(0, n, args.chunk), desc="writing"):
            batch = ds[i : i + args.chunk]
            k = len(batch["pixels"])
            pix[i : i + k] = np.stack(
                [np.asarray(img, dtype=np.uint8)[..., :3] for img in batch["pixels"]]
            )
            pro[i : i + k] = np.asarray(batch["proprio"], dtype=np.float64)
            act[i : i + k] = np.asarray(batch["action"], dtype=np.float64)

    print(f"Wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

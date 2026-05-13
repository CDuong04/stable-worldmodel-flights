"""Add step_idx and ep_idx columns to rocket_expert.h5 (needed by eval_wm.py)."""

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

H5 = Path("/oscar/data/jpober/cduong5/swm_cache/datasets/rocket_expert.h5")


def main():
    with h5py.File(H5, "a") as f:
        existing = list(f.keys())
        print(f"existing keys: {existing}")
        n = int(f["ep_len"][:].sum())
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        assert n == f["pixels"].shape[0], "ep_len does not sum to pixels rows"

        ep_idx = np.repeat(np.arange(len(ep_len), dtype=np.int64), ep_len)
        step_idx = np.concatenate([np.arange(L, dtype=np.int64) for L in ep_len])
        assert ep_idx.shape == step_idx.shape == (n,)

        blosc = hdf5plugin.Blosc(
            cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
        )
        for name, arr in (("ep_idx", ep_idx), ("step_idx", step_idx)):
            if name in f:
                del f[name]
            f.create_dataset(
                name, data=arr, chunks=(min(1000, n),), **blosc
            )
            print(f"wrote {name} ({arr.dtype}, shape={arr.shape})")

        print("final keys:", list(f.keys()))


if __name__ == "__main__":
    main()

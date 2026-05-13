"""Verify the converted rocket_expert.h5 and exercise HDF5Dataset."""

import sys
import h5py
import numpy as np

H5_PATH = "/oscar/data/jpober/cduong5/swm_cache/datasets/rocket_expert.h5"
CACHE_DIR = "/oscar/data/jpober/cduong5/swm_cache"


def main():
    print(f"Opening {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        keys = list(f.keys())
        print("keys:", keys)
        n = f["pixels"].shape[0]
        e = f["ep_len"].shape[0]
        assert f["ep_len"][:].sum() == n, "ep_len does not sum to N"
        assert f["pixels"].dtype == np.uint8
        assert f["pixels"].shape[1:] == (480, 480, 3)
        assert f["proprio"].shape == (n, 17)
        assert f["action"].shape == (n, 7)
        print(f"N={n} transitions, E={e} episodes")
        print(f"ep_len min/mean/max: {f['ep_len'][:].min()}/{f['ep_len'][:].mean():.1f}/{f['ep_len'][:].max()}")
        print(f"first offsets: {f['ep_offset'][:5]}")

    import stable_worldmodel as swm
    d = swm.data.HDF5Dataset(
        name="rocket_expert",
        num_steps=4,
        frameskip=1,
        cache_dir=CACHE_DIR,
        keys_to_load=["pixels", "action", "proprio"],
    )
    print(f"HDF5Dataset len={len(d)}")
    s = d[0]
    for k, v in s.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    # Expected: pixels (4,3,224,224) uint8; action (4,7) float64; proprio (4,17) float64
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())

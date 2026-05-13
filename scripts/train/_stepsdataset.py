"""Local shim for the historical `swm.data.StepsDataset` that was removed upstream.

Restored verbatim from git commit 35a29b81 (stable_worldmodel/data.py). The
upstream `stable_worldmodel` library deleted this class in favour of
`HDF5Dataset`, but the rocket_expert_all dataset is stored in HuggingFace
Arrow format and our training script was built against the `StepsDataset`
API. This local shim keeps the training script working without upstream
surgery.
"""
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch

import stable_worldmodel as swm


class StepsDataset(spt.data.HFDataset):
    """Dataset for loading multi-step trajectory sequences.

    Wraps an HF Arrow dataset that has 'episode_idx', 'step_idx', and
    'action' columns. Returns consecutive-step windows with frame skipping.

    Args:
        path (str): Name or path of the dataset within cache directory.
        num_steps (int): Number of consecutive steps per sample.
        frameskip (int): Number of steps between sampled frames.
        cache_dir (str | None): Base cache directory.
        transform (callable | None): Optional dict-to-dict transform.
        *args, **kwargs: Passed through to spt.data.HFDataset.
    """

    def __init__(
        self,
        path,
        *args,
        num_steps=2,
        frameskip=1,
        cache_dir=None,
        transform=None,
        **kwargs,
    ):
        data_dir = Path(cache_dir or swm.data.get_cache_dir(), path)
        super().__init__(str(data_dir), *args, **kwargs)

        self.data_dir = data_dir
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.transform = transform

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx'"
        assert "step_idx"    in self.dataset.column_names, "Dataset must have 'step_idx'"
        assert "action"      in self.dataset.column_names, "Dataset must have 'action'"

        self.dataset.set_format("torch")
        # Cache primitive (per-row) action dim so __getitem__'s frameskip fold
        # has a known target shape; the post-transform assertion below catches
        # silent shape regressions (e.g. transform re-permuting (T,A)->(A,T)).
        a0 = self.dataset[0]["action"]
        self._primitive_action_dim = int(np.asarray(a0).reshape(-1).shape[0])

        ep_indices = self.dataset["episode_idx"][:]
        self.episodes = np.unique(ep_indices)
        self.episode_slices = {
            int(e): self.get_episode_slice(int(e), ep_indices) for e in self.episodes
        }
        valid = [
            max(0, len(s) - self.num_steps * self.frameskip + 1)
            for s in self.episode_slices.values()
        ]
        self.cum_slices = np.cumsum([0] + valid)
        self.idx_to_ep = (
            np.searchsorted(self.cum_slices, torch.arange(len(self)), side="right") - 1
        )
        self.img_cols = self.infer_img_path_columns()

    def get_episode_slice(self, episode_idx, episode_indices):
        indices = np.flatnonzero(episode_indices == episode_idx)
        if len(indices) <= (self.num_steps * self.frameskip):
            raise ValueError(
                f"Episode {episode_idx} too short ({len(indices)}) for "
                f"num_steps={self.num_steps} frameskip={self.frameskip}"
            )
        return indices

    def __len__(self):
        return int(self.cum_slices[-1])

    def infer_img_path_columns(self):
        """Columns whose first entry looks like a filesystem path to an image."""
        img = set()
        for col in self.dataset.column_names:
            sample = self.dataset[col][0]
            if isinstance(sample, (list, tuple)):
                sample = sample[0] if sample else None
            if isinstance(sample, str) and sample.endswith((".png", ".jpg", ".jpeg")):
                img.add(col)
        return img

    def __getitem__(self, idx):
        ep = int(self.idx_to_ep[idx])
        within = idx - self.cum_slices[ep]
        ep_slice = self.episode_slices[int(self.episodes[ep])]
        start = within
        steps = {col: [] for col in self.dataset.column_names}
        for k in range(self.num_steps):
            t = start + k * self.frameskip
            row_idx = int(ep_slice[t])
            row = self.dataset[row_idx]
            for col in self.dataset.column_names:
                if col == "action" and self.frameskip > 1:
                    # The DINO-WM action encoder is built with
                    # in_chans=frameskip * action_dim. For a visual transition
                    # from frame t to t+frameskip, concatenate the primitive
                    # actions executed inside that interval.
                    actions = []
                    for j in range(self.frameskip):
                        step_row = self.dataset[int(ep_slice[t + j])]
                        action = step_row[col]
                        if not torch.is_tensor(action):
                            action = torch.as_tensor(action)
                        actions.append(action.reshape(-1))
                    v = torch.cat(actions, dim=0)
                else:
                    v = row[col]
                if col in self.img_cols and isinstance(v, str):
                    import PIL.Image as Image
                    v = np.array(Image.open(v).convert("RGB"))
                steps[col].append(v)

        # Stack across the num_steps dimension. Image windows intentionally
        # stay as mutable per-frame lists until after the transform pipeline:
        # stable_pretraining writes resized crops back into "pixels.0",
        # "pixels.1", ... and those frames may change spatial size.
        out = {}
        for col, lst in steps.items():
            if col in self.img_cols:
                out[col] = lst
            elif isinstance(lst[0], torch.Tensor):
                arr = torch.stack(lst)
                out[col] = arr
            elif isinstance(lst[0], np.ndarray):
                arr = torch.from_numpy(np.stack(lst))
                out[col] = arr
            else:
                out[col] = lst

        # Per-col flat window access: "pixels.0", "pixels.1", ...
        flat = {}
        for col, arr in out.items():
            if isinstance(arr, list):
                for i, frame in enumerate(arr):
                    flat[f"{col}.{i}"] = frame
            elif isinstance(arr, torch.Tensor) and arr.dim() >= 1:
                for i in range(arr.shape[0]):
                    flat[f"{col}.{i}"] = arr[i]
            flat[col] = arr

        flat = self.transform(flat) if self.transform else flat

        # After per-frame image transforms have produced consistently sized
        # tensors, restore the model-facing shape: (T, C, H, W). DataLoader
        # then collates this as (B, T, C, H, W).
        for col in self.img_cols:
            value = flat.get(col)
            if isinstance(value, list) and value and all(torch.is_tensor(v) for v in value):
                flat[col] = torch.stack(value, dim=0)

        # Guard the action fold: dinowm's Embedder is built with
        # in_chans=frameskip*action_dim and expects (B, T, A_packed) with
        # A_packed = frameskip * primitive_action_dim. A regression here
        # surfaces in conv1d as a cryptic channel-count error.
        if "action" in flat and torch.is_tensor(flat["action"]):
            expected = (self.num_steps, self.frameskip * self._primitive_action_dim)
            got = tuple(flat["action"].shape)
            assert got == expected, (
                f"StepsDataset action fold broken: got {got}, "
                f"expected {expected} (num_steps={self.num_steps}, "
                f"frameskip={self.frameskip}, primitive_action_dim="
                f"{self._primitive_action_dim})"
            )

        return flat

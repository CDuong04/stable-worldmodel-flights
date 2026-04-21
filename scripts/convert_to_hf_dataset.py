"""Convert collected expert data (images + pickle) to HuggingFace dataset format."""

import os
import pickle
import numpy as np
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence, Image as HFImage


def main():
    data_dir = Path("data/expert_trajectories")
    meta_path = data_dir / "expert_with_images_meta.pkl"
    state_path = data_dir / "expert_default_gnc.pkl"

    print(f"Loading metadata from {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print(f"Loading state data from {state_path}")
    with open(state_path, "rb") as f:
        state_data = pickle.load(f)

    trajectories = meta["trajectories"]
    state_trajs = state_data["trajectories"]
    print(f"{len(trajectories)} image trajectories, {len(state_trajs)} state trajectories")

    records = []
    for ep_idx, (img_traj, st_traj) in enumerate(zip(trajectories, state_trajs)):
        img_dir = img_traj["img_dir"]
        obs = st_traj["obs"]
        actions = st_traj["actions"]
        length = min(img_traj["length"], st_traj["length"])

        for t in range(length):
            img_path = os.path.join(img_dir, f"{t:04d}.png")
            if not os.path.exists(img_path):
                continue
            records.append({
                "pixels": img_path,
                "proprio": obs[t].tolist(),
                "action": actions[t].tolist(),
                "episode": ep_idx,
                "step": t,
            })

    print(f"Total records: {len(records)}")

    ds = Dataset.from_list(records)
    ds = ds.cast_column("pixels", HFImage())

    save_path = str(data_dir / "rocket_expert_default_hf")
    ds.save_to_disk(save_path)
    print(f"Saved HF dataset to {save_path}")
    print(f"  {len(ds)} rows, columns: {ds.column_names}")


if __name__ == "__main__":
    main()

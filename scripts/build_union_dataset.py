"""Build union HuggingFace dataset = nominal + in-flight disturb (PDG).

Source datasets:
  data/expert_trajectories/             — 500 nominal PDG trajectories
  data/expert_trajectories_disturb_pdg/ — 500 in-flight disturb PDG trajectories

Output:
  data/expert_trajectories_union/rocket_expert_union/  — concat HF dataset
  Sym-linked into STABLEWM_HOME so spt.data.StepsDataset can find it.

Each record carries 'pixels' (path), 'proprio' (17-D state), 'action' (7-D),
'episode_idx', 'step_idx'. The disturb trajectories don't have a paired
state pickle (state was stored inline in the meta), so we read state from
the meta itself.
"""
import os
import pickle
from pathlib import Path

from datasets import Dataset, concatenate_datasets


def trajs_from_meta(meta_path: Path, ep_offset: int = 0):
    """Yield records from a meta.pkl. Each traj has 'obs', 'actions', 'img_dir'."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    trajs = meta["trajectories"]
    records = []
    for local_idx, t in enumerate(trajs):
        ep_idx = ep_offset + local_idx
        img_dir = t["img_dir"]
        obs = t["obs"]
        actions = t["actions"]
        length = min(t["length"], len(obs), len(actions))
        for step in range(length):
            img_path = os.path.join(img_dir, f"{step:04d}.png")
            if not os.path.exists(img_path):
                continue
            records.append({
                "pixels": os.path.abspath(img_path),
                "proprio": obs[step].tolist() if hasattr(obs[step], 'tolist') else list(obs[step]),
                "action": actions[step].tolist() if hasattr(actions[step], 'tolist') else list(actions[step]),
                "episode_idx": int(ep_idx),
                "step_idx": int(step),
            })
    return records, len(trajs)


def main():
    nominal_dir = Path("data/expert_trajectories")
    disturb_dir = Path("data/expert_trajectories_disturb_pdg")

    print("Reading nominal meta...")
    nom_records, nom_n_trajs = trajs_from_meta(nominal_dir / "expert_with_images_meta.pkl", ep_offset=0)
    print(f"  {nom_n_trajs} nominal trajectories, {len(nom_records)} transitions")

    print("Reading disturb meta...")
    dist_records, dist_n_trajs = trajs_from_meta(
        disturb_dir / "expert_with_images_meta.pkl", ep_offset=nom_n_trajs
    )
    print(f"  {dist_n_trajs} disturb trajectories, {len(dist_records)} transitions")

    union_records = nom_records + dist_records
    print(f"Total: {len(union_records)} transitions across {nom_n_trajs + dist_n_trajs} trajectories")

    ds = Dataset.from_list(union_records)
    union_dir = Path("data/expert_trajectories_union")
    union_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(union_dir / "rocket_expert_union")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")

    # Sym-link into STABLEWM_HOME so spt.data.StepsDataset(name='rocket_expert_union') finds it
    cache_dir = Path(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    link_path = cache_dir / "rocket_expert_union"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(os.path.abspath(save_path), link_path)
    print(f"Symlinked: {link_path} -> {save_path}")


if __name__ == "__main__":
    main()

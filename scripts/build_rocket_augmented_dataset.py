"""Concatenate expert union data with on-policy RocketJEPA rollouts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main():
    p = argparse.ArgumentParser(description="Build rocket expert + on-policy HF dataset")
    p.add_argument("--base", default="data/expert_trajectories_union/rocket_expert_union")
    p.add_argument("--extra", nargs="+", default=["data/expert_trajectories_onpolicy/rocket_onpolicy_cem_clf/dataset"])
    p.add_argument("--out", default="data/expert_trajectories_union/rocket_expert_union_onpolicy")
    p.add_argument("--link-name", default="rocket_expert_union_onpolicy")
    args = p.parse_args()

    datasets = [load_from_disk(args.base)]
    ep_offset = int(max(datasets[0]["episode_idx"])) + 1
    for path in args.extra:
        ds = load_from_disk(path)
        ep_ids = [int(x) + ep_offset for x in ds["episode_idx"]]
        ds = ds.remove_columns(["episode_idx"]).add_column("episode_idx", ep_ids)
        datasets.append(ds)
        ep_offset = max(ep_ids) + 1

    out = concatenate_datasets(datasets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_path))
    print(f"Saved {len(out)} rows to {out_path}")

    cache_dir = Path(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    link_path = cache_dir / args.link_name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(os.path.abspath(out_path), link_path)
    print(f"Symlinked: {link_path} -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

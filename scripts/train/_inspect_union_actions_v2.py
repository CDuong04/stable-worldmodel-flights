"""Stage 2 diagnostic: invoke the EXACT get_data() pipeline used by
lejepa_wm.py and inspect a batch out of train and val DataLoaders.

The first diagnostic (v1) showed the local shim folds actions correctly
to (3, 14) when used with transform=None. But the actual training run
applies a transform pipeline and the model still sees 7-dim per
macro-step. We need to inspect the exact code path.
"""
from __future__ import annotations
import sys
from pathlib import Path

import hydra
import torch

SCRIPTS_TRAIN = Path(__file__).resolve().parent
if str(SCRIPTS_TRAIN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TRAIN))


@hydra.main(config_path="../train", config_name="rocket_lejepa_config", version_base="1.3")
def main(cfg):
    print("=" * 70)
    print("[A] Build the EXACT same DataModule as lejepa_wm.py")
    print("=" * 70)
    # Import the get_data symbol from the training entry script.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lejepa_wm_module", str(SCRIPTS_TRAIN / "lejepa_wm.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # Need the module to be importable to evaluate the @hydra.main-decorated
    # functions; defer execution failures.
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

    dm = mod.get_data(cfg)
    print(f"  type(dm)                 = {type(dm).__name__}")

    # Pull a train batch
    print("\n" + "=" * 70)
    print("[B] First train batch — what does the model actually see?")
    print("=" * 70)
    train_loader = dm._train if hasattr(dm, "_train") else getattr(dm, "train", None)
    if train_loader is None:
        # spt.data.DataModule may store under different attr; introspect
        for attr in ("train", "_train", "train_dataloader"):
            obj = getattr(dm, attr, None)
            if obj is not None:
                print(f"  using attr {attr!r}")
                if callable(obj):
                    train_loader = obj()
                else:
                    train_loader = obj
                break
    print(f"  train_loader type        = {type(train_loader).__name__}")
    batch = next(iter(train_loader))
    print(f"  batch.keys()             = {sorted(batch.keys())}")
    if "action" in batch:
        a = batch["action"]
        print(f"  batch['action'].shape    = {tuple(a.shape)}")
        print(f"  batch['action'].dtype    = {a.dtype}")

    # Pull a val batch
    print("\n" + "=" * 70)
    print("[C] First val batch — same?")
    print("=" * 70)
    val_loader = None
    for attr in ("val", "_val", "val_dataloader"):
        obj = getattr(dm, attr, None)
        if obj is not None:
            print(f"  using attr {attr!r}")
            if callable(obj):
                val_loader = obj()
            else:
                val_loader = obj
            break
    print(f"  val_loader type          = {type(val_loader).__name__}")
    batch_v = next(iter(val_loader))
    if "action" in batch_v:
        av = batch_v["action"]
        print(f"  val batch['action'].shape= {tuple(av.shape)}")
        print(f"  val batch['action'].dtype= {av.dtype}")

    # Inspect the underlying dataset object
    print("\n" + "=" * 70)
    print("[D] What dataset class is behind the DataLoader?")
    print("=" * 70)
    inner = train_loader.dataset
    print(f"  train_loader.dataset     = {type(inner).__name__}")
    if hasattr(inner, "dataset"):
        print(f"    .dataset             = {type(inner.dataset).__name__}")
        if hasattr(inner.dataset, "transform"):
            print(f"    .dataset.transform   = {inner.dataset.transform!r}")

    # Step-by-step on the underlying dataset
    print("\n" + "=" * 70)
    print("[E] Underlying dataset[0]['action'].shape (no DataLoader collate)")
    print("=" * 70)
    base = inner
    while hasattr(base, "dataset"):
        base = base.dataset
    print(f"  base type                = {type(base).__name__}")
    item = base[0] if hasattr(base, "__getitem__") else None
    if item is not None and "action" in item:
        ai = item["action"]
        if hasattr(ai, "shape"):
            print(f"  base[0]['action'].shape  = {tuple(ai.shape)}")
        else:
            print(f"  base[0]['action'] type   = {type(ai).__name__}, repr={ai!r}")

    print("\nDone v2.")


if __name__ == "__main__":
    main()

"""Evaluate RocketJEPA observation-probe state estimation.

This script measures the experiment described around ``tab:decoder`` and P3 in
``rocket_jepa.tex``:

* direct probe error on held-out expert frames, d_phi(encode(I_t)) vs x_t;
* multi-step decoded rollout error, d_phi(z_hat_{t+h}) vs x_{t+h};
* physics re-simulation gap from decoded x_hat_t under the logged actions.

Outputs are written as JSON, a TeX table fragment, and plot files.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_from_disk

from stable_worldmodel.clf_cost import make_rocket_V
from stable_worldmodel.policy import AutoCostModel
from stable_worldmodel.solver.rocket_physics import RocketPhysicsModel


STATE_GROUPS = {
    "position": [0, 1, 2],
    "velocity": [3, 4, 5],
    "orientation": [6, 7, 8, 9],
    "angular_velocity": [10, 11, 12],
    "fuel": [13],
    "target_rel": [14, 15, 16],
}

PAPER_ROWS = [
    ("Position (m)", "position"),
    ("Velocity (m/s)", "velocity"),
    ("Orientation (quat)", "orientation"),
    ("Angular velocity (rad/s)", "angular_velocity"),
    ("Fuel fraction", "fuel"),
]


def _inject_obs_probe_symbol() -> None:
    """Make old pickled checkpoints that reference __main__.ObsProbe load."""
    train_dir = Path(__file__).resolve().parent / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    mod = importlib.import_module("lejepa_wm")
    for name in ("ObsProbe", "ScalarProbe", "DeltaProbe"):
        if hasattr(mod, name):
            setattr(sys.modules["__main__"], name, getattr(mod, name))


def load_model(model_name: str, device: torch.device):
    _inject_obs_probe_symbol()
    model = AutoCostModel(model_name)
    model.to(device)
    model.eval()
    model.device = str(device)
    probe = getattr(model, "obs_probe", None)
    if probe is None:
        raise RuntimeError(
            f"World model {model_name!r} has no baked-in obs_probe. "
            "Use a RocketJEPA checkpoint trained with obs_probe.enabled=true."
        )
    probe.to(device).eval()
    return model, probe


def image_to_tensor(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    return (ten - 0.5) / 0.5


def compute_norm_stats(ds, frameskip: int):
    proprio = np.asarray(ds["proprio"], dtype=np.float32)
    action = np.asarray(ds["action"], dtype=np.float32)
    p_mean = proprio.mean(axis=0)
    p_std = np.maximum(proprio.std(axis=0), 1e-6)
    a_mean = np.tile(action.mean(axis=0), frameskip)
    a_std = np.tile(np.maximum(action.std(axis=0), 1e-6), frameskip)
    return p_mean, p_std, a_mean, a_std


def build_episode_index(ds):
    ep_ids = np.asarray(ds["episode_idx"], dtype=np.int64)
    step_ids = np.asarray(ds["step_idx"], dtype=np.int64)
    out: dict[int, list[int]] = defaultdict(list)
    for row, ep in enumerate(ep_ids):
        out[int(ep)].append(row)
    for ep, rows in out.items():
        rows.sort(key=lambda i: int(step_ids[i]))
    return dict(out)


def sample_windows(ds, episode_rows, args):
    rng = random.Random(args.seed)
    episodes = sorted(episode_rows)
    rng.shuffle(episodes)
    n_val = max(1, int(math.ceil(len(episodes) * args.val_episode_fraction)))
    eval_eps = sorted(episodes[:n_val])

    needed = (args.history - 1 + max(args.horizons)) * args.frameskip
    candidates = []
    for ep in eval_eps:
        rows = episode_rows[ep]
        last_start = len(rows) - 1 - needed
        if last_start >= 0:
            for start_pos in range(last_start + 1):
                candidates.append((ep, start_pos))
    if not candidates:
        raise RuntimeError("No valid held-out windows for the requested horizons.")
    rng.shuffle(candidates)
    return candidates[: args.num_windows]


def packed_action(ds, rows, pos: int, frameskip: int) -> np.ndarray:
    pieces = []
    for j in range(frameskip):
        pieces.append(np.asarray(ds[int(rows[pos + j])]["action"], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces, axis=0)


def make_batch(ds, episode_rows, windows, args, p_mean, p_std, a_mean, a_std):
    H = max(args.horizons)
    hset = list(args.horizons)
    hist_imgs, hist_props, action_seq = [], [], []
    future_imgs, future_states = [], []
    x0_states, primitive_actions = [], []

    for ep, start in windows:
        rows = episode_rows[ep]
        visual_positions = [start + j * args.frameskip for j in range(args.history + H)]

        hist_imgs.append(torch.stack([
            image_to_tensor(ds[int(rows[pos])]["pixels"])
            for pos in visual_positions[: args.history]
        ], dim=0))
        hp = np.stack([
            np.asarray(ds[int(rows[pos])]["proprio"], dtype=np.float32)
            for pos in visual_positions[: args.history]
        ], axis=0)
        hist_props.append((hp - p_mean) / p_std)

        seq = np.stack([
            (packed_action(ds, rows, start + j * args.frameskip, args.frameskip) - a_mean) / a_std
            for j in range(args.history + H - 1)
        ], axis=0)
        action_seq.append(seq)

        future_imgs.append(torch.stack([
            image_to_tensor(ds[int(rows[start + (args.history - 1 + h) * args.frameskip])]["pixels"])
            for h in hset
        ], dim=0))
        future_states.append(np.stack([
            np.asarray(
                ds[int(rows[start + (args.history - 1 + h) * args.frameskip])]["proprio"],
                dtype=np.float32,
            )
            for h in hset
        ], axis=0))

        x0_pos = start + (args.history - 1) * args.frameskip
        x0_states.append(np.asarray(ds[int(rows[x0_pos])]["proprio"], dtype=np.float32))
        primitive_actions.append(np.stack([
            np.asarray(ds[int(rows[x0_pos + j])]["action"], dtype=np.float32)
            for j in range(H * args.frameskip)
        ], axis=0))

    return {
        "hist_pixels": torch.stack(hist_imgs, dim=0),
        "hist_proprio": torch.from_numpy(np.stack(hist_props, axis=0)).float(),
        "action_seq": torch.from_numpy(np.stack(action_seq, axis=0)).float(),
        "future_pixels": torch.stack(future_imgs, dim=0),
        "future_states": torch.from_numpy(np.stack(future_states, axis=0)).float(),
        "x0_states": torch.from_numpy(np.stack(x0_states, axis=0)).float(),
        "primitive_actions": torch.from_numpy(np.stack(primitive_actions, axis=0)).float(),
    }


class Accumulator:
    def __init__(self, horizons):
        self.horizons = [int(h) for h in horizons]
        self.data = {
            mode: {
                int(h): {
                    "groups": {g: {"sse": 0.0, "n": 0} for g in STATE_GROUPS},
                    "state": {"sse": 0.0, "n": 0},
                    "V": {"sse": 0.0, "n": 0},
                    "l2": {"sum": 0.0, "n": 0},
                }
                for h in self.horizons
            }
            for mode in ("direct", "rollout", "physics_resim")
        }

    def add(self, mode: str, horizon: int, pred: torch.Tensor, true: torch.Tensor, V_fn):
        err = (pred - true).detach().float().cpu()
        for group, idx in STATE_GROUPS.items():
            sub = err[:, idx]
            self.data[mode][horizon]["groups"][group]["sse"] += float((sub ** 2).sum())
            self.data[mode][horizon]["groups"][group]["n"] += int(sub.numel())
        self.data[mode][horizon]["state"]["sse"] += float((err ** 2).sum())
        self.data[mode][horizon]["state"]["n"] += int(err.numel())
        self.data[mode][horizon]["l2"]["sum"] += float(torch.linalg.norm(err, dim=1).sum())
        self.data[mode][horizon]["l2"]["n"] += int(err.shape[0])
        with torch.no_grad():
            v_err = (V_fn(pred) - V_fn(true)).detach().float().cpu()
        self.data[mode][horizon]["V"]["sse"] += float((v_err ** 2).sum())
        self.data[mode][horizon]["V"]["n"] += int(v_err.numel())

    def summary(self):
        out = {}
        for mode, by_h in self.data.items():
            out[mode] = {}
            for h, rec in by_h.items():
                groups = {
                    group: {
                        "mse": vals["sse"] / max(vals["n"], 1),
                        "rmse": math.sqrt(vals["sse"] / max(vals["n"], 1)),
                    }
                    for group, vals in rec["groups"].items()
                }
                state_mse = rec["state"]["sse"] / max(rec["state"]["n"], 1)
                V_mse = rec["V"]["sse"] / max(rec["V"]["n"], 1)
                out[mode][str(h)] = {
                    "groups": groups,
                    "state_mse": state_mse,
                    "state_rmse": math.sqrt(state_mse),
                    "V_mse": V_mse,
                    "V_rmse": math.sqrt(V_mse),
                    "mean_l2": rec["l2"]["sum"] / max(rec["l2"]["n"], 1),
                    "n": rec["l2"]["n"],
                }
        return out


def evaluate(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = load_from_disk(args.dataset)
    p_mean, p_std, a_mean, a_std = compute_norm_stats(ds, args.frameskip)
    episode_rows = build_episode_index(ds)
    windows = sample_windows(ds, episode_rows, args)
    model, probe = load_model(args.model, device)
    physics = RocketPhysicsModel(dt=args.physics_dt).to(device).eval()
    V_fn = make_rocket_V(alpha=args.alpha, beta=args.beta)
    acc = Accumulator(args.horizons)
    p_mean_t = torch.as_tensor(p_mean, dtype=torch.float32, device=device)
    p_std_t = torch.as_tensor(p_std, dtype=torch.float32, device=device)
    probe_space = args.probe_output_space

    max_h = max(args.horizons)
    for offset in range(0, len(windows), args.batch_size):
        chunk = windows[offset: offset + args.batch_size]
        batch = make_batch(ds, episode_rows, chunk, args, p_mean, p_std, a_mean, a_std)
        hist_pixels = batch["hist_pixels"].to(device)
        hist_proprio = batch["hist_proprio"].to(device)
        action_seq = batch["action_seq"].to(device)
        future_pixels = batch["future_pixels"].to(device)
        future_states = batch["future_states"].to(device)
        primitive_actions = batch["primitive_actions"].to(device)

        with torch.no_grad():
            rollout_info = model.rollout(
                {"pixels": hist_pixels, "proprio": hist_proprio},
                action_seq,
            )
            pred_pixels = rollout_info["predicted_pixels_embed"]

            direct_info = model.encode(
                {"pixels": future_pixels},
                pixels_key="pixels",
                target="direct_embed",
            )
            direct_z = direct_info["pixels_direct_embed"].mean(dim=2)
            B, T, D = direct_z.shape
            direct_x_probe = probe(direct_z.reshape(B * T, D)).reshape(B, T, -1)

            if probe_space == "auto":
                raw_mse = torch.mean((direct_x_probe[:, 0] - future_states[:, 0]) ** 2).item()
                unnorm = direct_x_probe[:, 0] * p_std_t + p_mean_t
                unnorm_mse = torch.mean((unnorm - future_states[:, 0]) ** 2).item()
                probe_space = "normalized" if unnorm_mse < raw_mse else "raw"
                print(
                    "[state-est] probe output space: "
                    f"{probe_space} (raw_mse={raw_mse:.4g}, unnorm_mse={unnorm_mse:.4g})",
                    flush=True,
                )

            def to_raw_state(x):
                if probe_space == "normalized":
                    return x * p_std_t + p_mean_t
                return x

            direct_x = to_raw_state(direct_x_probe)

            x0_info = model.encode(
                {"pixels": hist_pixels[:, -1:]},
                pixels_key="pixels",
                target="x0_embed",
            )
            x0_z = x0_info["pixels_x0_embed"].mean(dim=2).squeeze(1)
            x0_hat = to_raw_state(probe(x0_z))
            phys_traj = physics.rollout(x0_hat, primitive_actions)

            for hi, h in enumerate(args.horizons):
                true = future_states[:, hi]
                z_idx = args.history + int(h) - 1
                roll_z = pred_pixels[:, z_idx].mean(dim=1)
                roll_x = to_raw_state(probe(roll_z))
                phys_x = phys_traj[:, int(h) * args.frameskip]

                acc.add("direct", int(h), direct_x[:, hi], true, V_fn)
                acc.add("rollout", int(h), roll_x, true, V_fn)
                acc.add("physics_resim", int(h), phys_x, true, V_fn)

        done = min(offset + args.batch_size, len(windows))
        print(f"[state-est] processed {done}/{len(windows)} windows (max_h={max_h})", flush=True)

    return {
        "model": args.model,
        "dataset": args.dataset,
        "num_windows": len(windows),
        "horizons": [int(h) for h in args.horizons],
        "history": args.history,
        "frameskip": args.frameskip,
        "alpha": args.alpha,
        "beta": args.beta,
        "probe_output_space": probe_space,
        "probe_params": int(sum(p.numel() for p in probe.parameters())),
        "metrics": acc.summary(),
    }


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def write_tex_table(results: dict, path: Path) -> None:
    horizons = results["horizons"]
    metrics = results["metrics"]["rollout"]
    lines = [
        "% Auto-generated by scripts/evaluate_state_estimation.py",
        "\\begin{tabular}{l" + "c" * len(horizons) + "}",
        "\\toprule",
        "\\textbf{State dimension} & " + " & ".join([f"\\textbf{{{h}-step}}" for h in horizons]) + " \\\\",
        "\\midrule",
    ]
    for label, key in PAPER_ROWS:
        vals = [fmt(metrics[str(h)]["groups"][key]["mse"]) for h in horizons]
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def write_plots(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = np.asarray(results["horizons"], dtype=int)
    metrics = results["metrics"]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, key in PAPER_ROWS:
        y = [metrics["rollout"][str(h)]["groups"][key]["mse"] for h in horizons]
        ax.plot(horizons, y, marker="o", linewidth=1.8, label=label.replace(" (m/s)", "").replace(" (m)", ""))
    ax.set_yscale("log")
    ax.set_xlabel("Latent rollout horizon")
    ax.set_ylabel("Decoded state MSE")
    ax.set_title("RocketJEPA probe trust horizon")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "state_estimation_rollout_mse.png", dpi=220)
    fig.savefig(out_dir / "state_estimation_rollout_mse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mode, label in [
        ("direct", "Observed embedding"),
        ("rollout", "Predicted latent rollout"),
        ("physics_resim", "Physics re-sim from decoded state"),
    ]:
        y = [metrics[mode][str(h)]["state_mse"] for h in horizons]
        ax.plot(horizons, y, marker="o", linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("17D state MSE")
    ax.set_title("State-estimation error decomposition")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "state_estimation_modes.png", dpi=220)
    fig.savefig(out_dir / "state_estimation_modes.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="RocketJEPA state-estimation experiment")
    p.add_argument("--model", default="/oscar/scratch/aiyer40/rocket_lejepa_wm_union")
    p.add_argument("--dataset", default="data/expert_trajectories_union/rocket_expert_union")
    p.add_argument("--out-dir", default="outputs/rocket_jepa/state_estimation")
    p.add_argument("--num-windows", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=2)
    p.add_argument("--val-episode-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    p.add_argument("--physics-dt", type=float, default=0.025)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--probe-output-space", default="auto",
                   choices=["auto", "raw", "normalized"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(args)

    json_path = out_dir / "state_estimation_metrics.json"
    tex_path = out_dir / "state_estimation_table.tex"
    json_path.write_text(json.dumps(results, indent=2))
    write_tex_table(results, tex_path)
    write_plots(results, out_dir)

    print(f"[state-est] wrote {json_path}")
    print(f"[state-est] wrote {tex_path}")
    print(f"[state-est] wrote plots under {out_dir}")


if __name__ == "__main__":
    main()

"""Aggregate stable-MPC sweep results into a comparison table.

Reads every JSON file in results/sweep/, parses the per-episode summary, and
prints a paired-seed comparison table grouped by (ablation, level).

Usage:
  srun --pty --time=00:05:00 python scripts/aggregate_stable_mpc_sweep.py \
      --sweep-dir results/sweep \
      --output results/sweep_summary.csv

Columns:
  ablation         which constraint set was active (c1c2c3 / c1c2 / c3)
  level            disturbance level (easy / extreme)
  seed             paired seed
  episode_length   simulator steps before terminal
  episode_return   total reward
  mean_descent     mean realized V-descent rate (positive = converging)
  violation_rate   fraction of steps with V increase
  final_V          terminal Lyapunov value
  fallback_count   how many times runtime safety belt fired during episode
  wall_time_s      wall-clock seconds for the episode
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_filename(stem: str) -> tuple[str, str, int] | None:
    """results/sweep/{ablation}_{level}_seed{N}.json → (ablation, level, seed)."""
    # Expect tags like c1c2c3_extreme_seed42.
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None
    try:
        seed = int(parts[1])
    except ValueError:
        return None
    head = parts[0]
    # The remaining part: {ablation}_{level} where level has no underscore.
    chunks = head.rsplit("_", 1)
    if len(chunks) != 2:
        return None
    ablation, level = chunks
    return ablation, level, seed


def collect_rows(sweep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(sweep_dir.glob("*.json")):
        parsed = _parse_filename(path.stem)
        if parsed is None:
            print(f"[warn] skipping unrecognized filename: {path.name}")
            continue
        ablation, level, seed = parsed
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"[warn] {path.name}: malformed JSON ({e})")
            continue
        for ep in payload.get("episodes", []):
            safety = ep.get("safety", {})
            rows.append({
                "file": path.name,
                "ablation": ablation,
                "level": level,
                "seed": seed,
                "episode_length": ep.get("episode_length"),
                "episode_return": ep.get("episode_return"),
                "mean_descent": safety.get("mean_descent"),
                "violation_rate": safety.get("violation_rate"),
                "final_V": safety.get("final_V"),
                "max_V": safety.get("max_V"),
                "fallback_count": ep.get("fallback_count"),
                "fallback_triggers": safety.get("fallback_triggers"),
                "wall_time_s": ep.get("wall_time_s"),
            })
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["ablation"], r["level"])].append(r)

    header = (
        f"{'ablation':<10} {'level':<8} {'#seeds':>6}  "
        f"{'len':>6} {'return':>9}  "
        f"{'mean_δ':>9} {'viol_rate':>10}  "
        f"{'final_V':>9} {'fallback':>9} {'wall_s':>7}"
    )
    print("\n" + header)
    print("-" * len(header))

    for (ab, lvl), group in sorted(grouped.items()):
        n = len(group)
        def col(key: str, fmt: str = "{:.4f}") -> str:
            vals = [r[key] for r in group if r.get(key) is not None]
            if not vals:
                return "—"
            m = mean(vals)
            return fmt.format(m)

        print(
            f"{ab:<10} {lvl:<8} {n:>6}  "
            f"{col('episode_length', '{:.0f}'):>6} "
            f"{col('episode_return', '{:.2f}'):>9}  "
            f"{col('mean_descent'):>9} {col('violation_rate'):>10}  "
            f"{col('final_V', '{:.2f}'):>9} "
            f"{col('fallback_count', '{:.1f}'):>9} "
            f"{col('wall_time_s', '{:.1f}'):>7}"
        )


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV: {path}")


def save_summary_json(rows: list[dict], path: Path) -> None:
    grouped: dict[str, dict] = {}
    for r in rows:
        key = f"{r['ablation']}_{r['level']}"
        grouped.setdefault(key, {
            "ablation": r["ablation"],
            "level": r["level"],
            "episode_length": [],
            "episode_return": [],
            "mean_descent": [],
            "violation_rate": [],
            "final_V": [],
            "fallback_count": [],
        })
        for k in ("episode_length", "episode_return", "mean_descent",
                  "violation_rate", "final_V", "fallback_count"):
            if r.get(k) is not None:
                grouped[key][k].append(r[k])
    out = {}
    for key, vals in grouped.items():
        out[key] = {
            "ablation": vals["ablation"],
            "level": vals["level"],
            "n_seeds": len(vals["episode_length"]),
        }
        for k in ("episode_length", "episode_return", "mean_descent",
                  "violation_rate", "final_V", "fallback_count"):
            if vals[k]:
                out[key][f"{k}_mean"] = float(mean(vals[k]))
                if len(vals[k]) > 1:
                    out[key][f"{k}_std"] = float(stdev(vals[k]))
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote summary JSON: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default=str(REPO_ROOT / "results/sweep"))
    p.add_argument("--output-csv", default=str(REPO_ROOT / "results/sweep_summary.csv"))
    p.add_argument("--output-json", default=str(REPO_ROOT / "results/sweep_summary.json"))
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep dir does not exist: {sweep_dir}", file=sys.stderr)
        return 1

    rows = collect_rows(sweep_dir)
    print(f"Loaded {len(rows)} episode records from {sweep_dir}")
    print_table(rows)
    save_csv(rows, Path(args.output_csv))
    save_summary_json(rows, Path(args.output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

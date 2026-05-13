"""Analyze the paper-grade RocketJEPA CLF-MPC paired experiment.

Consumes one-episode JSON files emitted by ``scripts/evaluate_rocket_clf.py``
and writes:

  * paired CLF-vs-MPC statistics with bootstrap confidence intervals;
  * an alarm-calibration table from per-step decoded Lyapunov descent traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def parse_float_tag(text: str) -> float:
    return float(text.replace("p", ".").replace("m", "-"))


def load_episode_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    pat = re.compile(r"__lambda([^_]+)_eta([^_]+)__seed(\d+)\.json$")
    for path in sorted(results_dir.glob("*.json")):
        match = pat.search(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        lam = parse_float_tag(match.group(1))
        eta = parse_float_tag(match.group(2))
        seed = int(match.group(3))
        for level, metrics in data.get("by_level", {}).items():
            for ep in metrics.get("episodes", []):
                row = {
                    "path": str(path),
                    "seed": int(ep.get("seed", seed)),
                    "level": level,
                    "lambda_V": lam,
                    "eta": eta,
                    "success": float(bool(ep.get("success", False))),
                    "violation_rate": float(ep["violation_rate"]),
                    "mean_descent": float(ep["mean_descent"]),
                    "min_descent": float(ep.get("min_descent", np.nan)),
                    "return": float(ep["return"]),
                    "steps": float(ep["T"]),
                    "max_V": float(ep.get("max_V", np.nan)),
                    "final_V": float(ep.get("final_V", np.nan)),
                    "plan_pred_violation_rate": float(ep.get("plan_pred_violation_rate", np.nan)),
                    "plan_pred_mean_descent": float(ep.get("plan_pred_mean_descent", np.nan)),
                    "plan_clf_penalty": float(ep.get("plan_clf_penalty", np.nan)),
                    "n_replans": float(ep.get("n_replans", np.nan)),
                    "delta_trace": [float(x) for x in ep.get("delta_trace", [])],
                }
                rows.append(row)
    return rows


def paired(rows: list[dict], level: str, lam_a: float, lam_b: float) -> list[tuple[dict, dict]]:
    a = {r["seed"]: r for r in rows if r["level"] == level and r["lambda_V"] == lam_a}
    b = {r["seed"]: r for r in rows if r["level"] == level and r["lambda_V"] == lam_b}
    seeds = sorted(set(a) & set(b))
    return [(a[s], b[s]) for s in seeds]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def signflip_p_less(diffs: np.ndarray, max_exact: int = 20, rng_seed: int = 1234) -> float:
    """One-sided paired randomization p-value for mean(diff) < 0."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    obs = float(diffs.mean())
    mags = np.abs(diffs)
    n = mags.size
    if n <= max_exact:
        count = 0
        total = 1 << n
        for mask in range(total):
            s = 0.0
            for i, mag in enumerate(mags):
                s += mag if (mask >> i) & 1 else -mag
            if (s / n) <= obs + 1e-12:
                count += 1
        return float(count / total)

    rng = np.random.default_rng(rng_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(200_000, n))
    means = (signs * mags).mean(axis=1)
    return float(np.mean(means <= obs + 1e-12))


def metric_summary(pairs: list[tuple[dict, dict]], metric: str, rng: np.random.Generator,
                   n_boot: int, lower_is_better: bool = True) -> dict:
    base = np.array([p[0][metric] for p in pairs], dtype=float)
    clf = np.array([p[1][metric] for p in pairs], dtype=float)
    diff = clf - base
    lo, hi = bootstrap_ci(diff, rng, n_boot)
    return {
        f"{metric}_mpc_mean": float(np.nanmean(base)),
        f"{metric}_clf_mean": float(np.nanmean(clf)),
        f"{metric}_delta_mean": float(np.nanmean(diff)),
        f"{metric}_delta_ci95_low": lo,
        f"{metric}_delta_ci95_high": hi,
        f"{metric}_p_better": signflip_p_less(diff if lower_is_better else -diff),
        f"{metric}_wins_better": int(np.sum(diff < 0 if lower_is_better else diff > 0)),
    }


def write_dict_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def conformal_quantile(scores: np.ndarray, q: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    if scores.size == 0:
        return float("nan")
    # Split-conformal finite-sample quantile index, clipped at the largest score.
    idx = int(math.ceil((scores.size + 1) * q)) - 1
    idx = min(max(idx, 0), scores.size - 1)
    return float(scores[idx])


def alarm_fire(delta: np.ndarray, threshold: float, k: int) -> tuple[bool, int | None]:
    below = delta < -threshold
    streak = 0
    for i, hit in enumerate(below):
        streak = streak + 1 if bool(hit) else 0
        if streak >= k:
            return True, i
    return False, None


def alarm_calibration(rows: list[dict], level: str, lam: float,
                      quantiles: list[float], ks: list[int]) -> list[dict]:
    subset = [
        r for r in rows
        if r["level"] == level and r["lambda_V"] == lam and r["delta_trace"]
    ]
    severities = np.concatenate([
        np.maximum(0.0, -np.asarray(r["delta_trace"], dtype=float))
        for r in subset
    ]) if subset else np.array([])

    out = []
    for q in quantiles:
        tau = conformal_quantile(severities, q)
        for k in ks:
            fires = []
            steps = []
            failures = []
            true_pos = false_pos = false_neg = 0
            for r in subset:
                fired, step = alarm_fire(np.asarray(r["delta_trace"], dtype=float), tau, k)
                failed = r["success"] < 0.5
                fires.append(float(fired))
                if step is not None:
                    steps.append(step)
                failures.append(float(failed))
                true_pos += int(fired and failed)
                false_pos += int(fired and not failed)
                false_neg += int((not fired) and failed)
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else float("nan")
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else float("nan")
            out.append({
                "level": level,
                "lambda_V": lam,
                "quantile": q,
                "threshold": tau,
                "k": k,
                "n": len(subset),
                "alarm_rate": float(np.mean(fires)) if fires else float("nan"),
                "mean_alarm_step": float(np.mean(steps)) if steps else float("nan"),
                "failure_rate": float(np.mean(failures)) if failures else float("nan"),
                "precision": precision,
                "recall": recall,
            })
    return out


def write_paired_tex(stats: dict, path: Path) -> None:
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}"

    def pp(x: float) -> str:
        return f"{100.0 * x:.2f}"

    lines = [
        "% Auto-generated by scripts/analyze_rocket_clf_paper.py",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Metric & MPC & CLF & $\Delta$ & 95\% CI & Wins & $p_{\mathrm{sf}}$ \\",
        r"\midrule",
        (
            "Violation rate & "
            f"{pct(stats['violation_rate_mpc_mean'])} & "
            f"{pct(stats['violation_rate_clf_mean'])} & "
            f"{pp(stats['violation_rate_delta_mean'])} pp & "
            f"[{pp(stats['violation_rate_delta_ci95_low'])}, "
            f"{pp(stats['violation_rate_delta_ci95_high'])}] & "
            f"{stats['violation_rate_wins_better']}/{stats['n_pairs']} & "
            f"{stats['violation_rate_p_better']:.4f} \\\\"
        ),
        (
            "Mean descent & "
            f"{stats['mean_descent_mpc_mean']:.4f} & "
            f"{stats['mean_descent_clf_mean']:.4f} & "
            f"{stats['mean_descent_delta_mean']:.4f} & "
            f"[{stats['mean_descent_delta_ci95_low']:.4f}, "
            f"{stats['mean_descent_delta_ci95_high']:.4f}] & "
            f"{stats['mean_descent_wins_better']}/{stats['n_pairs']} & "
            f"{stats['mean_descent_p_better']:.4f} \\\\"
        ),
        (
            "Final $V$ & "
            f"{stats['final_V_mpc_mean']:.1f} & "
            f"{stats['final_V_clf_mean']:.1f} & "
            f"{stats['final_V_delta_mean']:.1f} & "
            f"[{stats['final_V_delta_ci95_low']:.1f}, "
            f"{stats['final_V_delta_ci95_high']:.1f}] & "
            f"{stats['final_V_wins_better']}/{stats['n_pairs']} & "
            f"{stats['final_V_p_better']:.4f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_alarm_tex(rows: list[dict], path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/analyze_rocket_clf_paper.py",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$q$ & $\tau_q$ & $k$ & $n$ & Alarm rate & Mean alarm step & Failure rate \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['quantile']:.2f} & {r['threshold']:.4f} & {int(r['k'])} & "
            f"{int(r['n'])} & {100*r['alarm_rate']:.1f} & "
            f"{r['mean_alarm_step']:.1f} & {100*r['failure_rate']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rocket_jepa/clf_paired"))
    parser.add_argument("--level", default="extreme")
    parser.add_argument("--baseline-lambda", type=float, default=0.0)
    parser.add_argument("--clf-lambda", type=float, default=20.0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    rows = load_episode_rows(args.results_dir)
    pairs = paired(rows, args.level, args.baseline_lambda, args.clf_lambda)
    rng = np.random.default_rng(20260428)

    stats = {
        "level": args.level,
        "baseline_lambda": args.baseline_lambda,
        "clf_lambda": args.clf_lambda,
        "n_pairs": len(pairs),
        "paired_seeds": [p[0]["seed"] for p in pairs],
    }
    metric_directions = {
        "violation_rate": True,
        "mean_descent": False,
        "final_V": True,
        "max_V": True,
        "return": False,
    }
    for metric, lower_is_better in metric_directions.items():
        stats.update(metric_summary(pairs, metric, rng, args.n_boot, lower_is_better))
    diagnostic_directions = {
        "plan_pred_violation_rate": True,
        "plan_pred_mean_descent": False,
        "plan_clf_penalty": True,
    }
    for metric, lower_is_better in diagnostic_directions.items():
        if any(
            np.isfinite(p[0].get(metric, np.nan)) or np.isfinite(p[1].get(metric, np.nan))
            for p in pairs
        ):
            stats.update(metric_summary(pairs, metric, rng, args.n_boot, lower_is_better))
    stats["success_mpc_mean"] = float(np.mean([p[0]["success"] for p in pairs])) if pairs else float("nan")
    stats["success_clf_mean"] = float(np.mean([p[1]["success"] for p in pairs])) if pairs else float("nan")

    alarm_rows = alarm_calibration(
        rows, level=args.level, lam=args.baseline_lambda,
        quantiles=[0.50, 0.75, 0.90, 0.95], ks=[1, 3, 5],
    )
    recommended = next(
        (r for r in alarm_rows
         if r["n"] > 0 and np.isfinite(r["threshold"])
         and abs(r["quantile"] - 0.95) < 1e-9 and int(r["k"]) == 3),
        None,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    write_dict_csv([{k: v for k, v in stats.items() if k != "paired_seeds"}],
                   args.out_dir / "paired_stats.csv")
    write_paired_tex(stats, args.out_dir / "paired_stats_table.tex")
    write_dict_csv(alarm_rows, args.out_dir / "alarm_calibration.csv")
    write_alarm_tex(alarm_rows, args.out_dir / "alarm_calibration_table.tex")
    (args.out_dir / "alarm_calibration.json").write_text(json.dumps({
        "rows": alarm_rows,
        "recommended": recommended,
    }, indent=2))

    print(f"Loaded {len(rows)} episode rows from {args.results_dir}")
    print(f"Paired seeds: {stats['paired_seeds']}")
    print(f"Wrote paired stats to {args.out_dir / 'paired_stats.json'}")
    if recommended:
        print(
            "Recommended fallback alarm: "
            f"threshold={recommended['threshold']:.6f}, k={int(recommended['k'])}"
        )


if __name__ == "__main__":
    main()

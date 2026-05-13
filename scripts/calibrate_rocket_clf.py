"""Calibrate RocketJEPA CLF shield margins from paired eval JSONs.

The evaluator logs predicted first-step CLF descent at each replan and the
realized monitor trace for the executed rollout. This script estimates a
conservative epsilon for the shield:

    raw_violation = -delta_pred + eta + epsilon

Use the q90/q95 error quantile as ``--shield-calibration-epsilon`` in later
CLF-MPC runs.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _as_scalar(value):
    if isinstance(value, list):
        if not value:
            return None
        return _as_scalar(value[0])
    if value is None:
        return None
    return float(value)


def _predicted_exec_delta(rec, default_exec_idx: int):
    pred = _as_scalar(rec.get("pred_exec_delta"))
    if pred is not None:
        return pred, int(rec.get("pred_exec_idx", default_exec_idx))

    trace = rec.get("pred_delta_trace")
    if isinstance(trace, list) and trace:
        idx = int(rec.get("pred_exec_idx", default_exec_idx))
        idx = min(max(idx, 0), len(trace) - 1)
        return float(trace[idx]), idx

    pred = _as_scalar(rec.get("pred_first_delta"))
    if pred is not None:
        return pred, 0

    return None, default_exec_idx


def iter_errors(paths, default_exec_idx: int = 2):
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path) as f:
                data = json.load(f)
            for level, level_rec in data.get("by_level", {}).items():
                for ep in level_rec.get("episodes", []):
                    true = ep.get("delta_trace") or []
                    for rec in ep.get("plan_diagnostics", []):
                        if not rec.get("available", False):
                            continue
                        pred, pred_idx = _predicted_exec_delta(rec, default_exec_idx)
                        if pred is None:
                            continue
                        t = int(rec.get("t", 0))
                        if t < 0 or t >= len(true):
                            continue
                        actual = float(true[t])
                        yield {
                            "path": path,
                            "level": level,
                            "seed": ep.get("seed"),
                            "t": t,
                            "pred_idx": pred_idx,
                            "pred_delta": pred,
                            "actual_delta": actual,
                            "abs_error": abs(actual - pred),
                            "signed_error": actual - pred,
                        }


def main():
    p = argparse.ArgumentParser(description="Calibrate RocketJEPA CLF shield epsilon")
    p.add_argument("inputs", nargs="+", help="JSON files or glob patterns from evaluate_rocket_clf.py")
    p.add_argument("--out", default="/users/aiyer40/scratch/results_lejepa_v4/clf_calibration/rocket_clf_epsilon.json")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95])
    p.add_argument("--default-exec-idx", type=int, default=2,
                   help="Fallback predicted delta index for older JSONs without pred_exec_delta.")
    args = p.parse_args()

    rows = list(iter_errors(args.inputs, default_exec_idx=args.default_exec_idx))
    if not rows:
        raise SystemExit("No paired prediction/realization records found.")

    abs_err = np.asarray([r["abs_error"] for r in rows], dtype=np.float64)
    signed = np.asarray([r["signed_error"] for r in rows], dtype=np.float64)
    out = {
        "n": int(len(rows)),
        "mean_abs_error": float(abs_err.mean()),
        "max_abs_error": float(abs_err.max()),
        "mean_signed_error": float(signed.mean()),
        "quantiles": {
            f"q{int(q * 100):02d}": float(np.quantile(abs_err, q))
            for q in args.quantiles
        },
        "recommended_shield_calibration_epsilon": float(np.quantile(abs_err, 0.9)),
        "note": "Use q90 for less conservative sweeps, q95 for paper-grade robust shield.",
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"-> wrote {path}")


if __name__ == "__main__":
    main()

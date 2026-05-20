"""Plot V(t) trajectories on extreme disturbance: CEM vs adaptive.

Uses V_t_trace from CEM JSONs (results/paper_headline/p1_cem_extreme_*.json)
and safety_trace.V_t_trace from retune_v3 JSONs (results/paper_retune/c3_adaptive_v3_extreme_*.json).

Output: results/paper_figures/fig_v_trajectory.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="extreme")
    ap.add_argument("--out", default="results/paper_figures/fig_v_trajectory.pdf")
    args = ap.parse_args()

    cem_dir = Path("results/paper_headline")
    ada_dir = Path("results/paper_retune")

    # CEM traces
    cem_traces = []
    cem_succs = []
    for f in sorted(cem_dir.glob(f"p1_cem_{args.level}_seed*.json")):
        d = json.loads(f.read_text())
        for ep in d["by_level"][args.level].get("episodes", []):
            tr = ep.get("V_t_trace") or []
            if tr:
                cem_traces.append(tr)
                cem_succs.append(bool(ep.get("success")))

    # Adaptive traces
    ada_traces = []
    ada_succs = []
    for f in sorted(ada_dir.glob(f"c3_adaptive_v3_{args.level}_seed*.json")):
        d = json.loads(f.read_text())
        for ep in d.get("episodes", []):
            st = ep.get("safety_trace") or {}
            tr = st.get("V_t_trace") or []
            if tr:
                ada_traces.append(tr)
                ada_succs.append(bool(ep.get("success")))

    print(f"CEM:      {len(cem_traces)} episodes, {sum(cem_succs)} landings")
    print(f"adaptive: {len(ada_traces)} episodes, {sum(ada_succs)} landings")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)

    CEM_COLOR = "#4C72B0"
    ADA_COLOR = "#55A467"

    for ax, traces, succs, label, color in [
        (axes[0], cem_traces, cem_succs, "CEM (baseline)", CEM_COLOR),
        (axes[1], ada_traces, ada_succs, "ours (adaptive)", ADA_COLOR),
    ]:
        for tr, s in zip(traces, succs):
            alpha = 0.85 if s else 0.45
            ls = "-" if s else "--"
            ax.plot(tr, color=color, alpha=alpha, linestyle=ls, linewidth=1.5)
        # Mean ± std band
        if traces:
            max_len = max(len(t) for t in traces)
            mat = np.full((len(traces), max_len), np.nan)
            for i, tr in enumerate(traces):
                mat[i, :len(tr)] = tr
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            ax.plot(mean, color="black", linewidth=2.0, label="mean", zorder=10)
        n_land = sum(succs)
        n_total = len(succs)
        ax.set_title(f"{label}: landed {n_land}/{n_total}")
        ax.set_xlabel("control step")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$V(x_t)$ on realized trajectory")

    # Annotation
    cem_peak = np.nanmean([max(t) for t in cem_traces]) if cem_traces else 0
    ada_peak = np.nanmean([max(t) for t in ada_traces]) if ada_traces else 0
    fig.suptitle(
        f"Lyapunov trajectories on {args.level} disturbance "
        f"(peak $V$: CEM={cem_peak:.0f}, adaptive={ada_peak:.0f}, "
        f"{(cem_peak-ada_peak)/cem_peak*100:+.1f}\\% tighter)",
        fontsize=10, y=1.00,
    )

    # Legend handles
    from matplotlib.lines import Line2D
    h_land  = Line2D([0], [0], color="gray", linestyle="-",  alpha=0.85, label="landed")
    h_fail  = Line2D([0], [0], color="gray", linestyle="--", alpha=0.45, label="failed")
    h_mean  = Line2D([0], [0], color="black", linewidth=2.0,            label="mean")
    axes[1].legend(handles=[h_land, h_fail, h_mean], frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

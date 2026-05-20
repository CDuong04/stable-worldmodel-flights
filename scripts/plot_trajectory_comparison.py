"""Plot V(t) traces over a single episode: CEM vs adaptive at hard level.

Shows the bounded-trajectory narrative visually.  CEM's V trace wanders into
high values (30k+) before any descent; adaptive's V trace stays in a tighter
band.  This is the figure that captures the SMT practical-stability claim.

CEM JSONs contain V_t_trace.  Adaptive JSONs (from updated eval script) will
contain safety_trace.V_t_trace once re-run.  Until then, we plot CEM only and
overlay adaptive's max_V as a horizontal band.

Output: results/paper_figures/fig_v_traces.pdf
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
    ap.add_argument("--headline-dir", default="results/paper_headline")
    ap.add_argument("--level", default="hard")
    ap.add_argument("--out", default="results/paper_figures/fig_v_traces.pdf")
    args = ap.parse_args()

    sweep_dir = Path(args.headline_dir)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # --- CEM traces (have V_t_trace) ---
    cem_files = sorted(sweep_dir.glob(f"p1_cem_{args.level}_*.json"))
    if not cem_files:
        print(f"[plot] no CEM files for level {args.level}")
        return
    cem_traces = []
    cem_succs = []
    for f in cem_files:
        d = json.loads(f.read_text())
        for ep in d["by_level"][args.level].get("episodes", []):
            trace = ep.get("V_t_trace") or []
            if trace:
                cem_traces.append(trace)
                cem_succs.append(ep.get("success", False))

    for tr, succ in zip(cem_traces, cem_succs):
        c = "#4C72B0" if succ else "#C44E52"  # blue success / red fail
        alpha = 0.8 if succ else 0.45
        ls = "-" if succ else "--"
        ax.plot(tr, color=c, alpha=alpha, linestyle=ls, linewidth=1.2)

    # --- Adaptive: overlay max_V as horizontal band (until traces available) ---
    ada_files = sorted(sweep_dir.glob(f"c3_adaptive_{args.level}_*.json"))
    ada_max_v = []
    for f in ada_files:
        d = json.loads(f.read_text())
        for ep in d.get("episodes", []):
            mv = ep.get("safety", {}).get("max_V")
            if mv:
                ada_max_v.append(mv)
    cem_max_v = [max(t) for t in cem_traces if t]
    if ada_max_v and cem_max_v:
        # Horizontal lines showing peak max_V for each method.
        ax.axhline(np.mean(cem_max_v), color="#4C72B0", linestyle=":", linewidth=1.5,
                   label=f"CEM mean peak max_V ({np.mean(cem_max_v):.0f})")
        ax.axhline(np.mean(ada_max_v), color="#55A467", linestyle=":", linewidth=1.5,
                   label=f"adaptive mean peak max_V ({np.mean(ada_max_v):.0f})")

    # Legend handles for trajectory styles.
    blue_solid = plt.Line2D([0], [0], color="#4C72B0", linestyle="-", label="CEM (landed)")
    red_dash   = plt.Line2D([0], [0], color="#C44E52", linestyle="--", label="CEM (failed)")
    cem_band   = plt.Line2D([0], [0], color="#4C72B0", linestyle=":", linewidth=1.5,
                            label=f"CEM peak max_V ({np.mean(cem_max_v):.0f})" if cem_max_v else "CEM peak")
    ada_band   = plt.Line2D([0], [0], color="#55A467", linestyle=":", linewidth=1.5,
                            label=f"adaptive peak max_V ({np.mean(ada_max_v):.0f})" if ada_max_v else "adaptive peak")
    ax.legend(handles=[blue_solid, red_dash, cem_band, ada_band], frameon=False, fontsize=9, loc="upper left")

    ax.set_xlabel("Control step")
    ax.set_ylabel(r"$V(\hat{x}_t)$ along the realized trajectory")
    ax.set_title(f"Lyapunov traces on the rocket-landing task ({args.level} disturbance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=160)
    print(f"[plot] wrote {args.out}")
    print(f"[plot] CEM traces: {len(cem_traces)}  succ: {sum(cem_succs)}/{len(cem_succs)}")
    print(f"[plot] CEM peak max_V mean: {np.mean(cem_max_v):.1f}")
    print(f"[plot] adaptive peak max_V mean: {np.mean(ada_max_v):.1f}")


if __name__ == "__main__":
    main()

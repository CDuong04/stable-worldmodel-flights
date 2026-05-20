"""Plot cumulative reward over an episode, CEM vs adaptive on extreme.

Reward per step isn't stored in the existing JSONs, but V_t_trace serves as
an excellent proxy: lower V == closer to pad == higher reward.  We plot
1 / (1 + V_t) which inverts V into a reward-like signal (0 = bad, 1 = at pad).

For a true reward curve, re-running episodes with per-step reward logging is
needed (covered separately).

Output: results/paper_figures/fig_reward_curves.{pdf,png}
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
    ap.add_argument("--out", default="results/paper_figures/fig_reward_curves.pdf")
    args = ap.parse_args()

    cem_dir = Path("results/paper_headline")
    ada_dir = Path("results/paper_retune")

    cem_traces = []
    cem_succs = []
    for f in sorted(cem_dir.glob(f"p1_cem_{args.level}_seed*.json")):
        d = json.loads(f.read_text())
        for ep in d["by_level"][args.level].get("episodes", []):
            tr = ep.get("V_t_trace") or []
            if tr:
                cem_traces.append(np.array(tr))
                cem_succs.append(bool(ep.get("success")))

    ada_traces = []
    ada_succs = []
    for f in sorted(ada_dir.glob(f"c3_adaptive_v3_{args.level}_seed*.json")):
        d = json.loads(f.read_text())
        for ep in d.get("episodes", []):
            tr = (ep.get("safety_trace") or {}).get("V_t_trace") or []
            if tr:
                ada_traces.append(np.array(tr))
                ada_succs.append(bool(ep.get("success")))

    print(f"CEM:      {len(cem_traces)} episodes, {sum(cem_succs)} landings")
    print(f"adaptive: {len(ada_traces)} episodes, {sum(ada_succs)} landings")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Panel A: per-step reward proxy (1 / (1 + V)).
    CEM_COLOR = "#4C72B0"
    ADA_COLOR = "#55A467"
    for ax_ix, (traces, succs, label, color) in enumerate([
        (cem_traces, cem_succs, "CEM (baseline)", CEM_COLOR),
        (ada_traces, ada_succs, "ours (adaptive)", ADA_COLOR),
    ]):
        # Per-step reward proxy
        for tr, s in zip(traces, succs):
            r = 1.0 / (1.0 + tr)
            cum = np.cumsum(r)
            alpha = 0.7 if s else 0.35
            ls = "-" if s else "--"
            axes[0].plot(cum, color=color, alpha=alpha, linestyle=ls, linewidth=1.2)

    # Add mean cumulative for each method
    for traces, color, label in [(cem_traces, CEM_COLOR, "CEM mean"),
                                  (ada_traces, ADA_COLOR, "adaptive mean")]:
        if not traces:
            continue
        max_len = max(len(t) for t in traces)
        mat = np.full((len(traces), max_len), np.nan)
        for i, tr in enumerate(traces):
            r = 1.0 / (1.0 + np.array(tr))
            cum = np.cumsum(r)
            mat[i, :len(cum)] = cum
        mean = np.nanmean(mat, axis=0)
        axes[0].plot(mean, color=color, linewidth=2.5, label=label, zorder=10)

    axes[0].set_xlabel("control step")
    axes[0].set_ylabel(r"cumulative reward proxy $\sum_t \frac{1}{1+V(x_t)}$")
    axes[0].set_title("Cumulative goal-proximity reward")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel B: episode total returns
    methods = ["CEM", "ours"]
    means = []
    stds = []
    for traces in [cem_traces, ada_traces]:
        if not traces:
            means.append(0); stds.append(0); continue
        totals = [np.sum(1.0/(1.0+tr)) for tr in traces]
        means.append(np.mean(totals))
        stds.append(np.std(totals)/np.sqrt(len(totals)))

    x_pos = np.arange(len(methods))
    bars = axes[1].bar(x_pos, means, yerr=stds, capsize=6,
                       color=[CEM_COLOR, ADA_COLOR], alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("episode cumulative reward proxy")
    axes[1].set_title(f"Episode totals ({args.level}, 5 seeds)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

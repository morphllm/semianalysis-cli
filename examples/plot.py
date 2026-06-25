#!/usr/bin/env python3
"""Plot the serving economics emitted by `semianalysis --format json`.

This is a standalone *consumer* of the CLI's data — the CLI itself only returns
data; visualization lives here so the two stay decoupled.

    semianalysis minimax3 --format json --out data.json
    python examples/plot.py data.json -o chart.png
    # or pipe:
    semianalysis glm5 --json | python examples/plot.py -o chart.png

Two panels share the concurrency x-axis:
  • left  — $/1M output tokens vs concurrency (cost-vs-load curve)
  • right — output throughput per GPU vs concurrency (the raw engine number)

One line per (hardware, ISL/OSL) series; the cheapest output point is starred.

Requires matplotlib:  pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def _series_key(r: dict) -> str:
    return f"{r.get('hardware', '?')}  {r.get('isl', '?')}/{r.get('osl', '?')}"


def render(payload: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")  # headless: never needs a display server
    import matplotlib.pyplot as plt

    records = [r for r in payload.get("records", []) if r.get("cost_out") is not None]
    if not records:
        sys.exit("no priced records to plot")

    series: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        series[_series_key(r)].append(r)
    for rows in series.values():
        rows.sort(key=lambda r: r.get("conc", 0))

    cheapest = min(records, key=lambda r: r["cost_out"])

    fig, (ax_cost, ax_tput) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")

    for i, (key, rows) in enumerate(sorted(series.items())):
        color = cmap(i % 10)
        concs = [r.get("conc", 0) for r in rows]
        ax_cost.plot(concs, [r["cost_out"] for r in rows], marker="o", color=color, label=key, linewidth=1.8)
        ax_tput.plot(
            concs, [r.get("metrics", {}).get("output_tput_per_gpu", 0) for r in rows],
            marker="o", color=color, label=key, linewidth=1.8,
        )

    ax_cost.scatter(
        [cheapest.get("conc", 0)], [cheapest["cost_out"]],
        s=260, marker="*", color="#16a34a", zorder=5, edgecolors="white", linewidths=1.0,
        label=f"cheapest ${cheapest['cost_out']:.3f}",
    )

    ax_cost.set(title="Serving cost vs load", xlabel="Concurrency", ylabel="$ / 1M output tokens")
    ax_cost.set_xscale("log", base=2)
    ax_cost.set_yscale("log")
    ax_cost.grid(True, which="both", alpha=0.25)
    ax_cost.legend(fontsize=7, loc="best")

    ax_tput.set(title="Output throughput vs load", xlabel="Concurrency", ylabel="Output tok/s per GPU")
    ax_tput.set_xscale("log", base=2)
    ax_tput.grid(True, which="both", alpha=0.25)
    ax_tput.legend(fontsize=7, loc="best")

    recipe = f"{payload.get('framework', '?')}+{payload.get('spec', '?')}"
    fig.suptitle(
        f"{payload.get('model', '?')}   ·   {payload.get('date', '?')}   ·   "
        f"{payload.get('tier', '?')} GPU rate   ·   {recipe}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="-", help="JSON file from `semianalysis --json` (default: stdin).")
    ap.add_argument("-o", "--out", default="chart.png", help="Output PNG path (default: chart.png).")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.data == "-" else open(args.data).read()
    payload = json.loads(raw)
    path = render(payload, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""`semianalysis` — derive real serving cost from SemiAnalysis benchmarks.

Pulls SemiAnalysis InferenceX per-GPU throughput for an open model, prefers the
dynamo-sglang + MTP serving recipe (falls back and says so when unavailable),
and converts throughput → $/1M tokens at a GPU rental tier. With `--plot` it
also writes a cost-vs-load + throughput-vs-load chart.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import core as sa

console = Console()


def main(
    model: str = typer.Argument(
        help="Model: SemiAnalysis name or alias (glm5, dsv4, minimax3, kimi, qwen35, gptoss, ...).",
    ),
    date: str = typer.Option("latest", "--date", "-d", help="Benchmark date (YYYY-MM-DD), 'latest', or 'all'."),
    tier: str = typer.Option("neocloud", "--tier", help="GPU cost tier: hyperscaler, neocloud, or rental."),
    framework: str = typer.Option("dynamo-sglang", "--framework", "-f", help="Serving framework filter, or 'any'."),
    spec: str = typer.Option("mtp", "--spec", help="Speculative decode method: mtp, none, or 'any'."),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Restrict to one GPU (gb300, b200, mi355x, ...)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Max rows to show (0 = all; table format only)."),
    fmt: str = typer.Option("table", "--format", "-F", help="Output format: table, json, or csv."),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Write json/csv to this file instead of stdout."),
    json_out: bool = typer.Option(False, "--json", help="Shorthand for --format json."),
):
    """Derive actual $/1M-token serving cost from SemiAnalysis benchmarks.

    Pulls per-GPU throughput for the model, prefers the dynamo-sglang + MTP recipe
    (falls back and says so when unavailable), and converts throughput → $/1M tokens
    at a GPU rental tier. Emit a Rich table (default), JSON, or CSV — the json/csv
    rows carry the full raw benchmark record plus the derived cost fields.
    """
    fmt = "json" if json_out else fmt.lower()
    if fmt not in ("table", "json", "csv"):
        console.print(f"[red]Unknown format '{fmt}'[/red] [dim](use: table, json, csv)[/dim]")
        raise typer.Exit(1)
    if tier not in sa.TIERS:
        console.print(f"[red]Unknown tier '{tier}'[/red] [dim](use: {', '.join(sa.TIERS)})[/dim]")
        raise typer.Exit(1)

    sa_model = sa.resolve_model(model)
    try:
        all_records = sa.fetch_benchmarks(sa_model)
    except ValueError as e:
        console.print(f"[red]{_escape(str(e))}[/red]")
        console.print(f"[dim]Models: {', '.join(sa.KNOWN_MODELS)}[/dim]")
        console.print(f"[dim]Aliases: {', '.join(sorted(set(sa.MODEL_ALIASES)))}[/dim]")
        raise typer.Exit(1)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]SemiAnalysis fetch failed: {_escape(str(e))}[/red]")
        raise typer.Exit(1)

    if not all_records:
        console.print(f"[yellow]No benchmark records for {sa_model}[/yellow]")
        raise typer.Exit(1)

    if date == "all":
        records, use_date = all_records, "all"
    else:
        if date == "latest":
            # Prefer the latest date that actually has the requested recipe, so
            # "give me dynamo-sglang+mtp" doesn't silently fall back to a worse
            # framework just because the newest run skipped it.
            preferred, _ = sa.select(all_records, framework=framework, spec=spec, hardware=hardware)
            use_date = sa.latest_date(preferred) or sa.latest_date(all_records)
        else:
            use_date = date
        records = [r for r in all_records if r.get("date") == use_date]
        if not records:
            dates = sorted({r["date"] for r in all_records})
            console.print(f"[yellow]No records for {sa_model} on {use_date}[/yellow] [dim](have: {', '.join(dates)})[/dim]")
            raise typer.Exit(1)

    selected, notes = sa.select(records, framework=framework, spec=spec, hardware=hardware)
    enriched = [sa.enrich(r, tier) for r in selected]
    # Sort by hardware then concurrency for a readable cost-vs-load curve.
    enriched.sort(key=lambda r: (r.get("hardware", ""), r.get("conc", 0)))

    # Cheapest output cost (best margin) + lowest-latency interactive point.
    costed = [r for r in enriched if r.get("cost_out") is not None]
    cheapest = min(costed, key=lambda r: r["cost_out"]) if costed else None
    fastest = min(
        (r for r in enriched if r.get("metrics", {}).get("mean_ttft") is not None),
        key=lambda r: r["metrics"]["mean_ttft"], default=None,
    )

    if fmt == "json":
        payload = {
            "model": sa_model, "date": use_date, "tier": tier,
            "framework": framework, "spec": spec,
            "notes": notes, "records": enriched,
            "cheapest_output": cheapest,
        }
        text = json.dumps(payload, indent=2, default=str)
        _emit(text, out)
        return

    if fmt == "csv":
        text = _to_csv(enriched)
        _emit(text, out)
        return

    console.print(
        f"[bold]{sa_model}[/bold]  [dim]date={use_date}  tier={tier} "
        f"(${'/'.join(str(sa.GPU_COST.get(r['hardware'], {}).get(tier, '?')) for r in enriched[:1]) or '?'}/gpu/hr)  "
        f"recipe={framework}+{spec}[/dim]"
    )
    for n in notes:
        console.print(f"  [yellow]fallback:[/yellow] [dim]{_escape(n)}[/dim]")

    table = Table(show_lines=False, expand=False)
    table.add_column("HW", style="bold cyan", no_wrap=True)
    table.add_column("Recipe", no_wrap=True, style="dim")
    table.add_column("Conc", justify="right", no_wrap=True)
    table.add_column("ISL/OSL", justify="right", no_wrap=True, style="dim")
    table.add_column("tot/gpu", justify="right", no_wrap=True)
    table.add_column("out/gpu", justify="right", no_wrap=True)
    table.add_column("TTFT", justify="right", no_wrap=True, style="dim")
    table.add_column("$/1M in", justify="right", no_wrap=True)
    table.add_column("$/1M out", justify="right", no_wrap=True, style="bold")
    table.add_column("$/1M tot", justify="right", no_wrap=True)

    rows = enriched if limit <= 0 else enriched[:limit]
    for r in rows:
        m = r.get("metrics", {})
        ttft = m.get("mean_ttft")
        fw = r.get("framework", "?").replace("dynamo-", "dyn-")
        recipe = f"{fw}+{r.get('spec_method', '?')}" + ("/disagg" if r.get("disagg") else "")
        out_cell = _money(r["cost_out"])
        if cheapest is not None and r is cheapest:
            out_cell = f"[green]{out_cell}[/green]"
        table.add_row(
            r.get("hardware", "?"),
            recipe,
            str(r.get("conc", "?")),
            f"{r.get('isl', '?')}/{r.get('osl', '?')}",
            f"{m.get('tput_per_gpu', 0):.0f}",
            f"{m.get('output_tput_per_gpu', 0):.0f}",
            f"{ttft:.1f}s" if ttft is not None else "-",
            _money(r["cost_in"]),
            out_cell,
            _money(r["cost_total"]),
        )
    console.print(table)
    if limit > 0 and len(enriched) > limit:
        console.print(f"[dim]… {len(enriched) - limit} more rows (use --limit 0)[/dim]")

    # ── Summary callouts ──
    if cheapest:
        console.print(
            f"\n[bold green]Cheapest output[/bold green]: "
            f"[bold]{_money(cheapest['cost_out'])}/1M[/bold] out  "
            f"[dim]({_money(cheapest['cost_in'])} in, {_money(cheapest['cost_total'])} total) "
            f"@ conc={cheapest.get('conc')} on {cheapest.get('hardware')} "
            f"({cheapest.get('framework')}/{cheapest.get('spec_method')})[/dim]"
        )
    if fastest:
        fm = fastest["metrics"]
        console.print(
            f"[bold]Fastest interactive[/bold]: TTFT {fm.get('mean_ttft', 0):.2f}s, "
            f"{fm.get('mean_tpot', 0) * 1000:.1f}ms/tok @ conc={fastest.get('conc')} "
            f"[dim]→ {_money(fastest.get('cost_out'))}/1M out[/dim]"
        )
    console.print("[dim]→ --format json|csv for the full raw benchmark data.[/dim]")


def _emit(text: str, out: Optional[str]) -> None:
    """Write `text` to a file when `out` is given, else stdout.

    Plain stdout, not console.print — Rich would hard-wrap and corrupt JSON/CSV.
    """
    if out:
        with open(out, "w", newline="") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        console.print(f"[green]wrote[/green] {out}")
    else:
        print(text)


def _to_csv(records: list[dict]) -> str:
    """Flatten enriched records to CSV: scalar top-level cols + metrics.* + costs.

    `metrics` is exploded into `metric_<name>` columns; the column set is the
    union across all rows so heterogeneous records still line up.
    """
    flat_rows = []
    for r in records:
        flat = {k: v for k, v in r.items() if k not in ("metrics", "workers")}
        for mk, mv in (r.get("metrics") or {}).items():
            flat[f"metric_{mk}"] = mv
        flat_rows.append(flat)

    columns: list[str] = []
    seen = set()
    for row in flat_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                columns.append(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat_rows)
    return buf.getvalue()


def _money(v: Optional[float]) -> str:
    """Format a $/1M-token figure with sensible precision, or '-' when unknown."""
    if v is None:
        return "-"
    if v >= 100:
        return f"${v:.0f}"
    if v >= 1:
        return f"${v:.2f}"
    return f"${v:.3f}"


def _escape(text: str) -> str:
    """Escape Rich markup characters."""
    return text.replace("[", "\\[").replace("]", "\\]")


def run() -> None:
    """Console-script entry point: single command, no sub-command name."""
    typer.run(main)


if __name__ == "__main__":
    run()

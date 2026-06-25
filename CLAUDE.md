# semianalysis-cli

A small, single-command CLI that turns [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/inference) per-GPU throughput benchmarks into real `$/1M-token` serving cost. It returns **data** — Rich table, JSON, or CSV. Nothing else.

## Layout

```
semianalysis_cli/
  __init__.py    # version + package docstring
  core.py        # data layer: fetch, filter, cost math. Deps: requests only.
  cli.py         # presentation: typer command, table/json/csv rendering.
examples/
  plot.py        # standalone matplotlib consumer of the CLI's JSON. NOT imported by the CLI.
docs/
  example-chart.png
pyproject.toml   # console script `semianalysis`; [plot] extra = matplotlib
```

## Design rules (hold these)

- **`core.py` is pure data and has no presentation or heavy deps.** It fetches from the SemiAnalysis API, filters to a serving recipe, and computes cost. It must not import typer, rich, or matplotlib. Keep its only third-party dep `requests`.
- **The CLI returns data; it does not visualize.** Output is table / JSON / CSV. Do **not** add plotting, image, or charting logic to `cli.py` or `core.py`. Visualization is a separate concern that lives in `examples/plot.py` and consumes the CLI's JSON. Keep that boundary.
- **`enrich()` is lossless.** It does `out = dict(record)` and only *adds* `gpu_hr` + `cost_in/out/total`. Never drop raw fields — JSON/CSV consumers rely on the full benchmark record (throughput, full latency distribution, power, parallelism) being present.
- **Filtering never silently degrades.** `select()` prefers `dynamo-sglang + MTP`, but when a model/date lacks it, it relaxes one axis at a time and appends a human-readable note. Preserve the "announce the fallback" behavior — don't swap in worse numbers quietly.
- **Model names are exact API strings.** The SemiAnalysis API 400s on display tokens. Friendly aliases (`glm5`, `dsv4`, `minimax3`, …) live in `core.MODEL_ALIASES`; `KNOWN_MODELS` derives from its values. Add new models there.
- **GPU rates** are SemiAnalysis's own three-tier map in `core.GPU_COST` (hyperscaler / neocloud / rental). That's the single place to adjust rates.

## Cost model

```
$/1M tok = gpu_cost_per_hour * 1e6 / (tput_per_gpu * 3600)
```

computed separately for input (`input_tput_per_gpu`), output (`output_tput_per_gpu`), and blended total (`tput_per_gpu`).

## Output formats

| Format | Renderer | Notes |
|---|---|---|
| `table` (default) | `rich` | Human view + cheapest-output / fastest-interactive callouts. Curated columns. |
| `json` (`--json`) | `json.dumps` via `print` | Envelope `{model, date, tier, framework, spec, notes, records, cheapest_output}`. **Never** `console.print` JSON — Rich hard-wraps and corrupts it; use the plain `_emit` helper. |
| `csv` | `csv.DictWriter` via `_emit` | One flat row/record; `metrics.*` → `metric_<name>` columns; column set is the union across rows. |

`--out PATH` routes json/csv to a file (via `_emit`), else stdout.

## Local dev

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[plot]"          # CLI + matplotlib for the example

semianalysis minimax3 --limit 4   # smoke test (hits the live API)
semianalysis glm5 --format csv --out /tmp/glm5.csv
semianalysis glm5 --json | python examples/plot.py -o /tmp/chart.png
```

There's no test suite or network mock yet — verification is running the CLI against the live API. The API gzip-compresses responses (`requests` handles it; a bare `curl` needs `--compressed` + a `referer: https://inferencex.semianalysis.com/inference` header).

## Provenance

Extracted from the Morph `tab` monorepo's `testapi semianalysis` command. The in-monorepo copy still exists and retains an OpenRouter price-comparison surface that was removed here; the two are not auto-synced.

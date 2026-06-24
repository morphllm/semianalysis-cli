# semianalysis-cli

What does it actually cost to serve an open model, and how does that compare to what OpenRouter charges?

[SemiAnalysis InferenceX](https://inferencex.semianalysis.com/inference) publishes per-GPU throughput benchmarks for open models across hardware, framework, precision, and speculative-decode method. This CLI pulls those numbers, converts throughput to `$/1M tokens` at a GPU rental rate, and holds the result against live OpenRouter provider prices.

```
$/1M tok = gpu_cost_per_hour * 1e6 / (tput_per_gpu * 3600)
```

computed separately for input, output, and blended total.

## Install

```bash
pip install git+https://github.com/morphllm/semianalysis-cli.git
```

Or from a clone:

```bash
git clone https://github.com/morphllm/semianalysis-cli.git
cd semianalysis-cli
pip install -e .
```

## Usage

```bash
semianalysis minimax3                    # latest dynamo-sglang+MTP recipe, neocloud GPU rate
semianalysis glm5 --tier rental          # hyperscaler | neocloud | rental GPU rates
semianalysis dsv4 --hardware gb300       # restrict to one GPU
semianalysis glm5 --date all             # every benchmark date, not just latest
semianalysis kimi --framework any --spec any         # drop the recipe preference
semianalysis minimax3 --openrouter minimax/minimax-m3  # override the price-comparison slug
semianalysis glm5 --json                 # machine-readable (rows carry cost_in/out/total)
```

### What it prints

- A per-config table (one row per concurrency × ISL/OSL point), sorted by hardware then load, with `$/1M` in/out/total.
- **Cheapest output** operating point (highest throughput = best margin), highlighted.
- **Fastest interactive** point (lowest TTFT).
- The **OpenRouter** comparison: list price, cheapest provider, and how many × below your serving cost sits.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--date`, `-d` | `latest` | Benchmark date (`YYYY-MM-DD`), `latest`, or `all`. `latest` is recipe-aware. |
| `--tier` | `neocloud` | GPU cost tier: `hyperscaler`, `neocloud`, or `rental`. |
| `--framework`, `-f` | `dynamo-sglang` | Serving framework filter, or `any`. |
| `--spec` | `mtp` | Speculative-decode method: `mtp`, `none`, or `any`. |
| `--hardware`, `-hw` | — | Restrict to one GPU (`gb300`, `b200`, `mi355x`, …). |
| `--openrouter`, `-or` | — | Override the OpenRouter slug for the price comparison. |
| `--limit`, `-l` | `0` | Max table rows (`0` = all). |
| `--json` | off | Machine-readable output (plain stdout, safe to pipe). |

## Notes

- **Model names are exact API strings, not display tokens** (`Qwen-3.5-397B-A17B`, `DeepSeek-V4-Pro`, `MiniMax-M3`). Use the friendly aliases (`glm5`, `dsv4`, `minimax3`, `kimi`, `qwen35`, `gptoss`, …); a wrong name 400s.
- **Recipe preference, never silent fallback.** The default filter is `dynamo-sglang + MTP` (the fastest serving recipe). When a model/date has no run for it, the filter relaxes one axis at a time and prints a `fallback:` note — it never quietly swaps in worse numbers.
- **GPU rates** are SemiAnalysis's own three-tier map (Owning/Hyperscaler, Owning/Neocloud, Rental), baked into `GPU_COST` in `semianalysis_cli/core.py`. Adjust there if your rates differ.
- The SemiAnalysis API gzip-compresses every response; `requests` handles it. A bare `curl` needs `--compressed` plus a `referer: https://inferencex.semianalysis.com/inference` header.

## License

MIT

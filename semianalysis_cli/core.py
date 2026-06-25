"""SemiAnalysis InferenceX cost analysis.

SemiAnalysis (inferencex.semianalysis.com) publishes per-GPU throughput
benchmarks for open models across hardware/framework/precision/spec-method.
From a throughput number and a GPU rental rate we can derive the *actual*
serving cost in $/1M tokens.

Cost model (lifted verbatim from their bundle):

    $/1M tok = gpu_cost_per_hour * 1e6 / (tput_per_gpu * 3600)

computed separately for input (`input_tput_per_gpu`), output
(`output_tput_per_gpu`), and blended total (`tput_per_gpu`). Three GPU-rate
tiers ship in their UI — Owning/Hyperscaler (`costh`), Owning/Neocloud
(`costn`), Rental (`costr`) — the `GPU_COST` table below is their map.

The operating point we care about is **dynamo-sglang + MTP speculative
decoding** (the fastest serving recipe). When a model/date has no run for
that combo, `select()` falls back to the best available and says so.
"""

from __future__ import annotations

from typing import Optional

import requests

# ── SemiAnalysis ───────────────────────────────────────────────────

SA_BASE = "https://inferencex.semianalysis.com/api/v1"
SA_HEADERS = {"accept": "*/*", "referer": "https://inferencex.semianalysis.com/inference"}

# Per-GPU rental rate, $/hr, by hardware key. Three tiers as SemiAnalysis
# models them. Source: inferencex bundle HARDWARE registry (costh/costn/costr).
GPU_COST = {
    "h100":   {"hyperscaler": 1.30, "neocloud": 1.69, "rental": 1.30},
    "h200":   {"hyperscaler": 1.41, "neocloud": 1.74, "rental": 1.60},
    "b200":   {"hyperscaler": 1.95, "neocloud": 2.34, "rental": 2.90},
    "b300":   {"hyperscaler": 2.34, "neocloud": 2.808, "rental": 3.48},
    "gb200":  {"hyperscaler": 2.21, "neocloud": 2.75, "rental": 3.30},
    "gb300":  {"hyperscaler": 2.652, "neocloud": 3.30, "rental": 3.96},
    "mi300x": {"hyperscaler": 1.12, "neocloud": 1.40, "rental": 1.55},
    "mi325x": {"hyperscaler": 1.28, "neocloud": 1.59, "rental": 1.80},
    "mi355x": {"hyperscaler": 1.48, "neocloud": 1.90, "rental": 2.10},
}

TIERS = ("hyperscaler", "neocloud", "rental")

# Friendly aliases (what an operator types) → exact SemiAnalysis model name.
# The API name is NOT the display token: it 400s on anything but these exact
# strings (sourced from the bundle's Model enum).
MODEL_ALIASES = {
    "glm5": "GLM-5", "glm-5": "GLM-5", "glm": "GLM-5", "glm52": "GLM-5",
    "dsv4": "DeepSeek-V4-Pro", "deepseek": "DeepSeek-V4-Pro", "dsv4flash": "DeepSeek-V4-Pro",
    "deepseekv4": "DeepSeek-V4-Pro", "dsv4pro": "DeepSeek-V4-Pro", "deepseek-v4": "DeepSeek-V4-Pro",
    "dsr1": "DeepSeek-R1-0528", "deepseek-r1": "DeepSeek-R1-0528",
    "minimax3": "MiniMax-M3", "minimaxm3": "MiniMax-M3", "m3": "MiniMax-M3", "minimax-m3": "MiniMax-M3",
    "minimax": "MiniMax-M2.5", "minimax25": "MiniMax-M2.5", "m25": "MiniMax-M2.5",
    "kimi": "Kimi-K2.5", "kimik2": "Kimi-K2.5", "k25": "Kimi-K2.5",
    "qwen35": "Qwen-3.5-397B-A17B", "qwen3.5": "Qwen-3.5-397B-A17B", "qwen": "Qwen-3.5-397B-A17B",
    "gptoss": "gpt-oss-120b", "gpt-oss": "gpt-oss-120b", "oss120b": "gpt-oss-120b",
    "llama70b": "Llama-3.3-70B-Instruct-FP8", "llama3.3": "Llama-3.3-70B-Instruct-FP8",
    "llama3.1": "Llama-3.1-70B-Instruct-FP8-KV",
}

# Exact SemiAnalysis API model names (for help text / validation), derived from
# the alias targets so there's one source of truth.
KNOWN_MODELS = sorted(set(MODEL_ALIASES.values()))


def resolve_model(name: str) -> str:
    """Map a friendly alias to the SemiAnalysis model name (pass-through if unknown)."""
    return MODEL_ALIASES.get(name.strip().lower(), name)


def fetch_benchmarks(model: str, date: Optional[str] = None, timeout: int = 30) -> list[dict]:
    """Fetch benchmark records for a model. `date=None` returns every date.

    Raises ValueError on the API's `{"error": "Unknown model"}` sentinel.
    """
    params = {"model": model}
    if date:
        params["date"] = date
        params["exact"] = "true"
    r = requests.get(f"{SA_BASE}/benchmarks", params=params, headers=SA_HEADERS, timeout=timeout)
    # The API answers an unknown model with either 400 or a 200 {"error": ...}.
    if r.status_code in (400, 404):
        raise ValueError(f"unknown model '{model}' (exact API name required)")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(f"{model}: {data['error']}")
    return data


def latest_date(records: list[dict]) -> Optional[str]:
    return max((r["date"] for r in records if r.get("date")), default=None)


def cost_per_mtok(tput_per_gpu: float, gpu_hr: float) -> Optional[float]:
    """$/1M tokens for a per-GPU throughput (tok/s/gpu) at a GPU rate ($/hr)."""
    if not tput_per_gpu:
        return None
    return gpu_hr * 1e6 / (tput_per_gpu * 3600.0)


def enrich(record: dict, tier: str) -> dict:
    """Attach $/1M input/output/total to a record for the given cost tier.

    Records on hardware without a known rate get None costs (still listed).
    """
    hw = record.get("hardware", "")
    gpu_hr = GPU_COST.get(hw, {}).get(tier)
    m = record.get("metrics", {})
    out = dict(record)
    out["gpu_hr"] = gpu_hr
    if gpu_hr is None:
        out["cost_in"] = out["cost_out"] = out["cost_total"] = None
    else:
        out["cost_in"] = cost_per_mtok(m.get("input_tput_per_gpu"), gpu_hr)
        out["cost_out"] = cost_per_mtok(m.get("output_tput_per_gpu"), gpu_hr)
        out["cost_total"] = cost_per_mtok(m.get("tput_per_gpu"), gpu_hr)
    return out


def select(
    records: list[dict],
    framework: str = "dynamo-sglang",
    spec: str = "mtp",
    hardware: Optional[str] = None,
) -> tuple[list[dict], list[str]]:
    """Filter records to the preferred serving recipe, falling back gracefully.

    Returns (filtered_records, notes). `framework`/`spec` accept "any" to skip
    that filter. When the preferred framework or spec yields nothing, the
    filter is relaxed one axis at a time and a note records what happened.
    """
    notes: list[str] = []
    pool = records
    if hardware:
        pool = [r for r in pool if r.get("hardware") == hardware]
        if not pool:
            notes.append(f"no records on {hardware}; ignoring hardware filter")
            pool = records

    def by_framework(rs, fw):
        return rs if fw == "any" else [r for r in rs if r.get("framework") == fw]

    def by_spec(rs, sp):
        return rs if sp == "any" else [r for r in rs if r.get("spec_method") == sp]

    fw_pool = by_framework(pool, framework)
    if not fw_pool and framework != "any":
        avail = sorted({r.get("framework") for r in pool})
        notes.append(f"no '{framework}' runs; using all frameworks {avail}")
        fw_pool = pool

    sp_pool = by_spec(fw_pool, spec)
    if not sp_pool and spec != "any":
        avail = sorted({r.get("spec_method") for r in fw_pool})
        notes.append(f"no spec='{spec}' runs in that set; using {avail}")
        sp_pool = fw_pool

    return sp_pool, notes

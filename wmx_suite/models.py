"""Model registry — reads HF-cache config.json and classifies each model's memory behavior.

Key distinction (learned the hard way):
  * Sliding-window models (Gemma, GPT-OSS) get a RotatingKVCache, which MLX CANNOT
    quantize (`to_quantized` raises NotImplementedError). Forcing --kv-bits 4 on them
    crashes any prompt >= quantized_kv_start (5000) tokens. They must run fp16 KV.
    Their sliding window caps most layers, so KV growth is already gentle.
  * Linear+full models (Qwen3.5 9B/27B) get a standard KVCache and quantize fine;
    4-bit meaningfully lowers high-context memory.

Only `full_attention` layers grow unboundedly with context. `sliding_attention` layers
are window-capped and `linear_attention` layers use a fixed recurrent state, so neither
contributes to context-scaling KV growth.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

HUB = os.path.expanduser("~/.cache/huggingface/hub")
UNBOUNDED_CUSTOM_CACHE_TYPES = {"qwen3_5", "qwen3_5_text"}


@dataclass
class ModelInfo:
    hf_id: str
    weights_gb: float
    n_layers: int
    growing_layers: int          # full_attention layers — the ones whose KV grows with context
    kv_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    max_context: int | None
    cache_type: str              # "RotatingKVCache" | "standard"
    can_quantize_kv: bool        # False => must run fp16 KV (do NOT pass --kv-bits 4)
    layer_types: dict
    max_kv_size_enforced: bool = True

    def fp16_kv_bytes_per_token(self) -> float:
        """Analytic fp16 KV-cache growth per token (K and V), counting only growing layers.
        Matches measured mlx_true slope closely on Gemma/Qwen."""
        if not (self.kv_heads and self.head_dim):
            return 0.0
        return self.growing_layers * self.kv_heads * self.head_dim * 2 * 2  # 2=(K,V), 2 bytes fp16

    def as_dict(self) -> dict:
        d = asdict(self)
        d["layer_types"] = json.dumps(self.layer_types)
        return d


def _cache_dir(hf_id: str) -> str:
    return os.path.join(HUB, "models--" + hf_id.replace("/", "--"))


def cache_updated_at(hf_id: str) -> float | None:
    """Newest artifact mtime in any locally cached model snapshot."""
    root = _cache_dir(hf_id)
    if not os.path.isdir(root):
        return None
    latest: float | None = None
    for dirpath, _, filenames in os.walk(os.path.join(root, "snapshots")):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                mtime = max(os.lstat(path).st_mtime, os.path.getmtime(path))
            except OSError:
                continue
            latest = mtime if latest is None else max(latest, mtime)
    return latest


def fit_is_stale(hf_id: str, characterized_at: str | None) -> bool:
    """Whether cache artifacts are newer than a characterization run."""
    if not characterized_at:
        return False
    try:
        characterized = datetime.fromisoformat(characterized_at)
    except ValueError:
        return False
    if characterized.tzinfo is None:
        characterized = characterized.replace(tzinfo=timezone.utc)
    cache_mtime = cache_updated_at(hf_id)
    if cache_mtime is None:
        return False
    # DB timestamps use second precision; avoid false positives within that second.
    return cache_mtime > characterized.timestamp() + 1.0


def weights_gb(hf_id: str) -> float:
    """Real on-disk weight size from the cache `blobs/` dir (not the symlinked snapshot)."""
    blobs = os.path.join(_cache_dir(hf_id), "blobs")
    total = 0
    for f in glob.glob(os.path.join(blobs, "*")):
        if os.path.islink(f) or not os.path.isfile(f):
            continue
        # weights only — skip tokenizer/json blobs by size heuristic is unreliable; sum all
        total += os.path.getsize(f)
    return total / 1e9


def _read_raw_config(hf_id: str) -> dict | None:
    cfgs = glob.glob(os.path.join(_cache_dir(hf_id), "snapshots", "*", "config.json"))
    if not cfgs:
        return None
    with open(cfgs[0]) as fh:
        return json.load(fh)


def _read_config(hf_id: str) -> dict | None:
    c = _read_raw_config(hf_id)
    if c is None:
        return None
    # Gemma/VLM nest the text model config
    return c.get("text_config", c) if isinstance(c.get("text_config"), dict) else c


def describe(hf_id: str) -> ModelInfo | None:
    raw = _read_raw_config(hf_id)
    if raw is None:
        return None
    t = raw.get("text_config", raw) if isinstance(raw.get("text_config"), dict) else raw
    layer_types = t.get("layer_types", []) or []
    lt = Counter(layer_types)
    n_layers = t.get("num_hidden_layers", len(layer_types))
    growing = lt.get("full_attention", 0) if layer_types else n_layers
    sliding_enabled = t.get("use_sliding_window", True)
    has_sliding = (
        (sliding_enabled is not False and bool(t.get("sliding_window")))
        or lt.get("sliding_attention", 0) > 0
    )
    return ModelInfo(
        hf_id=hf_id,
        weights_gb=round(weights_gb(hf_id), 2),
        n_layers=n_layers,
        growing_layers=growing,
        kv_heads=t.get("num_key_value_heads"),
        head_dim=t.get("head_dim"),
        hidden_size=t.get("hidden_size"),
        max_context=t.get("max_position_embeddings"),
        cache_type="RotatingKVCache" if has_sliding else "standard",
        can_quantize_kv=not has_sliding,
        layer_types=dict(lt),
        max_kv_size_enforced=(
            raw.get("model_type") not in UNBOUNDED_CUSTOM_CACHE_TYPES
            and t.get("model_type") not in UNBOUNDED_CUSTOM_CACHE_TYPES
        ),
    )


def scan_cache() -> list[str]:
    """Return hf_ids of all mlx-community models present in the local HF cache."""
    out = []
    for d in glob.glob(os.path.join(HUB, "models--mlx-community--*")):
        name = os.path.basename(d).replace("models--", "").replace("--", "/", 1).replace("--", "-")
        # reconstruct: models--org--rest -> org/rest
        base = os.path.basename(d)[len("models--"):]
        org, _, rest = base.partition("--")
        out.append(f"{org}/{rest}")
    return sorted(set(out))

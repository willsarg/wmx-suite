"""Render function for ``wmx-suite show <hf_id>``.

Data schema
-----------
``data`` is a dict with the following keys:

    hf_id              : str   — full HuggingFace model ID
    weights_gb         : float — human-rounded weight size in GB (e.g. 12.1)
    n_layers           : int   — total transformer layers
    growing_layers     : int   — layers whose KV cache grows with context
    max_context        : int   — architecture token limit (e.g. 131072)
    kv_label           : str   — human KV-cache label, one of:
                                   "sliding-window (RotatingKVCache)"
                                   "standard (KVCache)"
    can_quantize_kv    : bool  — whether the KV cache can be 4-bit compressed
    growth_gb_per_1k   : float — approximate memory growth per 1k tokens (e.g. 0.025)

    # verbose-only raw fields
    cache_type         : str   — raw MLX cache class name
    kv_heads           : int
    head_dim           : int
    hidden_size        : int
    layer_types        : str   — repr of layer-type dict (e.g. '{"sliding_attention": 12, ...}')
    max_kv_size_enforced : bool
    is_causal          : bool
    fp16_kv_bytes_per_token : int  — raw bytes per token (e.g. 24576)
"""
from __future__ import annotations

_PREFIX = "mlx-community/"


def _strip(hf_id: str) -> str:
    return hf_id[len(_PREFIX):] if hf_id.startswith(_PREFIX) else hf_id


def render(console, data: dict) -> None:
    """Render model architecture + memory class to *console*.

    Normal output: model name as accent header, two field groups
    ("what it is" / "how its memory behaves"), next_block.

    Verbose appends the raw architecture dump as a ``raw`` appendix.
    """
    hf_id = data["hf_id"]
    weights_gb = data["weights_gb"]
    n_layers = data["n_layers"]
    growing_layers = data["growing_layers"]
    max_context = data["max_context"]
    kv_label = data["kv_label"]
    can_quantize_kv = data["can_quantize_kv"]
    growth_gb_per_1k = data["growth_gb_per_1k"]

    # Model name as accent header
    console.emit(console.style("accent", hf_id))
    console.emit()

    # --- Group 1: what it is ---
    console.emit(console.section("what it is"))
    console.emit(console.field(
        "weights", f"{weights_gb:.1f} GB",
        "size it occupies just to load, before any context",
    ))
    console.emit(console.field(
        "layers", str(n_layers),
        f"{growing_layers} of them grow with context; the rest are fixed",
    ))
    console.emit(console.field(
        "max context", f"{max_context:,} tok",
        "architecture limit — NOT your safe limit (run characterize)",
    ))

    console.emit()

    # --- Group 2: how its memory behaves ---
    console.emit(console.section("how its memory behaves"))

    if not can_quantize_kv:
        kv_val = (
            console.style("warn", "sliding-window")
            + console.style("dim", " (RotatingKVCache)")
        )
        kv_gloss = "this type can't be 4-bit compressed — it runs at fp16"
    else:
        kv_val = (
            console.style("good", "standard")
            + console.style("dim", " (KVCache)")
        )
        kv_gloss = "can be compressed to 4-bit → more context fits in the same memory"

    console.emit(console.field("KV cache", kv_val, kv_gloss))
    console.emit(console.field(
        "growth",
        f"≈{growth_gb_per_1k:.3f} GB / 1k tok",
        "how fast memory climbs as the conversation grows"
        + ("" if can_quantize_kv else " (fp16)"),
    ))

    # --- Verbose appendix ---
    raw_lines = []
    raw_fields = [
        ("n_layers",              str(data["n_layers"])),
        ("growing_layers",        str(data["growing_layers"])),
        ("kv_heads",              str(data["kv_heads"])),
        ("head_dim",              str(data["head_dim"])),
        ("hidden_size",           str(data["hidden_size"])),
        ("cache_type",            data["cache_type"]),
        ("can_quantize_kv",       str(data["can_quantize_kv"])),
        ("layer_types",           data["layer_types"]),
        ("max_kv_size_enforced",  str(data["max_kv_size_enforced"])),
        ("is_causal",             str(data["is_causal"])),
        ("fp16 KV bytes/token",   str(data["fp16_kv_bytes_per_token"])),
    ]
    for key, val in raw_fields:
        raw_lines.append(f"  {key:<20} {val}")

    raw_out = console.raw("raw architecture (--verbose)", raw_lines)
    if raw_out:
        console.emit()
        console.emit(raw_out)

    # --- next_block ---
    short = _strip(hf_id)
    console.emit(console.next_block([
        (f"wmx-suite characterize {short}",
         "find its safe context ceiling on your Mac"),
        (f"wmx-suite run --model {short} --dry-run",
         "preview the launch plan"),
    ]))

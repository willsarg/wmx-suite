"""Render function for ``wmx-suite scan``.

Data schema
-----------
``data`` is a dict with the following keys:

    models : list[dict]
        Each dict has:
            hf_id        : str   — full HuggingFace model ID
                                   (e.g. "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
            weights_gb   : float — size on disk rounded to 1 dp (e.g. 5.0)
            kv_label     : str   — human label: "quantizable KV" or
                                   "fp16-only KV (sliding-window cache)"
            cache_type   : str   — raw cache class name (e.g. "KVCache",
                                   "RotatingKVCache")
            weights_gb_exact : float — unrounded weight size (e.g. 4.98)

    registered : int  — count of models successfully registered into the DB
"""
from __future__ import annotations

_PREFIX = "mlx-community/"


def _strip(hf_id: str) -> str:
    return hf_id[len(_PREFIX):] if hf_id.startswith(_PREFIX) else hf_id


def render(console, data: dict) -> None:
    """Render the scan result to *console*.

    Normal output:
        Section header, one ✓ row per model (short name, human GB, KV label),
        registered count, inline gloss, next_block.

    Verbose appends cache_type and exact GB in brackets after the KV label.
    """
    models = data["models"]
    registered = data["registered"]

    # Column widths
    names = [_strip(m["hf_id"]) for m in models]
    name_w = max((len(n) for n in names), default=0)

    console.emit(console.section("Found these MLX models in your Hugging Face cache:"))
    console.emit()

    for m, short in zip(models, names):
        human_gb = f"{m['weights_gb']:.1f} GB"
        kv = console.style("dim", m["kv_label"])
        line = (
            "  " + console.glyph("ok") + "  "
            + short.ljust(name_w)
            + "   " + human_gb.rjust(8)
            + "   " + kv
        )
        if console.verbose:
            detail = f"   [{m['cache_type']}, {m['weights_gb_exact']:.2f} GB]"
            line += console.style("dim", detail)
        console.emit(line)

    console.emit()
    console.emit(console.style("good", f"registered {registered} models"))

    # Inline gloss (always shown in normal and verbose)
    console.emit(console.style("gloss",
        "  quantizable KV   can compress its context cache to 4-bit → more context fits\n"
        "  fp16-only KV     cache must stay full precision → memory climbs faster"
    ))

    console.emit(console.next_block([
        ("wmx-suite characterize <model>",
         "measure each one's safe context ceiling"),
    ]))

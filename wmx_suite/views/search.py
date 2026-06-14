"""Render function for ``wmx-suite search`` — Hub model discovery.

Data schema
-----------
``data`` is a dict with:

    query   : str        — the search query
    author  : str | None — author/org filter (e.g. "mlx-community"), or None
    limit   : int        — max results requested
    count   : int        — number of results returned
    results : list[dict] — each:
        id         : str   — full HF repo id (e.g. "mlx-community/gemma-4-e4b-it-4bit")
        downloads  : int
        likes      : int
        quant      : str   — quantization label ("4-bit"/"8-bit"/"bf16"/"—")
        mlx        : bool   — whether it's an MLX build

Verbose appends a raw appendix with the full ids + the pipeline tag.
"""
from __future__ import annotations


def _short(model_id: str, author: str | None) -> str:
    if author and model_id.startswith(author + "/"):
        return model_id.split("/", 1)[1]
    return model_id


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def render(console, data: dict) -> None:
    c = console
    author = data.get("author")
    results = data["results"]
    where = f" from {author}" if author else ""

    if not results:
        c.emit(c.guidance(
            f"No models found for '{data['query']}'{where}.",
            ["The Hub search came back empty — try a broader or differently-spelled query."],
            [
                ("wmx-suite search <broader-term>", "loosen the query"),
                ("open https://huggingface.co/models", "browse the Hub in a browser"),
            ],
        ))
        return

    c.emit(c.section(f"Models matching '{data['query']}'{where} "
                     f"(top {len(results)} by downloads)"))
    c.emit()

    names = [_short(r["id"], author) for r in results]
    columns = [
        ("model", "l", "value"),
        ("downloads", "r", "metric"),
        ("likes", "r", "value"),
        ("quant", "l", "value"),
    ]
    rows = []
    for r, short in zip(results, names):
        rows.append((
            short,
            _fmt_count(r["downloads"]),
            _fmt_count(r["likes"]),
            r["quant"],
        ))
    c.emit(c.table(columns, rows))

    if c.verbose:
        raw = [f"{r['id']}  (mlx={r['mlx']})" for r in results]
        appendix = c.raw("raw (--verbose)", raw)
        if appendix:
            c.emit()
            c.emit(appendix)

    top = results[0]["id"]
    c.emit(c.next_block([
        (f"hf download {top}", "download a model into your HF cache"),
        ("wmx-suite scan", "register it with the suite"),
        ("wmx-suite characterize <model>", "measure its safe context ceiling"),
    ]))

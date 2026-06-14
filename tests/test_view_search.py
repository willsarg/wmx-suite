"""Golden tests for the search view (plain mode + verbose + color)."""
import io

from wmx_suite.ui import Console
from wmx_suite.views import search as view_search


def _data():
    return {
        "query": "gemma-4",
        "author": "mlx-community",
        "limit": 5,
        "count": 2,
        "results": [
            {"id": "mlx-community/gemma-4-e4b-it-4bit", "downloads": 43648,
             "likes": 18, "quant": "4-bit", "mlx": True},
            {"id": "mlx-community/gemma-4-12B-it-8bit", "downloads": 55348,
             "likes": 33, "quant": "8-bit", "mlx": True},
        ],
    }


def _render(data, *, verbose=False, color=False):
    buf = io.StringIO()
    view_search.render(Console(color=color, verbose=verbose, stream=buf), data)
    return buf.getvalue()


def test_search_table_and_next():
    out = _render(_data())
    assert "Models matching 'gemma-4' from mlx-community" in out
    # short names (author stripped) in the table
    assert "gemma-4-e4b-it-4bit" in out
    assert "mlx-community/gemma-4-e4b-it-4bit" not in out.split("next")[0]
    assert "4-bit" in out and "8-bit" in out
    assert "43.6k" in out          # downloads humanized
    # next block points at the top result with its FULL id for hf download
    assert "hf download mlx-community/gemma-4-e4b-it-4bit" in out
    assert "\033" not in out       # plain mode: zero ANSI


def test_search_empty_guidance():
    data = {"query": "zzz", "author": "mlx-community", "limit": 5,
            "count": 0, "results": []}
    out = _render(data)
    assert "No models found for 'zzz'" in out
    assert "\033" not in out


def test_search_verbose_appendix():
    out = _render(_data(), verbose=True)
    assert "raw" in out
    assert "mlx-community/gemma-4-12B-it-8bit" in out   # full ids in appendix


def test_search_color_has_ansi():
    out = _render(_data(), color=True)
    assert "\033" in out

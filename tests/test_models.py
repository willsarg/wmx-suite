import json
import os
from datetime import datetime, timezone

import pytest

from wmx_suite import models


def _describe(monkeypatch, config):
    monkeypatch.setattr(models, "_read_raw_config", lambda _hf_id: config)
    monkeypatch.setattr(models, "weights_gb", lambda _hf_id: 4.25)
    return models.describe("mlx-community/test")


def test_positive_sliding_window_is_rotating(monkeypatch):
    info = _describe(monkeypatch, {"num_hidden_layers": 4, "sliding_window": 512})
    assert info.cache_type == "RotatingKVCache"
    assert info.can_quantize_kv is False


def test_sliding_attention_layer_is_rotating(monkeypatch):
    info = _describe(
        monkeypatch,
        {
            "num_hidden_layers": 3,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "linear_attention",
            ],
        },
    )
    assert info.cache_type == "RotatingKVCache"
    assert info.growing_layers == 1


def test_standard_config_is_quantizable(monkeypatch):
    info = _describe(monkeypatch, {"num_hidden_layers": 4})
    assert info.cache_type == "standard"
    assert info.can_quantize_kv is True
    assert info.growing_layers == 4


def test_explicitly_disabled_sliding_window_is_standard(monkeypatch):
    # Qwen2.5-VL carries this metadata but does not enable a sliding cache.
    info = _describe(
        monkeypatch,
        {
            "num_hidden_layers": 4,
            "sliding_window": 32768,
            "use_sliding_window": False,
        },
    )
    assert info.cache_type == "standard"
    assert info.can_quantize_kv is True


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_5", "num_hidden_layers": 4},
        {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 4},
        },
    ],
)
def test_qwen35_custom_cache_does_not_enforce_max_kv_size(monkeypatch, config):
    assert _describe(monkeypatch, config).max_kv_size_enforced is False


def test_read_config_selects_nested_text_config(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    snapshot = hub / "models--mlx-community--test" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "HUB", str(hub))
    assert models._read_config("mlx-community/test") == {"num_hidden_layers": 7}


def test_read_config_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    assert models._read_config("mlx-community/missing") is None


def test_fp16_kv_bytes_per_token_uses_only_growing_layers():
    info = models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=4.0,
        n_layers=4,
        growing_layers=2,
        kv_heads=8,
        head_dim=128,
        hidden_size=1024,
        max_context=32768,
        cache_type="standard",
        can_quantize_kv=True,
        layer_types={"full_attention": 2, "linear_attention": 2},
    )
    assert info.fp16_kv_bytes_per_token() == 2 * 8 * 128 * 2 * 2


@pytest.mark.parametrize(("kv_heads", "head_dim"), [(None, 128), (8, None)])
def test_fp16_kv_bytes_per_token_requires_metadata(kv_heads, head_dim):
    info = models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=4.0,
        n_layers=4,
        growing_layers=2,
        kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=1024,
        max_context=32768,
        cache_type="standard",
        can_quantize_kv=True,
        layer_types={},
    )
    assert info.fp16_kv_bytes_per_token() == 0.0


def test_cache_updated_at_uses_newest_nested_artifact(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    cache = hub / "models--mlx-community--test"
    older = cache / "blobs" / "old"
    newer = cache / "snapshots" / "abc" / "config.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    monkeypatch.setattr(models, "HUB", str(hub))

    assert models.cache_updated_at("mlx-community/test") == 2000


def test_cache_updated_at_returns_none_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    assert models.cache_updated_at("mlx-community/missing") is None


def test_cache_updated_at_ignores_refs_and_negative_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    cache = hub / "models--mlx-community--test"
    snapshot = cache / "snapshots" / "commit" / "weight"
    unused_blob = cache / "blobs" / "unused"
    ref = cache / "refs" / "main"
    missing = cache / ".no_exist" / "commit" / "tokenizer.model"
    for path in (snapshot, unused_blob, ref, missing):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    os.utime(snapshot, (1000, 1000))
    os.utime(unused_blob, (5000, 5000))
    os.utime(ref, (3000, 3000))
    os.utime(missing, (4000, 4000))
    monkeypatch.setattr(models, "HUB", str(hub))

    assert models.cache_updated_at("mlx-community/test") == 1000


def test_cache_updated_at_counts_new_snapshot_symlink(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    cache = hub / "models--mlx-community--test"
    blob = cache / "blobs" / "weight"
    snapshot = cache / "snapshots" / "new" / "weight"
    blob.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    blob.write_text("weight", encoding="utf-8")
    os.utime(blob, (1000, 1000))
    snapshot.symlink_to(blob)
    os.utime(snapshot, (2000, 2000), follow_symlinks=False)
    monkeypatch.setattr(models, "HUB", str(hub))

    assert models.cache_updated_at("mlx-community/test") == 2000


def test_fit_is_stale_when_cache_is_more_than_one_second_newer(monkeypatch):
    characterized = datetime.fromtimestamp(1000, timezone.utc).isoformat()
    monkeypatch.setattr(models, "cache_updated_at", lambda _hf_id: 1001.1)
    assert models.fit_is_stale("mlx-community/test", characterized) is True


def test_fit_is_not_stale_within_timestamp_precision_tolerance(monkeypatch):
    characterized = datetime.fromtimestamp(1000, timezone.utc).isoformat()
    monkeypatch.setattr(models, "cache_updated_at", lambda _hf_id: 1000.9)
    assert models.fit_is_stale("mlx-community/test", characterized) is False

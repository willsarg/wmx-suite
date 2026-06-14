import pytest
import sys
from unittest.mock import MagicMock, patch

from wmx_suite import db, models
from wmx_suite.cli import cmd_scan, cmd_show, _run
from wmx_suite.models import ModelInfo


def test_get_classes_import_resolves():
    """Regression: _get_classes must be importable from mlx_lm.utils.

    If a future mlx_lm release renames or moves _get_classes, this test fails
    loudly in CI rather than silently marking every model as non-causal.
    """
    from mlx_lm.utils import _get_classes  # noqa: F401
    assert callable(_get_classes), "_get_classes must be callable"


def test_is_causal_classification():
    # 1. Causal LLM config (should be causal)
    causal_config = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "architectures": ["Qwen2ForCausalLM"],
    }
    
    with patch("mlx_lm.utils._get_classes") as mock_get:
        mock_get.return_value = (MagicMock(), MagicMock())
        with patch("wmx_suite.models._read_raw_config", return_value=causal_config):
            with patch("wmx_suite.models.weights_gb", return_value=1.0):
                info = models.describe("test/causal")
                assert info is not None
                assert info.is_causal is True

    # 2. Non-causal config (should not be causal)
    with patch("mlx_lm.utils._get_classes", side_effect=ValueError("Unsupported model type")):
        with patch("wmx_suite.models._read_raw_config", return_value={"model_type": "modernbert"}):
            with patch("wmx_suite.models.weights_gb", return_value=1.0):
                info = models.describe("test/non-causal")
                assert info is not None
                assert info.is_causal is False


def test_scan_filters_non_causal(monkeypatch):
    # Mock scan_cache to return two models
    monkeypatch.setattr(models, "scan_cache", lambda: ["test/causal", "test/non-causal"])
    
    # Mock describe to return one causal and one non-causal
    causal_info = MagicMock(is_causal=True, can_quantize_kv=True, weights_gb=2.0)
    causal_info.as_dict.return_value = {}
    non_causal_info = MagicMock(is_causal=False)
    
    def mock_describe(hf_id):
        if hf_id == "test/causal":
            return causal_info
        return non_causal_info
        
    monkeypatch.setattr(models, "describe", mock_describe)
    
    # Mock database upsert
    upserted = []
    monkeypatch.setattr(db, "connect", lambda: MagicMock())
    monkeypatch.setattr(db, "upsert_model", lambda con, info: upserted.append(info))

    # Capture rendered output via a no-color Console writing to a buffer.
    import io
    from types import SimpleNamespace
    from wmx_suite.ui import Console
    buf = io.StringIO()
    cmd_scan(SimpleNamespace(console=Console(color=False, verbose=False, stream=buf)))

    # Should only upsert the causal model
    assert len(upserted) == 1
    # Check that registered count is 1
    assert "registered 1 models" in buf.getvalue()


def test_run_prompts_for_characterization(monkeypatch):
    # Mock launcher.plan to return estimated first
    plans = [
        {"source": "estimated", "kv_bits": 4, "cache_type": "standard", "max_kv_size": 2048,
         "live_base_gb": 3.0, "model_base_gb": 4.0, "base_abs_gb": 7.0, "slope_gb_per_k": 0.1,
         "wall_gb": 17.0, "threshold_gb": 15.0, "model_max": 32768, "max_kv_size_enforced": True},
        {"source": "measured", "kv_bits": 4, "cache_type": "standard", "max_kv_size": 4096,
         "live_base_gb": 3.0, "model_base_gb": 4.0, "base_abs_gb": 7.0, "slope_gb_per_k": 0.1,
         "wall_gb": 17.0, "threshold_gb": 15.0, "model_max": 32768, "max_kv_size_enforced": True}
    ]
    
    plan_idx = [0]
    def mock_plan(model_id, margin_gb=None):
        p = plans[plan_idx[0]]
        if plan_idx[0] < len(plans) - 1:
            plan_idx[0] += 1
        return p
        
    monkeypatch.setattr("wmx_suite.launcher.plan", mock_plan)
    
    # Mock input to return yes
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    
    # Mock characterize to succeed
    char_called = [False]
    def mock_char(model_id, margin_gb=None, allow_min_probe=True):
        char_called[0] = True
        return {"refused": False}
    monkeypatch.setattr("wmx_suite.probe.characterize", mock_char)
    
    # Mock tokenizer load to prevent network lookup
    mock_tok = MagicMock()
    mock_tok.has_chat_template = False
    mock_tok.encode.return_value = [1, 2, 3]
    monkeypatch.setattr("wmx_suite.cli.load_tokenizer", lambda *args, **kwargs: mock_tok)
    
    # Mock execvp and logging to prevent actual process launch
    monkeypatch.setattr("wmx_suite.launcher.check_prompt", lambda *args: MagicMock(tokens=10, cap=4096, warn=False))
    monkeypatch.setattr("wmx_suite.launcher.effective_max_kv_size", lambda *args: 4096)
    monkeypatch.setattr("wmx_suite.launcher.build_argv", lambda *args, **kwargs: ["--model", "test/causal"])
    
    exec_called = [False]
    def mock_exec_logged(argv, model_id, cap):
        exec_called[0] = True
        
    monkeypatch.setattr("wmx_suite.cli._exec_logged", mock_exec_logged)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    
    _run(["--model", "test/causal"], margin=2.0, force=False, dry_run=False, log=True)
    
    assert char_called[0] is True
    assert exec_called[0] is True


def test_characterize_and_run_fail_fast_on_non_causal(monkeypatch):
    # Mock describe to return a non-causal model info
    non_causal_info = MagicMock(is_causal=False)
    monkeypatch.setattr(models, "describe", lambda hf_id: non_causal_info)
    
    # characterize should raise SystemExit with clean msg
    from wmx_suite.probe import characterize
    with pytest.raises(SystemExit) as excinfo:
        characterize("test/non-causal")
    assert "not a supported causal language model" in str(excinfo.value)
    
    # launcher.plan should return an error dictionary
    from wmx_suite.launcher import plan
    p = plan("test/non-causal")
    assert "error" in p
    assert "not a supported causal language model" in p["error"]

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path
from wmx_suite import db, launcher


class TestKokoroRemainingDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = Path(self.db_path)
        self.conn = db.connect()

    def tearDown(self):
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
        db.DB_PATH = self.original_db_path

    def test_voice_database_helpers(self):
        run_id = db.start_kokoro_voice_run(self.conn, "test-model", "0.20.0")
        self.assertEqual(run_id, 1)

        db.add_kokoro_voice_measurement(self.conn, run_id, "warm_switch", "af_heart", "am_adam", 5.2)
        db.add_kokoro_voice_measurement(self.conn, run_id, "cold_load", "none", "af_bella", 120.4)

        runs = db.get_all_kokoro_voice_runs(self.conn)
        self.assertEqual(len(runs), 1)

        measurements = db.get_kokoro_voice_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["cond_type"], "warm_switch")
        self.assertEqual(measurements[0]["duration_ms"], 5.2)

        latest = db.get_latest_kokoro_voice_run(self.conn)
        self.assertEqual(latest["id"], run_id)

    def test_cache_database_helpers(self):
        run_id = db.start_kokoro_cache_run(self.conn, "test-model", "0.20.0")
        self.assertEqual(run_id, 1)

        db.add_kokoro_cache_measurement(self.conn, run_id, 0, 1.2, 0.0)
        db.add_kokoro_cache_measurement(self.conn, run_id, 2, 1.25, 0.01)

        runs = db.get_all_kokoro_cache_runs(self.conn)
        self.assertEqual(len(runs), 1)

        measurements = db.get_kokoro_cache_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[1]["cache_size"], 2)
        self.assertEqual(measurements[1]["os_wired_gb"], 1.25)

        latest = db.get_latest_kokoro_cache_run(self.conn)
        self.assertEqual(latest["id"], run_id)

    def test_baseline_database_helpers(self):
        run_id = db.start_kokoro_baseline_run(self.conn, "test-model", "0.20.0")
        self.assertEqual(run_id, 1)

        db.add_kokoro_baseline_measurement(self.conn, run_id, 1.1, 1.95, 0.85)

        runs = db.get_all_kokoro_baseline_runs(self.conn)
        self.assertEqual(len(runs), 1)

        measurements = db.get_kokoro_baseline_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]["baseline_gb"], 1.1)
        self.assertEqual(measurements[0]["overhead_gb"], 0.85)

        latest_run = db.get_latest_kokoro_baseline_run(self.conn)
        self.assertEqual(latest_run["id"], run_id)

        latest_base = db.get_latest_kokoro_baseline(self.conn)
        self.assertIsNotNone(latest_base)
        self.assertEqual(latest_base["overhead_gb"], 0.85)


class TestKokoroRemainingWorkerLogic(unittest.TestCase):
    @patch("kokoro_mlx.KokoroTTS")
    @patch("kokoro_mlx.generate.generate")
    @patch("mlx.core.clear_cache")
    @patch("wmx_suite.kokoro_safety.over_threshold", return_value=False)
    @patch("wmx_suite.kokoro_safety.preflight", return_value=(100.0, 2.0, True))
    def test_voice_worker(self, mock_preflight, mock_over, mock_clear, mock_generate, mock_tts_cls):
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        mock_tts.list_voices.return_value = ["af_heart", "am_adam"]

        from wmx_suite import probe_worker_kokoro_voice
        with patch("sys.argv", ["probe_worker_kokoro_voice.py", "--repeats", "1"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_voice.main()
                self.assertTrue(mock_tts.close.called)
                # 2 warmup generates + 5 per-repeat generates = 7 calls
                self.assertEqual(mock_generate.call_count, 7)

    @patch("kokoro_mlx.KokoroTTS")
    @patch("kokoro_mlx.generate.generate")
    @patch("mlx.core.clear_cache")
    @patch("mlx.core.get_peak_memory")
    @patch("wmx_suite.system.wired_gb", return_value=2.0)
    @patch("wmx_suite.kokoro_safety.over_threshold", return_value=False)
    @patch("wmx_suite.kokoro_safety.preflight", return_value=(100.0, 2.0, True))
    def test_cache_worker(self, mock_preflight, mock_over, mock_wired, mock_peak, mock_clear, mock_generate, mock_tts_cls):
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        mock_tts.list_voices.return_value = ["af_heart", "am_adam"]
        mock_peak.return_value = 0.05 * 1e9

        from wmx_suite import probe_worker_kokoro_cache
        with patch("sys.argv", ["probe_worker_kokoro_cache.py", "--cache-sizes", "0,1"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_cache.main()
                self.assertTrue(mock_tts.close.called)

    @patch("kokoro_mlx.KokoroTTS")
    @patch("kokoro_mlx.generate.generate")
    @patch("mlx.core.clear_cache")
    @patch("wmx_suite.system.sample_settled_baseline")
    def test_baseline_worker(self, mock_baseline, mock_clear, mock_generate, mock_tts_cls):
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        # baseline starts at 1.0 GB, active settled baseline is 1.85 GB
        mock_baseline.side_effect = [1.0, 1.85]

        from wmx_suite import probe_worker_kokoro_baseline
        with patch("sys.argv", ["probe_worker_kokoro_baseline.py"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_baseline.main()
                self.assertTrue(mock_tts.close.called)


class TestLauncherSafetyIntegration(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = Path(self.db_path)
        self.conn = db.connect()

    def tearDown(self):
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
        db.DB_PATH = self.original_db_path

    @patch("wmx_suite.cli.launcher.plan")
    def test_co_run_kokoro_threshold_deduction(self, mock_plan):
        mock_plan.return_value = {
            "source": "measured",
            "max_kv_size": 2048,
            "kv_bits": 4,
            "cache_type": "standard",
            "live_base_gb": 3.0,
            "model_base_gb": 4.0,
            "base_abs_gb": 7.0,
            "slope_gb_per_k": 0.1,
            "wall_gb": 17.0,
            "threshold_gb": 15.0,
            "refuse": True,
            "reason": "Test refusal to avoid launch",
        }
        
        from wmx_suite import cli
        
        # 1. Without co-run-kokoro, it passes margin as is (e.g. 2.0)
        with self.assertRaises(SystemExit) as cm:
            cli._run(["--model", "test-causal"], margin=2.0, force=False, dry_run=True, co_run_kokoro=False)
        self.assertEqual(cm.exception.code, 2)
        mock_plan.assert_called_with("mlx-community/test-causal", margin_gb=2.0)

        # 2. With co-run-kokoro but no DB record, it adds 0.85 GB fallback overhead
        with self.assertRaises(SystemExit) as cm:
            cli._run(["--model", "test-causal"], margin=2.0, force=False, dry_run=True, co_run_kokoro=True)
        self.assertEqual(cm.exception.code, 2)
        mock_plan.assert_called_with("mlx-community/test-causal", margin_gb=2.85)

        # 3. Record a database baseline measurement of 0.65 GB overhead
        run_id = db.start_kokoro_baseline_run(self.conn, "mlx-community/Kokoro-82M-bf16", "0.20.0")
        db.add_kokoro_baseline_measurement(self.conn, run_id, 1.0, 1.65, 0.65)

        # 4. With co-run-kokoro and DB record, it adds 0.65 GB overhead
        with self.assertRaises(SystemExit) as cm:
            cli._run(["--model", "test-causal"], margin=2.0, force=False, dry_run=True, co_run_kokoro=True)
        self.assertEqual(cm.exception.code, 2)
        mock_plan.assert_called_with("mlx-community/test-causal", margin_gb=2.65)


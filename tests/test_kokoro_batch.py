import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path
from wmx_suite import db


class TestKokoroBatchDatabase(unittest.TestCase):
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

    def test_kokoro_batch_run_lifecycle(self):
        # 1. Start a run
        run_id = db.start_kokoro_batch_run(self.conn, "test-model", "af_heart", "0.20.0")
        self.assertEqual(run_id, 1)

        # 2. Add measurements
        db.add_kokoro_batch_measurement(self.conn, run_id, 1, 0.5, 200.0, 1.2)
        db.add_kokoro_batch_measurement(self.conn, run_id, 2, 0.45, 440.0, 1.5)

        # 3. Retrieve all runs
        runs = db.get_all_kokoro_batch_runs(self.conn)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["model_id"], "test-model")
        self.assertEqual(runs[0]["voice"], "af_heart")

        # 4. Retrieve measurements for run
        measurements = db.get_kokoro_batch_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["batch_size"], 1)
        self.assertEqual(measurements[0]["total_time"], 0.5)
        self.assertEqual(measurements[0]["cps"], 200.0)
        self.assertEqual(measurements[0]["peak_gb"], 1.2)

        # 5. Get latest run
        latest = db.get_latest_kokoro_batch_run(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], run_id)


class TestKokoroBatchWorkerLogic(unittest.TestCase):
    @patch("kokoro_mlx.KokoroTTS")
    @patch("kokoro_mlx.generate.generate")
    @patch("mlx.core.clear_cache")
    @patch("mlx.core.reset_peak_memory")
    @patch("mlx.core.get_peak_memory")
    @patch("wmx_suite.system.wired_gb", return_value=2.0)
    @patch("wmx_suite.kokoro_safety.preflight", return_value=(100.0, 2.0, True))
    def test_worker_generation(self, mock_preflight, mock_wired, mock_get_peak, mock_reset_peak, mock_clear_cache, mock_generate, mock_tts_cls):
        # Setup mocks
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        
        # Mock generate return value
        mock_generate.return_value = MagicMock()
        mock_get_peak.return_value = 1.5 * 1e9  # 1.5 GB in bytes

        # Check worker logic imports and basic execute
        from wmx_suite import probe_worker_kokoro_batch
        
        # Test main method arguments execution (using sys.argv patching)
        # Using batch-sizes "1,2" and repeats "2"
        with patch("sys.argv", ["probe_worker_kokoro_batch.py", "--model", "test-model", "--batch-sizes", "1,2", "--repeats", "2"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_batch.main()
                
                # Check that from_pretrained was called with the model name
                mock_tts_cls.from_pretrained.assert_called_with("test-model")
                
                # Check that generate was called:
                # - 1 for warmup
                # - 2 iterations for batch_size=1 (1 generate call per iteration = 2 calls)
                # - 2 iterations for batch_size=2 (2 generate calls per iteration = 4 calls)
                # Total expected calls = 1 + 2 + 4 = 7 calls
                self.assertEqual(mock_generate.call_count, 7)
                mock_tts.close.assert_called_once()

    @patch("kokoro_mlx.KokoroTTS")
    @patch("kokoro_mlx.generate.generate")
    @patch("mlx.core.clear_cache")
    @patch("mlx.core.reset_peak_memory")
    @patch("mlx.core.get_peak_memory")
    @patch("wmx_suite.system.wired_gb", return_value=7.0)
    @patch("wmx_suite.kokoro_safety.preflight", return_value=(8.0, 2.0, True))
    def test_worker_predictively_skips_unsafe_concurrent_rung(
        self, mock_preflight, mock_wired, mock_get_peak, mock_reset_peak,
        mock_clear_cache, mock_generate, mock_tts_cls):
        """RULE #1: a batch rung whose PREDICTED concurrent peak would breach the wall must
        be skipped before it runs, even though current wired memory is below threshold."""
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        mock_generate.return_value = MagicMock()
        mock_get_peak.return_value = 1.0 * 1e9

        # threshold 8.0, baseline 2.0, current wired 7.0 (under threshold).
        # batch 1 runs: hi=7.0 -> per_call = (7.0-2.0)/1 = 5.0 GB/call.
        # batch 2 predicted = 7.0 + 5.0*2 = 17.0 >= 8.0 -> SKIP before running.
        from wmx_suite import probe_worker_kokoro_batch
        with patch("sys.argv", ["probe_worker_kokoro_batch.py", "--model", "test-model",
                                "--batch-sizes", "1,2", "--repeats", "1"]):
            with patch("sys.exit"):
                probe_worker_kokoro_batch.main()
                # warmup (1) + batch 1 (1) = 2; batch 2 never runs.
                self.assertEqual(mock_generate.call_count, 2)
                mock_tts.close.assert_called_once()

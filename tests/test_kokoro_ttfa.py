import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from wmx_suite import db


class TestKokoroTtfadb(unittest.TestCase):
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

    def test_kokoro_ttfa_run_lifecycle(self):
        # 1. Start a run
        run_id = db.start_kokoro_ttfa_run(self.conn, "test-model-ttfa", "af_heart", "0.20.0")
        self.assertEqual(run_id, 1)

        # 2. Add measurements
        db.add_kokoro_ttfa_measurement(self.conn, run_id, 10, 0.15, 0.15, 1.0, 1.2, 0.9)
        db.add_kokoro_ttfa_measurement(self.conn, run_id, 1000, 1.1, 8.5, 7.7, 30.0, 3.2)

        # 3. Retrieve all runs
        runs = db.get_all_kokoro_ttfa_runs(self.conn)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["model_id"], "test-model-ttfa")

        # 4. Retrieve measurements for run
        measurements = db.get_kokoro_ttfa_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["text_length"], 10)
        self.assertEqual(measurements[0]["ttfa_sec"], 0.15)
        self.assertEqual(measurements[0]["total_sec"], 0.15)
        self.assertEqual(measurements[0]["speedup_ratio"], 1.0)
        self.assertEqual(measurements[0]["first_chunk_duration"], 1.2)
        self.assertEqual(measurements[0]["peak_gb"], 0.9)

        # 5. Get latest run
        latest = db.get_latest_kokoro_ttfa_run(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], run_id)


class TestKokoroTtfaWorkerLogic(unittest.TestCase):
    @patch("kokoro_mlx.KokoroTTS")
    @patch("mlx.core.clear_cache")
    @patch("mlx.core.reset_peak_memory")
    @patch("mlx.core.get_peak_memory")
    @patch("wmx_suite.system.wired_gb")
    @patch("wmx_suite.system.read_limits")
    def test_worker_generation_normal(self, mock_read_limits, mock_wired_gb, mock_get_peak, mock_reset_peak, mock_clear_cache, mock_tts_cls):
        # Setup mock limits: total 16GB, wall 10GB, wired 2GB. Margin 2.0GB means threshold is 8.0GB
        mock_limits = MagicMock()
        mock_limits.wired_now_gb = 2.0
        mock_limits.safe_threshold_gb.return_value = 8.0
        mock_read_limits.return_value = mock_limits
        mock_wired_gb.return_value = 3.0  # well below 8GB

        # Setup model mock
        mock_tts = MagicMock()
        mock_tts.SAMPLE_RATE = 24000
        mock_tts_cls.from_pretrained.return_value = mock_tts
        
        # Mock generate
        mock_res = MagicMock()
        mock_tts.generate.return_value = mock_res
        
        # Mock generate_stream
        mock_first_chunk = [0.0] * 24000  # 1 second of audio
        def dummy_stream(text, voice):
            yield mock_first_chunk
            yield [0.0] * 12000
        mock_tts.generate_stream.side_effect = dummy_stream
        
        mock_get_peak.return_value = 1.2 * 1e9

        from wmx_suite import probe_worker_kokoro_ttfa
        
        with patch("sys.argv", ["probe_worker_kokoro_ttfa.py", "--model", "test-model", "--lengths", "10,50", "--repeats", "2"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_ttfa.main()
                
                # generate is called for warmup (1) + non-streaming sweeps (2 lengths * 2 repeats = 4). Total = 5 calls
                self.assertEqual(mock_tts.generate.call_count, 5)
                # generate_stream is called for warmup (1) + streaming sweeps (2 lengths * 2 repeats = 4). Total = 5 calls
                self.assertEqual(mock_tts.generate_stream.call_count, 5)
                mock_tts.close.assert_called_once()

    @patch("kokoro_mlx.KokoroTTS")
    @patch("wmx_suite.system.wired_gb")
    @patch("wmx_suite.system.read_limits")
    def test_worker_safeguard_triggered(self, mock_read_limits, mock_wired_gb, mock_tts_cls):
        # Setup mock limits: total 16GB, wall 10GB, wired 2GB. Margin 2.0GB => threshold is 8.0GB
        mock_limits = MagicMock()
        mock_limits.wired_now_gb = 2.0
        mock_limits.safe_threshold_gb.return_value = 8.0
        mock_read_limits.return_value = mock_limits

        # First rung checks: wired memory starts at 3.0GB (runs normally).
        # Second rung checks: wired memory spikes to 8.5GB (safeguard triggers).
        mock_wired_gb.side_effect = [3.0, 8.5]  # precheck for rung 1, precheck for rung 2

        # Setup model mock
        mock_tts = MagicMock()
        mock_tts.SAMPLE_RATE = 24000
        mock_tts_cls.from_pretrained.return_value = mock_tts
        mock_first_chunk = [0.0] * 24000
        def dummy_stream(text, voice):
            yield mock_first_chunk
        mock_tts.generate_stream.side_effect = dummy_stream

        from wmx_suite import probe_worker_kokoro_ttfa

        # Sweep 2 lengths: 10, 50. First length 10 runs trial. Second length 50 triggers safeguard.
        with patch("sys.argv", ["probe_worker_kokoro_ttfa.py", "--model", "test-model", "--lengths", "10,50", "--repeats", "1"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro_ttfa.main()
                # generate called for: warmup (1) + length 10 non-streaming (1) = 2. Second length 50 bypassed by safeguard.
                self.assertEqual(mock_tts.generate.call_count, 2)
                mock_tts.close.assert_called_once()

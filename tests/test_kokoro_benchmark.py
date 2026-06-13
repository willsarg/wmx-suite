import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path
from wmx_suite import db
from wmx_suite.probe_worker_kokoro import get_text_of_length


class TestKokoroDatabase(unittest.TestCase):
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

    def test_kokoro_run_lifecycle(self):
        # 1. Start a run
        run_id = db.start_kokoro_run(self.conn, "test-model", "af_heart", "0.20.0")
        self.assertEqual(run_id, 1)

        # 2. Add measurements
        db.add_kokoro_measurement(self.conn, run_id, 10, 1.5, 0.05, 0.033, 200.0, 1.2)
        db.add_kokoro_measurement(self.conn, run_id, 50, 3.2, 0.12, 0.037, 416.6, 1.8)

        # 3. Retrieve all runs
        runs = db.get_all_kokoro_runs(self.conn)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["model_id"], "test-model")
        self.assertEqual(runs[0]["voice"], "af_heart")

        # 4. Retrieve measurements for run
        measurements = db.get_kokoro_measurements(self.conn, run_id)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["text_length"], 10)
        self.assertEqual(measurements[0]["audio_duration"], 1.5)
        self.assertEqual(measurements[0]["compute_time"], 0.05)
        self.assertEqual(measurements[0]["rtf"], 0.033)
        self.assertEqual(measurements[0]["cps"], 200.0)
        self.assertEqual(measurements[0]["peak_gb"], 1.2)

        # 5. Get latest run
        latest = db.get_latest_kokoro_run(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], run_id)


class TestKokoroTextGeneration(unittest.TestCase):
    def test_get_text_of_length(self):
        # Test 0 length
        self.assertEqual(get_text_of_length(0), "")
        self.assertEqual(get_text_of_length(-10), "")

        # Test short length
        text_10 = get_text_of_length(10)
        self.assertTrue(len(text_10) > 0)
        self.assertTrue(len(text_10) <= 25)  # accounting for space index padding

        # Test extremely long length
        text_5000 = get_text_of_length(5000)
        self.assertGreaterEqual(len(text_5000), 5000)


class TestKokoroWorkerLogic(unittest.TestCase):
    @patch("kokoro_mlx.KokoroTTS")
    @patch("mlx.core.clear_cache")
    @patch("mlx.core.reset_peak_memory")
    @patch("mlx.core.get_peak_memory")
    def test_worker_generation(self, mock_get_peak, mock_reset_peak, mock_clear_cache, mock_tts_cls):
        # Setup mocks
        mock_tts = MagicMock()
        mock_tts_cls.from_pretrained.return_value = mock_tts
        
        # Mock generate return value
        mock_res = MagicMock()
        mock_res.duration = 2.0
        mock_tts.generate.return_value = mock_res
        mock_get_peak.return_value = 1.5 * 1e9  # 1.5 GB in bytes

        # Check worker logic imports and basic execute
        from wmx_suite import probe_worker_kokoro
        
        # Test main method arguments execution (using sys.argv patching)
        with patch("sys.argv", ["probe_worker_kokoro.py", "--model", "test-model", "--lengths", "10,50", "--repeats", "2"]):
            with patch("sys.exit") as mock_exit:
                probe_worker_kokoro.main()
                
                # Check that from_pretrained was called with the model name
                mock_tts_cls.from_pretrained.assert_called_with("test-model")
                
                # Check that generate was called (at least once for warmup, and repeats * lengths times)
                # warmup is 1 run. 2 lengths * 2 repeats = 4 runs. Total = 5 calls to generate
                self.assertEqual(mock_tts.generate.call_count, 5)
                mock_tts.close.assert_called_once()

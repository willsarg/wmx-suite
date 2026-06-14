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

        # 2. Add measurements: positional 8-arg call (legacy) — os_wired_gb must default to None
        db.add_kokoro_measurement(self.conn, run_id, 10, 1.5, 0.05, 0.033, 200.0, 1.2)
        # Keyword call with os_wired_gb
        db.add_kokoro_measurement(self.conn, run_id, 50, 3.2, 0.12, 0.037, 416.6, 1.8,
                                  os_wired_gb=2.45)

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
        # Legacy positional call: os_wired_gb must be None
        self.assertIsNone(measurements[0]["os_wired_gb"])

        # Keyword call: os_wired_gb round-trips correctly
        self.assertEqual(measurements[1]["peak_gb"], 1.8)
        self.assertAlmostEqual(measurements[1]["os_wired_gb"], 2.45)

        # 5. Get latest run
        latest = db.get_latest_kokoro_run(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], run_id)

    def test_connect_migrates_old_db_missing_os_wired_column(self):
        """connect() must add os_wired_gb to a pre-existing OLD-schema table, idempotently."""
        # setUp() already ran connect() once (creating the new-schema table). Drop it and
        # recreate kokoro_measurements WITHOUT os_wired_gb to truly simulate an OLD database.
        raw = sqlite3.connect(self.db_path)
        raw.executescript(
            "DROP TABLE IF EXISTS kokoro_measurements;"
            "CREATE TABLE kokoro_measurements ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
            " text_length INTEGER NOT NULL, audio_duration REAL NOT NULL,"
            " compute_time REAL NOT NULL, rtf REAL NOT NULL, cps REAL NOT NULL,"
            " peak_gb REAL);"
        )
        raw.commit()
        raw.close()

        # Sanity: the column really is missing before migration.
        pre = sqlite3.connect(self.db_path)
        pre_cols = {row[1] for row in pre.execute("PRAGMA table_info(kokoro_measurements)")}
        pre.close()
        self.assertNotIn("os_wired_gb", pre_cols)

        # connect() must add the missing column via migration.
        con = db.connect()
        cols = {row[1] for row in con.execute("PRAGMA table_info(kokoro_measurements)")}
        self.assertIn("os_wired_gb", cols)

        # Inserts (including os_wired_gb) must work after migration.
        run_id = db.start_kokoro_run(con, "test-model", "af_heart", "0.20.0")
        db.add_kokoro_measurement(con, run_id, 10, 1.5, 0.05, 0.033, 200.0, 1.2,
                                  os_wired_gb=2.45)
        rows = db.get_kokoro_measurements(con, run_id)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["os_wired_gb"], 2.45)

        # A second connect must be a no-op (idempotent), no error.
        con2 = db.connect()
        self.assertIsNotNone(con2)
        cols2 = {row[1] for row in con2.execute("PRAGMA table_info(kokoro_measurements)")}
        self.assertIn("os_wired_gb", cols2)

        con.close()
        con2.close()


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
    @patch("wmx_suite.system.wired_gb", return_value=2.0)
    @patch("wmx_suite.kokoro_safety.over_threshold", return_value=False)
    @patch("wmx_suite.kokoro_safety.preflight", return_value=(100.0, 2.0, True))
    def test_worker_generation(self, mock_preflight, mock_over, mock_wired, mock_get_peak, mock_reset_peak, mock_clear_cache, mock_tts_cls):
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

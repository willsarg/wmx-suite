from pathlib import Path

from wmx_suite import db


def test_embeddings_db_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()

    run_id = db.start_embeddings_run(con, "mlx-community/test-modernbert", "0.31.2")
    assert isinstance(run_id, int)

    db.add_embeddings_measurement(
        con, run_id, batch_size=2, seq_len=512,
        os_wired_gb=4.5, peak_gb=3.1, throughput_tps=12345.0, latency_ms=82.9,
    )
    db.add_embeddings_measurement(
        con, run_id, batch_size=4, seq_len=128,
        os_wired_gb=4.8, peak_gb=3.3, throughput_tps=22222.0, latency_ms=44.1,
    )

    runs = db.get_all_embeddings_runs(con)
    assert len(runs) == 1
    assert runs[0]["model_id"] == "mlx-community/test-modernbert"
    assert runs[0]["mlx_version"] == "0.31.2"

    rows = db.get_embeddings_measurements(con, run_id)
    assert len(rows) == 2
    assert (rows[0]["batch_size"], rows[0]["seq_len"]) == (2, 512)
    assert (rows[1]["batch_size"], rows[1]["seq_len"]) == (4, 128)
    assert rows[0]["throughput_tps"] == 12345.0

    latest = db.get_latest_embeddings_run(con)
    assert latest["id"] == run_id

    con.execute("DELETE FROM embeddings_runs WHERE id = ?", (run_id,))
    con.commit()
    assert db.get_embeddings_measurements(con, run_id) == []

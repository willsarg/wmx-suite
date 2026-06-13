import pytest
import json
from wmx_suite import db
from wmx_suite.web.app import create_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    
    # insert test data
    db.upsert_model(
        con,
        {
            "hf_id": "mlx-community/test",
            "weights_gb": 4.0,
            "n_layers": 4,
            "growing_layers": 2,
            "kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 1024,
            "max_context": 32768,
            "cache_type": "standard",
            "can_quantize_kv": True,
            "layer_types": {},
        },
    )
    run_id = db.start_run(
        con,
        "mlx-community/test",
        kv_bits=4,
        kv_group_size=64,
        quantized_kv_start=5000,
        mlx_version="test",
        wall_gb=17.0,
        safe_threshold_gb=15.0,
    )
    db.add_measurement(
        con,
        run_id,
        2048,
        mlx_peak_gb=2.0,
        mlx_true_gb=2.2,
        os_wired_gb=5.0,
        status="ok",
    )
    db.save_fit(
        con,
        run_id,
        {
            "model_base_gb": 8.0,
            "slope_gb_per_k": 0.1,
            "r2": 1.0,
            "ref_baseline_gb": 3.0,
            "threshold_gb": 15.0,
            "safe_ceiling_ctx": 40000,
            "hard_wall_ctx": 60000,
            "n_points": 3,
        },
    )
    con.close()
    
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    rv = client.get("/")
    assert rv.status_code == 200
    html = rv.data.decode()
    assert "Dashboard" in html
    assert "mlx-community/test" in html
    assert "40,000" in html  # Safe ceiling formatting test


def test_model_detail_route(client):
    rv = client.get("/model/mlx-community/test")
    assert rv.status_code == 200
    html = rv.data.decode()
    assert "mlx-community/test" in html
    assert "Architecture Parameters" in html
    assert "Latest Fitting Statistics" in html
    assert "40,000" in html


def test_model_detail_not_found(client):
    rv = client.get("/model/mlx-community/unknown")
    assert rv.status_code == 404


def test_compare_route(client):
    rv = client.get("/compare")
    assert rv.status_code == 200
    html = rv.data.decode()
    assert "Compare" in html
    assert "mlx-community/test" in html


def test_api_system(client):
    rv = client.get("/api/system")
    assert rv.status_code == 200
    data = json.loads(rv.data.decode())
    assert "device" in data
    assert "total_gb" in data
    assert "wall_gb" in data
    assert "safe_threshold_gb" in data


def test_api_measurements(client):
    rv = client.get("/api/model/mlx-community/test/measurements/1")
    assert rv.status_code == 200
    data = json.loads(rv.data.decode())
    assert len(data) == 1
    assert data[0]["context"] == 2048
    assert data[0]["os_wired_gb"] == 5.0

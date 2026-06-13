from wmx_suite import db


def test_latest_fit_exposes_characterization_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
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

    fit = db.latest_fit(con, "mlx-community/test")

    assert fit["characterized_at"]


def test_latest_fits_returns_only_newest_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
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
    for base in (7.0, 8.0):
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
        db.save_fit(
            con,
            run_id,
            {
                "model_base_gb": base,
                "slope_gb_per_k": 0.1,
                "r2": 1.0,
                "ref_baseline_gb": 3.0,
                "threshold_gb": 15.0,
                "safe_ceiling_ctx": 40000,
                "hard_wall_ctx": 60000,
                "n_points": 3,
            },
        )

    rows = db.latest_fits(con)

    assert len(rows) == 1
    assert rows[0]["model_base_gb"] == 8.0
    assert rows[0]["characterized_at"]

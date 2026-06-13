from wmx_suite import db, profiles


def test_cold_start_constants_defaults_when_no_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    con = db.connect()
    factor, overhead, source = profiles.cold_start_constants(con)
    assert factor == profiles.DEFAULT_RESIDENT_FACTOR
    assert overhead == profiles.DEFAULT_FIXED_OVERHEAD_GB
    assert source == "default"


def test_cold_start_constants_uses_stored_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    con = db.connect()
    db.upsert_profile(con, key, resident_factor=1.05, fixed_overhead_gb=1.6,
                      model_id="m", n_points=2, mlx_version="9.9")
    factor, overhead, source = profiles.cold_start_constants(con)
    assert factor == 1.05
    assert overhead == 1.6
    assert source == "profile"


def test_machine_key_shape(monkeypatch):
    monkeypatch.setattr(profiles.system, "macos_major", lambda: 15)
    # mlx is installed; machine_key reads real device_info — just assert the shape/types.
    dev, ram, osv = profiles.machine_key()
    assert isinstance(dev, str)
    assert isinstance(ram, int) and ram >= 0
    assert osv == 15

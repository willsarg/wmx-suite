"""Command-line entry point for the suite.

    uv run mlx-suite system                 # show the machine's memory wall + swap
    uv run mlx-suite scan                    # register all mlx-community models in the cache
    uv run mlx-suite show <hf_id>            # architecture + memory class for one model
    uv run mlx-suite characterize <hf_id>   # safe probe -> fitted context ceiling
    uv run mlx-suite list                    # characterized models + ceilings from the DB
"""
from __future__ import annotations

import argparse

from . import db, models, probe
from .system import read_limits


def cmd_system(_):
    s = read_limits()
    print(f"device              : {s.device}")
    print(f"total RAM           : {s.total_gb:.2f} GB")
    print(f"crash wall          : {s.wall_gb:.2f} GB  (max_recommended_working_set_size)")
    print(f"max single buffer   : {s.max_buffer_gb:.2f} GB")
    print(f"swap free           : {s.swap_free_gb:.2f} GB" if s.swap_free_gb is not None
          else "swap free           : unknown")
    print(f"wired now (baseline): {s.wired_now_gb:.2f} GB")
    print(f"safe threshold (2GB): {s.safe_threshold_gb():.2f} GB")


def cmd_scan(_):
    con = db.connect()
    found = models.scan_cache()
    n = 0
    for hf_id in found:
        info = models.describe(hf_id)
        if info is None:
            continue
        db.upsert_model(con, info.as_dict())
        n += 1
        flag = "quantizable" if info.can_quantize_kv else "fp16-only (RotatingKVCache)"
        print(f"  {hf_id:60} {info.weights_gb:5.1f}GB  {flag}")
    print(f"registered {n} models")


def cmd_show(args):
    info = models.describe(args.hf_id)
    if info is None:
        raise SystemExit(f"not found in cache: {args.hf_id}")
    for k, v in info.as_dict().items():
        print(f"  {k:18}: {v}")
    bpt = info.fp16_kv_bytes_per_token()
    print(f"  fp16 KV bytes/token: {bpt:.0f} "
          f"(~{bpt * 1000 / 1e9:.4f} GB per 1k tokens)")


def cmd_characterize(args):
    probe.characterize(args.hf_id, margin_gb=args.margin)


def cmd_list(_):
    con = db.connect()
    rows = con.execute(
        "SELECT m.hf_id, m.weights_gb, m.cache_type, f.slope_gb_per_k, f.intercept_gb, "
        "f.safe_ceiling_ctx, f.hard_wall_ctx, f.r2 "
        "FROM fits f JOIN probe_runs r ON f.run_id=r.id JOIN models m ON r.hf_id=m.hf_id "
        "GROUP BY m.hf_id HAVING f.id=MAX(f.id) ORDER BY m.hf_id"
    ).fetchall()
    if not rows:
        print("no characterized models yet — run `characterize <hf_id>`")
        return
    for r in rows:
        print(f"  {r['hf_id']:48} safe≈{r['safe_ceiling_ctx']:>7,}  "
              f"wall≈{r['hard_wall_ctx']:>7,}  (R²={r['r2']})")


def main():
    ap = argparse.ArgumentParser(prog="mlx-suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("system").set_defaults(func=cmd_system)
    sub.add_parser("scan").set_defaults(func=cmd_scan)
    p = sub.add_parser("show"); p.add_argument("hf_id"); p.set_defaults(func=cmd_show)
    p = sub.add_parser("characterize"); p.add_argument("hf_id")
    p.add_argument("--margin", type=float, default=2.0); p.set_defaults(func=cmd_characterize)
    sub.add_parser("list").set_defaults(func=cmd_list)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

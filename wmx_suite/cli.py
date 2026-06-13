"""Command-line entry point for the suite.

    uv run wmx-suite system                 # show the machine's memory wall + swap
    uv run wmx-suite scan                    # register all mlx-community models in the cache
    uv run wmx-suite show <hf_id>            # architecture + memory class for one model
    uv run wmx-suite characterize <hf_id>   # safe probe -> fitted context ceiling
    uv run wmx-suite list                    # characterized models + ceilings from the DB
"""
from __future__ import annotations

import argparse
import os
import sys

from . import db, launcher, models, probe
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
    probe.characterize(args.hf_id, margin_gb=args.margin, allow_min_probe=args.min_probe,
                       repeats=args.repeats)


def cmd_list(_):
    con = db.connect()
    rows = con.execute(
        "SELECT m.hf_id, m.cache_type, f.model_base_gb, f.slope_gb_per_k, f.ref_baseline_gb, "
        "f.safe_ceiling_ctx, f.hard_wall_ctx, f.r2 "
        "FROM fits f JOIN probe_runs r ON f.run_id=r.id JOIN models m ON r.hf_id=m.hf_id "
        "GROUP BY m.hf_id HAVING f.id=MAX(f.id) ORDER BY m.hf_id"
    ).fetchall()
    if not rows:
        print("no characterized models yet — run `characterize <hf_id>`")
        return
    for r in rows:
        print(f"  {r['hf_id']:46} base={r['model_base_gb']:5.1f}GB "
              f"slope={r['slope_gb_per_k']:.4f}  safe≈{r['safe_ceiling_ctx']:>7,}  "
              f"wall≈{r['hard_wall_ctx']:>7,}  (R²={r['r2']})")


RUN_HELP = """usage: wmx-suite run [--margin GB] [--force] [--dry-run] -- <mlx_lm.generate args>

Safely launch mlx_lm.generate (replaces the old mlx_safe). Picks kv-bits by cache type,
caps --max-kv-size from the measured ceiling, and refuses if the run would breach the wall.
The passthrough args must include --model <hf_id>.

  --margin GB   safety cushion under the wall (default 2.0)
  --force       launch even if the planner refuses (may crash the machine)
  --dry-run     print the plan, do not launch
"""


def cmd_run_raw(run_args: list[str]):
    """Parse leading suite flags, then treat the remainder as mlx_lm.generate passthrough.

    Done manually (not argparse) because argparse.REMAINDER mishandles optionals that
    precede the positional, and we want `run --model X ...` to work as a drop-in.
    """
    margin, force, dry = 2.0, False, False
    i = 0
    while i < len(run_args):
        a = run_args[i]
        if a in ("-h", "--help"):
            print(RUN_HELP); return
        if a == "--margin":
            margin = float(run_args[i + 1]); i += 2; continue
        if a == "--force":
            force = True; i += 1; continue
        if a == "--dry-run":
            dry = True; i += 1; continue
        break  # first non-suite token: the rest is passthrough
    rest = run_args[i:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    _run(rest, margin=margin, force=force, dry_run=dry)


def _run(rest: list[str], *, margin: float, force: bool, dry_run: bool):
    """Safe replacement for the old mlx_safe: plan a launch, then exec mlx_lm.generate."""
    model_id = None
    for i, a in enumerate(rest):
        if a == "--model" and i + 1 < len(rest):
            model_id = rest[i + 1]
            break
    if model_id is None:
        raise SystemExit("[run] --model is required")

    p = launcher.plan(model_id, margin_gb=margin)
    if p.get("error"):
        raise SystemExit(f"[run] {p['error']}")

    kv = "fp16 (RotatingKVCache — not quantizable)" if p["kv_bits"] is None else f"{p['kv_bits']}-bit"
    print(f"[run] {model_id}", file=sys.stderr)
    print(f"[run] source={p['source']}  cache={p['cache_type']}  kv={kv}", file=sys.stderr)
    print(f"[run] live_base {p['live_base_gb']}GB + model {p['model_base_gb']}GB = "
          f"{p['base_abs_gb']}GB  |  slope {p['slope_gb_per_k']}GB/1k  |  "
          f"wall {p['wall_gb']}GB  threshold {p['threshold_gb']}GB", file=sys.stderr)

    if p.get("refuse"):
        print(f"[run] REFUSED: {p['reason']}", file=sys.stderr)
        if not force:
            print("[run] (pass --force to override at your own risk — may crash the machine)",
                  file=sys.stderr)
            sys.exit(2)
        print("[run] --force given; proceeding against safety advice.", file=sys.stderr)

    argv = launcher.build_argv(rest, p)
    print(f"[run] max-kv-size {p['max_kv_size']:,} tokens (model cap {p['model_max']:,})",
          file=sys.stderr)
    print(f"[run] exec: mlx_lm.generate {' '.join(argv)}\n", file=sys.stderr)
    if dry_run:
        print("[run] --dry-run: not launching.", file=sys.stderr)
        return
    os.execvp("mlx_lm.generate", ["mlx_lm.generate"] + argv)


def _main_argparse():
    ap = argparse.ArgumentParser(prog="wmx-suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("system").set_defaults(func=cmd_system)
    sub.add_parser("scan").set_defaults(func=cmd_scan)
    p = sub.add_parser("show"); p.add_argument("hf_id"); p.set_defaults(func=cmd_show)
    p = sub.add_parser("characterize"); p.add_argument("hf_id")
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--min-probe", action="store_true",
                   help="for borderline models, run a supervised 512-token probe to measure "
                        "the true base instead of refusing on the pessimistic estimate")
    p.add_argument("--repeats", type=int, default=probe.DEFAULT_REPEATS,
                   help="isolated runs per context rung; the median high-water is used "
                        "(smooths prefill-transient jitter)")
    p.set_defaults(func=cmd_characterize)
    sub.add_parser("list").set_defaults(func=cmd_list)
    # `run` is intercepted before argparse (see below) so it can pass arbitrary flags
    # through to mlx_lm.generate; this stub only makes it show up in `--help`.
    sub.add_parser("run", help="safely launch mlx_lm.generate (replaces mlx_safe): picks "
                               "kv-bits by cache type, caps --max-kv-size from the measured "
                               "ceiling, refuses if it would breach the wall")
    args = ap.parse_args()
    args.func(args)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        return cmd_run_raw(argv[1:])
    return _main_argparse()


if __name__ == "__main__":
    main()

"""Command-line entry point for the suite.

    uv run wmx-suite system                 # show the machine's memory wall + swap
    uv run wmx-suite health                  # live snapshot: pressure + per-model go/no-go
    uv run wmx-suite scan                    # register all mlx-community models in the cache
    uv run wmx-suite show <hf_id>            # architecture + memory class for one model
    uv run wmx-suite characterize <hf_id>   # safe probe -> fitted context ceiling
    uv run wmx-suite list                    # characterized models + ceilings from the DB
"""
from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import termios
from statistics import median

from mlx_lm.utils import load_tokenizer

from . import config, db, launcher, models, probe
from .system import read_limits, sample_settled_baseline


def _configured_margin(value=None) -> float:
    try:
        return config.margin_gb(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def cmd_system(_):
    s = read_limits()
    margin = _configured_margin()
    print(f"device              : {s.device}")
    print(f"total RAM           : {s.total_gb:.2f} GB")
    print(f"crash wall          : {s.wall_gb:.2f} GB  (max_recommended_working_set_size)")
    print(f"max single buffer   : {s.max_buffer_gb:.2f} GB")
    print(f"swap free           : {s.swap_free_gb:.2f} GB" if s.swap_free_gb is not None
          else "swap free           : unknown")
    print(f"wired now (baseline): {s.wired_now_gb:.2f} GB")
    print(f"safe threshold       : {s.safe_threshold_gb(margin):.2f} GB  "
          f"(wall − {margin:g}GB margin)")


def cmd_health(args):
    """Live 'can I run things safely right now?' snapshot: system pressure + per-model go/no-go."""
    color = sys.stdout.isatty()

    def c(text, name):
        codes = {"green": "32", "red": "31", "yellow": "33"}
        return f"\033[{codes[name]}m{text}\033[0m" if color else text

    margin = _configured_margin(args.margin)
    s = read_limits()
    threshold = s.safe_threshold_gb(margin)
    live_base = sample_settled_baseline()  # same baseline `run` uses, sampled once

    print(f"device              : {s.device}")
    print(f"crash wall          : {s.wall_gb:.2f} GB")
    print(f"safe threshold      : {threshold:.2f} GB  (wall − {margin:g}GB margin)")
    if s.swap_free_gb is None:
        print("swap free           : unknown")
    elif s.swap_free_gb < 2.0:
        print(f"swap free           : {s.swap_free_gb:.2f} GB   "
              + c("⚠ low — crossing the wall may hard-lock", "yellow"))
    else:
        print(f"swap free           : {s.swap_free_gb:.2f} GB")
    print(f"wired now           : {live_base:.2f} GB")
    print(f"free headroom       : {threshold - live_base:.2f} GB  (threshold − wired now)")

    con = db.connect()
    rows = con.execute(
        "SELECT DISTINCT m.hf_id, m.max_context FROM models m "
        "JOIN probe_runs r ON r.hf_id = m.hf_id JOIN fits f ON f.run_id = r.id "
        "ORDER BY m.hf_id"
    ).fetchall()
    if not rows:
        print("\nno characterized models yet — run `characterize <hf_id>`")
        return

    print("\ncharacterized models — can it load & run safely now?")
    width = max(len(r["hf_id"]) for r in rows)
    for r in rows:
        fit = db.latest_fit(con, r["hf_id"])
        if not fit or not fit.get("slope_gb_per_k"):
            continue
        pred = launcher.predict(
            model_base_gb=float(fit["model_base_gb"]),
            slope_gb_per_k=float(fit["slope_gb_per_k"]),
            live_base_gb=live_base, threshold_gb=threshold,
            wall_gb=s.wall_gb, model_max=r["max_context"],
        )
        go = (not pred.breaches_wall) and pred.safe_ctx >= launcher.MIN_USEFUL_CTX
        glyph = c("✓", "green") if go else c("✗", "red")
        hr = f"{'+' if pred.headroom_gb >= 0 else '-'}{abs(pred.headroom_gb):.2f}GB"
        if pred.breaches_wall:
            tail = c("won't load safely", "red")
        elif not go:
            tail = c(f"safe≈{pred.safe_ctx:,} ctx (too small)", "red")
        else:
            tail = f"safe≈{pred.safe_ctx:,} ctx"
        print(f"  {glyph}  {r['hf_id']:<{width}}  base {pred.base_abs_gb:5.2f}GB  "
              f"{hr:>9}  {tail}")


def cmd_scan(_):
    con = db.connect()
    found = models.scan_cache()
    n = 0
    for hf_id in found:
        info = models.describe(hf_id)
        if info is None or not info.is_causal:
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
    if not info.is_causal:
        print(f"WARNING: Model {args.hf_id} is in HF cache but is not a supported causal language model.")
    for k, v in info.as_dict().items():
        print(f"  {k:18}: {v}")
    bpt = info.fp16_kv_bytes_per_token()
    print(f"  fp16 KV bytes/token: {bpt:.0f} "
          f"(~{bpt * 1000 / 1e9:.4f} GB per 1k tokens)")


def cmd_characterize(args):
    margin = _configured_margin(args.margin)
    probe.characterize(args.hf_id, margin_gb=margin, allow_min_probe=args.min_probe,
                       repeats=args.repeats)


def cmd_list(_):
    con = db.connect()
    rows = db.latest_fits(con)
    if not rows:
        print("no characterized models yet — run `characterize <hf_id>`")
        return
    speeds = db.gen_speeds(con)
    for r in rows:
        line = (f"  {r['hf_id']:46} base={r['model_base_gb']:5.1f}GB "
                f"slope={r['slope_gb_per_k']:.4f}  safe≈{r['safe_ceiling_ctx']:>7,}  "
                f"wall≈{r['hard_wall_ctx']:>7,}  (R²={r['r2']})")
        s = speeds.get(r["hf_id"])
        if s:
            line += f"  gen≈{median(s):.1f} tok/s (n={len(s)})"
        if models.fit_is_stale(r["hf_id"], r["characterized_at"]):
            line += "  WARNING: fit may be stale — consider re-running characterize"
        print(line)


def cmd_web(args):
    """Launch the Flask dashboard web server."""
    try:
        from wmx_suite.web.app import create_app
    except ImportError as exc:
        raise SystemExit(
            "Flask is not installed or web modules are missing. "
            "Please run: pip install wills-wmx-suite[web] or uv sync --extra web"
        ) from exc

    app = create_app()
    if app is None:
        raise SystemExit("Failed to initialize Flask application.")

    if not args.debug:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"Launching wmx-suite dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)



RUN_HELP = """usage: wmx-suite run [--margin GB] [--force] [--dry-run] -- <mlx_lm.generate args>

Safely launch mlx_lm.generate. Picks kv-bits by cache type, caps --max-kv-size from the
measured ceiling, and refuses if the run would breach the wall.
The passthrough args must include --model <hf_id>.

  --margin GB   safety cushion under the wall (default: WMX_SUITE_MARGIN_GB or 2.0)
  --force       launch even if the planner refuses (may crash the machine)
  --dry-run     print the plan, do not launch
  --no-log      do not record generation speed (bare exec passthrough)
"""

_PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_GEN_RE = re.compile(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_PEAK_RE = re.compile(r"Peak memory:\s*([\d.]+)\s*GB")


def _record_generation(text: str, model_id: str, max_kv_size: int) -> None:
    """Parse mlx_lm's tok/s lines from captured output and store them. Best-effort —
    a cancelled or errored run has no generation stats, so nothing is logged."""
    gm = _GEN_RE.search(text)
    if not gm:
        return
    pm, km = _PROMPT_RE.search(text), _PEAK_RE.search(text)
    try:
        con = db.connect()
        db.log_generation(
            con, model_id,
            prompt_tokens=int(pm.group(1)) if pm else None,
            prompt_tps=float(pm.group(2)) if pm else None,
            gen_tokens=int(gm.group(1)),
            gen_tps=float(gm.group(2)),
            peak_gb=float(km.group(1)) if km else None,
            max_kv_size=max_kv_size,
        )
        print(f"[run] logged {gm.group(1)} tok @ {float(gm.group(2)):.1f} tok/s", file=sys.stderr)
    except Exception as e:  # logging must never break a run
        print(f"[run] (speed log failed: {e})", file=sys.stderr)


def _exec_logged(argv: list[str], model_id: str, max_kv_size: int) -> None:
    """Run mlx_lm.generate under a PTY so output streams live to the terminal unchanged,
    while we capture a copy to parse its tok/s stats. A PTY (not a plain pipe) is required:
    Python block-buffers stdout when it isn't a tty, which would batch the token stream."""
    master, slave = pty.openpty()
    if sys.stdout.isatty():  # match the child's window size so tqdm renders correctly
        try:
            cols, rows = os.get_terminal_size(sys.stdout.fileno())
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except Exception:
            pass
    proc = subprocess.Popen(["mlx_lm.generate"] + argv,
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    captured = bytearray()
    try:
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO once the child exits and closes its end
                break
            if not data:
                break
            os.write(1, data)  # tee live to our stdout, byte-for-byte
            captured.extend(data)
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        os.close(master)
    rc = proc.wait()
    _record_generation(captured.decode(errors="replace"), model_id, max_kv_size)
    sys.exit(rc)


def cmd_run_raw(run_args: list[str]):
    """Parse leading suite flags, then treat the remainder as mlx_lm.generate passthrough.

    Done manually (not argparse) because argparse.REMAINDER mishandles optionals that
    precede the positional, and we want `run --model X ...` to work as a drop-in.
    """
    margin, force, dry, log = None, False, False, True
    i = 0
    while i < len(run_args):
        a = run_args[i]
        if a in ("-h", "--help"):
            print(RUN_HELP); return
        if a == "--margin":
            if i + 1 >= len(run_args):
                raise SystemExit("[run] --margin requires a value")
            margin = run_args[i + 1]; i += 2; continue
        if a.startswith("--margin="):
            margin = a.split("=", 1)[1]; i += 1; continue
        if a == "--force":
            force = True; i += 1; continue
        if a == "--dry-run":
            dry = True; i += 1; continue
        if a == "--no-log":
            log = False; i += 1; continue
        break  # first non-suite token: the rest is passthrough
    rest = run_args[i:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    _run(rest, margin=margin, force=force, dry_run=dry, log=log)


def _run(rest: list[str], *, margin: float | str | None, force: bool,
         dry_run: bool, log: bool = True):
    """Crash-safe launch: plan a launch, then exec mlx_lm.generate."""
    model_id = None
    for i, a in enumerate(rest):
        if a == "--model" and i + 1 < len(rest):
            model_id = rest[i + 1]
            break
        if a.startswith("--model="):
            model_id = a.split("=", 1)[1]
            break
    if model_id is None:
        raise SystemExit("[run] --model is required")

    margin_gb = _configured_margin(margin)
    p = launcher.plan(model_id, margin_gb=margin_gb)
    if p.get("error"):
        raise SystemExit(f"[run] {p['error']}")

    if p["source"] == "estimated" and not force:
        if sys.stdin.isatty():
            print(f"[run] Model {model_id} has not been characterized yet.", file=sys.stderr)
            try:
                ans = input("[run] Characterize it now to find its safe context ceiling? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                print(file=sys.stderr)
                raise SystemExit("[run] Aborted.")
            if ans in ("y", "yes"):
                print(f"[run] Running characterization for {model_id}...", file=sys.stderr)
                try:
                    probe.characterize(model_id, margin_gb=margin_gb, allow_min_probe=True)
                    # Re-plan with the newly saved fit
                    p = launcher.plan(model_id, margin_gb=margin_gb)
                    if p.get("error"):
                        raise SystemExit(f"[run] Re-planning failed: {p['error']}")
                except Exception as e:
                    raise SystemExit(f"[run] Characterization failed: {e}")
            else:
                raise SystemExit("[run] Aborted. Run characterize first or pass --force to run with estimated limits.")
        else:
            raise SystemExit("[run] REFUSED: Model is uncharacterized and shell is non-interactive. Run characterize first or pass --force.")

    kv = "fp16 (RotatingKVCache — not quantizable)" if p["kv_bits"] is None else f"{p['kv_bits']}-bit"
    print(f"[run] {model_id}", file=sys.stderr)
    print(f"[run] source={p['source']}  cache={p['cache_type']}  kv={kv}", file=sys.stderr)
    if p.get("fit_stale"):
        print("[run] WARNING: fit may be stale — consider re-running characterize",
              file=sys.stderr)
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

    try:
        argv = launcher.build_argv(rest, p, force=force)
    except launcher.LaunchArgumentError as exc:
        raise SystemExit(f"[run] REFUSED: {exc}") from exc
    effective_cap = launcher.effective_max_kv_size(rest, p)
    prompt_check = None
    try:
        tokenizer = load_tokenizer(
            model_id,
            tokenizer_config_extra=launcher.tokenizer_config(rest),
        )
        prompt_check = launcher.check_prompt(rest, p, tokenizer)
    except launcher.LaunchArgumentError as exc:
        if not force:
            raise SystemExit(f"[run] REFUSED: {exc}") from exc
        print(f"[run] WARNING: prompt preflight bypassed: {exc}", file=sys.stderr)
    except Exception as exc:
        if not force:
            raise SystemExit(
                f"[run] REFUSED: prompt tokenization failed: {exc}; "
                "pass --force to bypass prompt preflight"
            ) from exc
        print(f"[run] WARNING: prompt tokenization failed and was bypassed: {exc}",
              file=sys.stderr)
    if not p.get("max_kv_size_enforced", True):
        print("[run] WARNING: this model's custom MLX cache does not enforce "
              "--max-kv-size", file=sys.stderr)
    if prompt_check is not None:
        print(f"[run] prompt {prompt_check.tokens:,} tokens "
              f"({prompt_check.tokens / prompt_check.cap:.0%} of cap)",
              file=sys.stderr)
        if prompt_check.tokens > prompt_check.cap:
            message = (f"prompt is {prompt_check.tokens:,} tokens, above "
                       f"the {prompt_check.cap:,}-token cap")
            if not force:
                raise SystemExit(f"[run] REFUSED: {message}")
            print(f"[run] WARNING: {message}; --force is overriding this refusal",
                  file=sys.stderr)
        if prompt_check.warn:
            print(f"[run] WARNING: prompt exceeds "
                  f"{launcher.PROMPT_WARNING_FRACTION:.0%} of the context cap",
                  file=sys.stderr)
    enforcement = "" if p.get("max_kv_size_enforced", True) else " (NOT runtime-enforced)"
    print(f"[run] max-kv-size {effective_cap:,} tokens{enforcement} "
          f"(model cap {p['model_max']:,})", file=sys.stderr)
    print(f"[run] exec: mlx_lm.generate {' '.join(argv)}\n", file=sys.stderr)
    if dry_run:
        print("[run] --dry-run: not launching.", file=sys.stderr)
        return
    if log:
        _exec_logged(argv, model_id, effective_cap)
    else:
        os.execvp("mlx_lm.generate", ["mlx_lm.generate"] + argv)


def _main_argparse():
    ap = argparse.ArgumentParser(prog="wmx-suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("system").set_defaults(func=cmd_system)
    p = sub.add_parser("health")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_health)
    sub.add_parser("scan").set_defaults(func=cmd_scan)
    p = sub.add_parser("show"); p.add_argument("hf_id"); p.set_defaults(func=cmd_show)
    p = sub.add_parser("characterize"); p.add_argument("hf_id")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.add_argument("--min-probe", action="store_true",
                   help="for borderline models, run a supervised 512-token probe to measure "
                        "the true base instead of refusing on the pessimistic estimate")
    p.add_argument("--repeats", type=int, default=probe.DEFAULT_REPEATS,
                   help="isolated runs per context rung; the median high-water is used "
                        "(smooths prefill-transient jitter)")
    p.set_defaults(func=cmd_characterize)
    sub.add_parser("list").set_defaults(func=cmd_list)
    p = sub.add_parser("web", help="Launch the Flask web UI dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    p.add_argument("--port", type=int, default=5001, help="Port to listen on")
    p.add_argument("--debug", action="store_true", help="Run in Flask debug mode")
    p.set_defaults(func=cmd_web)
    # `run` is intercepted before argparse (see below) so it can pass arbitrary flags
    # through to mlx_lm.generate; this stub only makes it show up in `--help`.
    sub.add_parser("run", help="safely launch mlx_lm.generate: picks kv-bits by cache "
                               "type, caps --max-kv-size from the measured ceiling, "
                               "refuses if it would breach the wall")
    args = ap.parse_args()
    args.func(args)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        return cmd_run_raw(argv[1:])
    return _main_argparse()


if __name__ == "__main__":
    main()

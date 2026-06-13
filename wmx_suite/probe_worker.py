"""Single isolated memory measurement for one (model, context) pair.

Run as a subprocess — one fresh process per context — so wired-memory residue from a
previous context never contaminates the high-water reading. Prints one JSON line.

Usage:
    python -m wmx_suite.probe_worker <hf_id> <context> [--kv-bits N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time


def _wired_gb() -> float:
    out = subprocess.check_output(["vm_stat"]).decode()
    page_size, wired = 4096, 0
    for line in out.splitlines():
        if "page size of" in line:
            page_size = int(line.split()[-2])
        if "Pages wired down" in line:
            wired = int(line.split()[-1].strip("."))
    return wired * page_size / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id")
    ap.add_argument("context", type=int)
    ap.add_argument("--kv-bits", type=int, default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=5000)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import generate, load

    # sample OS-wired memory continuously; report the high-water mark
    hi = [0.0]
    stop = [False]

    def sampler():
        while not stop[0]:
            hi[0] = max(hi[0], _wired_gb())
            time.sleep(0.05)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    baseline = _wired_gb()
    model, tok = load(args.hf_id)

    # build a prompt of exactly `context` tokens from repeated filler
    filler = "The quick brown fox jumps over the lazy dog. " * 20000
    ids = tok.encode(filler)
    result = {"hf_id": args.hf_id, "context": args.context,
              "baseline_wired_gb": round(baseline, 3)}
    if args.context > len(ids):
        result.update(status="error", note="not enough filler tokens")
        print(json.dumps(result), flush=True)
        return

    prompt = tok.decode(ids[: args.context])
    gen_kwargs = dict(max_tokens=args.max_tokens, verbose=False)
    if args.kv_bits is not None:
        gen_kwargs.update(kv_bits=args.kv_bits, kv_group_size=args.kv_group_size,
                          quantized_kv_start=args.quantized_kv_start)

    mx.clear_cache()
    mx.reset_peak_memory()
    hi[0] = 0.0
    time.sleep(0.2)
    try:
        generate(model, tok, prompt=prompt, **gen_kwargs)
    except Exception as e:  # e.g. RotatingKVCache Quantization NYI
        stop[0] = True
        result.update(status="error", note=f"{type(e).__name__}: {e}")
        print(json.dumps(result), flush=True)
        return

    stop[0] = True
    result.update(
        status="ok",
        mlx_peak_gb=round(mx.get_peak_memory() / 1e9, 3),
        mlx_true_gb=round((mx.get_active_memory() + mx.get_cache_memory()) / 1e9, 3),
        os_wired_gb=round(hi[0], 3),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

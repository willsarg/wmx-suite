"""Kokoro TTS active synthesis memory baseline benchmark worker.

Measures the exact fixed OS-wired memory overhead (GB) occupied by Kokoro TTS
during active synthesis.
Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import json
import time
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    try:
        import mlx.core as mx
        from kokoro_mlx import KokoroTTS
        from kokoro_mlx.generate import generate
        from wmx_suite.system import read_limits, sample_settled_baseline
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # 1. Settle and measure baseline system wired memory
    limits = read_limits()
    baseline_gb = sample_settled_baseline(settle=0.5, n=3, interval=0.2)

    # Check pre-flight
    threshold = limits.safe_threshold_gb(args.margin)
    if baseline_gb >= threshold:
        print(json.dumps({
            "status": "error",
            "note": f"Pre-flight aborted: System baseline memory ({baseline_gb:.2f} GB) is already at or above safe threshold ({threshold:.2f} GB)."
        }), flush=True)
        sys.exit(0)

    # 2. Load model and run warm synthesis
    try:
        tts = KokoroTTS.from_pretrained(args.model)
        phonemizer = tts._get_phonemizer(None, args.voice)
        
        # Warmup active generation to initialize Metal graphs and buffers
        generate(
            text="This is a warm baseline memory measurement.",
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice,
            phonemizer=phonemizer
        )
        mx.clear_cache()
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Model initialization or synthesis failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Measure active system memory (settled)
    active_gb = sample_settled_baseline(settle=0.5, n=3, interval=0.2)
    
    # Calculate overhead
    overhead_gb = active_gb - baseline_gb
    if overhead_gb < 0:
        # Prevent negative values due to OS caching flux
        overhead_gb = 0.0

    print(json.dumps({
        "status": "rung_done",
        "baseline_gb": round(baseline_gb, 3),
        "active_gb": round(active_gb, 3),
        "overhead_gb": round(overhead_gb, 3)
    }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()

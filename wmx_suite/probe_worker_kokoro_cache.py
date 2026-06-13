"""Kokoro TTS voice cache memory benchmark worker.

Sweeps cache sizes (number of active speaker style vectors loaded in memory)
to measure OS-wired memory usage and MLX peak memory scaling, enforcing RAM safeguards.
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
    ap.add_argument("--cache-sizes", default="0,1,2,4,8,16,24,32")
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    try:
        cache_sizes = [int(x.strip()) for x in args.cache_sizes.split(",") if x.strip()]
    except ValueError:
        print(json.dumps({"status": "error", "note": "Invalid cache-sizes argument"}), flush=True)
        sys.exit(1)

    try:
        import mlx.core as mx
        from kokoro_mlx import KokoroTTS
        from kokoro_mlx.generate import generate
        from wmx_suite.system import read_limits, wired_gb
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # Resolve limits and safe threshold
    limits = read_limits()
    threshold = limits.safe_threshold_gb(args.margin)

    # Pre-flight check
    if limits.wired_now_gb >= threshold:
        print(json.dumps({
            "status": "error",
            "note": f"Pre-flight aborted: System wired memory ({limits.wired_now_gb:.2f} GB) is already at or above safe threshold ({threshold:.2f} GB)."
        }), flush=True)
        sys.exit(0)

    # 1. Load model
    try:
        tts = KokoroTTS.from_pretrained(args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    voices = tts.list_voices()
    if not voices:
        print(json.dumps({"status": "error", "note": "No voices found in model registry."}), flush=True)
        sys.exit(1)

    # 2. Warmup run
    try:
        phonemizer = tts._get_phonemizer(None, voices[0])
        generate(
            text="warmup",
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=voices[0],
            phonemizer=phonemizer
        )
        mx.clear_cache()
        print(json.dumps({"status": "warmup_done"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Warmup run failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Sweep over cache sizes
    for cache_size in cache_sizes:
        if cache_size < 0:
            continue
        if cache_size > len(voices):
            cache_size = len(voices)

        # Active RAM check
        try:
            current_wired = wired_gb()
        except Exception:
            current_wired = limits.wired_now_gb
        if current_wired >= threshold:
            print(json.dumps({
                "status": "safeguard_triggered",
                "note": f"Active memory safeguard triggered during cache sweep: System wired memory ({current_wired:.2f} GB) reached safe threshold ({threshold:.2f} GB)."
            }), flush=True)
            break

        # Clear cache and evaluate memory to ensure clean baseline for this rung
        tts._voices._cache.clear()
        mx.clear_cache()
        time.sleep(0.1)

        # Load the target number of voices
        voices_to_load = voices[:cache_size]
        try:
            for voice_name in voices_to_load:
                tts._voices.load_voice(voice_name)
            
            # Settle a little for accurate memory readings
            time.sleep(0.05)
            
            os_wired_gb = wired_gb()
            peak_gb = mx.get_peak_memory() / 1e9
        except Exception as e:
            print(json.dumps({"status": "error", "note": f"Voice load failed at cache size {cache_size}: {e}"}), flush=True)
            break

        print(json.dumps({
            "status": "rung_done",
            "cache_size": cache_size,
            "os_wired_gb": round(os_wired_gb, 3),
            "peak_gb": round(peak_gb, 3)
        }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()

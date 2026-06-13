"""Kokoro TTS batch performance benchmark worker.

Runs concurrent synthesis sweeps within a single process to measure the impact of 
batch size (concurrency) on aggregate throughput (CPS) and peak GPU memory footprint,
enforcing active RAM safeguards. Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import sys

# 16 sentences of approximately 100 characters each to benchmark concurrency levels 1 to 16
BENCHMARK_SENTENCES = [
    "The sun rises gracefully over the sleeping mountains, casting a golden glow on the quiet valley below.",
    "Technology continues to evolve at an incredible speed, transforming how we work and connect every day.",
    "Reading books allows us to travel to distant places and experience life through many different eyes.",
    "A balanced diet and regular physical exercise are essential components of a healthy and happy lifestyle.",
    "The soft sound of rain falling on the leaves creates a peaceful and calming atmosphere in the forest.",
    "Learning a new language opens up exciting opportunities to explore diverse cultures around the world.",
    "Deep learning models are achieving remarkable performance in speech synthesis and computer vision.",
    "The historic city features narrow cobblestone streets, ancient stone buildings, and beautiful plazas.",
    "Developing software requires patience, logical thinking, and a commitment to continuous learning.",
    "Exploring the deep ocean reveals mysterious creatures that thrive in complete darkness and high pressure.",
    "Kindness is a simple yet powerful gift that can brighten someone's day and build stronger communities.",
    "The international space station orbits the Earth every ninety minutes, hosting research from many nations.",
    "Fresh coffee in the morning provides a wonderful aroma and a gentle energy boost to start the day.",
    "Protecting natural habitats is crucial for preserving the diverse plants and animals on our planet.",
    "Artistic expression takes many forms, including painting, music, poetry, and theatrical performance.",
    "Understanding historical events helps us make better decisions and navigate the challenges of the future."
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    try:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    except ValueError:
        print(json.dumps({"status": "error", "note": "Invalid batch-sizes argument"}), flush=True)
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

    # Pre-flight check: refuse to start if baseline already exceeds threshold
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

    phonemizer = tts._get_phonemizer(None, args.voice)

    # 2. Warmup run to compile Metal GPU kernels
    try:
        # Pre-load voice
        tts._voices.load_voice(args.voice)
        generate(
            text="warmup",
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice,
            phonemizer=phonemizer
        )
        mx.clear_cache()
        print(json.dumps({"status": "warmup_done"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Warmup run failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Sweep over batch sizes
    for batch_size in batch_sizes:
        if batch_size <= 0:
            continue
        if batch_size > len(BENCHMARK_SENTENCES):
            # Limit to the max available sentences
            batch_size = len(BENCHMARK_SENTENCES)

        # Check active RAM safeguard
        try:
            current_wired = wired_gb()
        except Exception:
            current_wired = limits.wired_now_gb  # fallback
            
        if current_wired >= threshold:
            print(json.dumps({
                "status": "safeguard_triggered",
                "note": f"Active memory safeguard triggered: System wired memory ({current_wired:.2f} GB) reached safe threshold ({threshold:.2f} GB)."
            }), flush=True)
            break

        sentences_to_use = BENCHMARK_SENTENCES[:batch_size]
        total_chars = sum(len(txt) for txt in sentences_to_use)

        run_times = []
        run_cpss = []
        run_peaks = []

        success = True

        def run_one_generation(text: str):
            # Bypasses the KokoroTTS instance lock and calls the underlying generate()
            # directly from multiple threads.  Thread-safety of concurrent generate() is
            # an ASSUMPTION (the MLX Metal runtime and kokoro_mlx internals are not
            # formally documented as thread-safe); no race has been observed in practice,
            # but this is not verified.  Plain Python strings are passed to keep all MLX
            # array allocations in-thread.
            return generate(
                text=text,
                model=tts._model,
                config=tts._config,
                voice_manager=tts._voices,
                voice=args.voice,
                phonemizer=phonemizer
            )

        for _ in range(max(1, args.repeats)):
            mx.clear_cache()
            mx.reset_peak_memory()
            time.sleep(0.05)  # small settle

            t0 = time.perf_counter()
            try:
                # NOTE: the wired-memory safeguard above runs only BETWEEN rungs, not
                # during a batch.  Concurrent synthesis transiently multiplies per-call
                # allocations, so a brief spike above the between-rung check is possible
                # before the next rung's check catches it.  This is acceptable here
                # because Kokoro is a small (82 M-param), static-footprint model whose
                # per-call transient allocation is low; it would be unsafe for larger
                # models or dynamically-growing workloads.
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = [executor.submit(run_one_generation, text) for text in sentences_to_use]
                    # Wait for all thread jobs to finish
                    _ = [fut.result() for fut in futures]
                t1 = time.perf_counter()
            except Exception as e:
                print(json.dumps({"status": "error", "note": f"Generation failed at batch size {batch_size}: {e}"}), flush=True)
                success = False
                break

            elapsed = t1 - t0
            if elapsed <= 0:
                elapsed = 0.001

            run_times.append(elapsed)
            run_cpss.append(total_chars / elapsed)
            run_peaks.append(mx.get_peak_memory() / 1e9)  # GB

        if not success:
            continue

        # Emit median values for the rung
        print(json.dumps({
            "status": "rung_done",
            "batch_size": batch_size,
            "total_time": round(statistics.median(run_times), 3),
            "cps": round(statistics.median(run_cpss), 2),
            "peak_gb": round(statistics.median(run_peaks), 3)
        }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streaming TTS Benchmark Script for Qwen3-TTS

Measures per prompt:
- Cold start (first request after server start)
- Warm runs (median/min/max/p95 of N runs)
- first_chunk_time_s (TTFB)
- total_time_s
- audio_duration_s
- chunks
- bytes_total
- throughput_bytes_s
- RTF (Real-Time Factor) = total_time_s / audio_duration_s
"""

import io
import sys
import wave
import time
import json
import argparse
import statistics as stats
from pathlib import Path
from datetime import datetime

import requests


# Test prompts of varying lengths
PROMPTS = [
    ("2 words", "Hello world"),
    ("short sentence", "Kia ora koutou, welcome to today's meeting."),
    ("medium paragraph",
     "The quick brown fox jumps over the lazy dog near the riverbank. "
     "This is a test of text-to-speech generation quality."),
    ("long paragraph",
     "Artificial intelligence has revolutionized the way we interact with technology. "
     "Text-to-speech technology has advanced significantly in recent years. "
     "Modern neural networks can generate remarkably natural-sounding speech. "
     "The Qwen3-TTS model represents the latest breakthrough in this field."),
]


def p95(values: list[float]) -> float:
    """Return p95 using inclusive quantiles for small sample sizes."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return stats.quantiles(values, n=100, method="inclusive")[94]


def _round(v: float | int, ndigits: int = 3):
    if isinstance(v, int):
        return v
    if v == float("inf"):
        return v
    return round(v, ndigits)


def summarize_metric(values: list[float | int]) -> dict:
    """Compute median/min/max/p95 for a metric list."""
    vals = [float(v) for v in values]
    return {
        "median": _round(stats.median(vals)),
        "min": _round(min(vals)),
        "max": _round(max(vals)),
        "p95": _round(p95(vals)),
    }


def pcm_duration_s(audio_bytes: bytes, sample_rate: int) -> float:
    """Calculate duration for 16-bit mono PCM stream."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0 for PCM")
    if not audio_bytes:
        raise ValueError("Received empty PCM audio payload")

    # 16-bit PCM: 2 bytes/sample, mono.
    sample_count = len(audio_bytes) / 2.0
    return sample_count / float(sample_rate)


def wav_duration_s(wav_bytes: bytes) -> tuple[float, int]:
    """Parse WAV header/data and return (duration_s, data_bytes)."""
    if not wav_bytes:
        raise ValueError("Received empty WAV payload")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frame_rate = wf.getframerate()
        frame_count = wf.getnframes()
        if frame_rate <= 0:
            raise ValueError("Invalid WAV sample rate in header")
        duration_s = frame_count / float(frame_rate)
        data_bytes = len(wf.readframes(frame_count))

    return duration_s, data_bytes


def get_health(base_url: str) -> dict:
    """Get server health status."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def post_tts_stream(
    base_url: str,
    text: str,
    voice: str = "Vivian",
    response_format: str = "pcm",
    chunk_size: int = 4096,
    sample_rate: int = 24000,
) -> tuple[bytes, dict]:
    """Make a streaming TTS request and return audio bytes with metrics."""
    if response_format not in {"pcm", "wav"}:
        raise ValueError("response_format must be one of: pcm, wav")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    url = f"{base_url}/v1/audio/speech"
    payload = {
        "model": "qwen3-tts",
        "voice": voice,
        "input": text,
        "stream": True,
        "response_format": response_format,
    }

    t0 = time.perf_counter()
    first_chunk_time_s = None
    chunks = 0
    total_bytes = 0
    parts: list[bytes] = []

    try:
        with requests.post(url, json=payload, timeout=600, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                if first_chunk_time_s is None:
                    first_chunk_time_s = time.perf_counter() - t0
                chunks += 1
                total_bytes += len(chunk)
                parts.append(chunk)
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:500] if e.response is not None else ""
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP error during streaming request: {e}. Body: {body}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during streaming request: {e}") from e

    total_time_s = time.perf_counter() - t0

    if first_chunk_time_s is None:
        raise RuntimeError("Stream completed without any audio chunks")

    audio = b"".join(parts)
    if not audio:
        raise RuntimeError("Stream produced zero bytes of audio")

    if response_format == "pcm":
        audio_duration_s = pcm_duration_s(audio, sample_rate=sample_rate)
        parsed_data_bytes = total_bytes
    else:
        audio_duration_s, parsed_data_bytes = wav_duration_s(audio)

    throughput_bytes_s = total_bytes / total_time_s if total_time_s > 0 else float("inf")
    rtf = total_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")

    metrics = {
        "first_chunk_time_s": first_chunk_time_s,
        "total_time_s": total_time_s,
        "audio_duration_s": audio_duration_s,
        "chunks": chunks,
        "bytes_total": total_bytes,
        "parsed_data_bytes": parsed_data_bytes,
        "throughput_bytes_s": throughput_bytes_s,
        "rtf": rtf,
    }
    return audio, metrics


def bench(
    base_url: str,
    label: str,
    warm_runs: int = 5,
    output_dir: Path = None,
    response_format: str = "pcm",
    chunk_size: int = 4096,
    sample_rate: int = 24000,
) -> dict:
    """Run benchmark and return results."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"Server: {base_url}")
    print(f"Response format: {response_format}")
    print(f"Chunk size: {chunk_size}")
    if response_format == "pcm":
        print(f"PCM sample rate: {sample_rate}")
    print(f"Warm runs: {warm_runs}")
    print(f"{'='*70}")

    health = get_health(base_url)
    print(f"\nServer health: {health.get('status', 'unknown')}")
    if "backend" in health:
        print(f"Backend: {health['backend'].get('name', 'unknown')}")
        print(f"Model: {health['backend'].get('model_id', 'unknown')}")

    results = {
        "label": label,
        "base_url": base_url,
        "timestamp": datetime.now().isoformat(),
        "health": health,
        "streaming": {
            "response_format": response_format,
            "chunk_size": chunk_size,
            "sample_rate": sample_rate,
        },
        "prompts": [],
    }

    metric_names = [
        "first_chunk_time_s",
        "total_time_s",
        "audio_duration_s",
        "chunks",
        "bytes_total",
        "throughput_bytes_s",
        "rtf",
    ]

    for name, text in PROMPTS:
        word_count = len(text.split())
        char_count = len(text)

        print(f"\n[{name}] ({word_count} words, {char_count} chars)")
        print(f"  Text: {text[:50]}..." if len(text) > 50 else f"  Text: {text}")

        prompt_result = {
            "name": name,
            "text": text,
            "word_count": word_count,
            "char_count": char_count,
        }

        try:
            # Cold run
            cold_audio, cold = post_tts_stream(
                base_url=base_url,
                text=text,
                response_format=response_format,
                chunk_size=chunk_size,
                sample_rate=sample_rate,
            )

            prompt_result["cold"] = {
                k: _round(v) for k, v in cold.items()
            }
            print(
                "  Cold: "
                f"TTFB {cold['first_chunk_time_s']:.3f}s | "
                f"total {cold['total_time_s']:.3f}s | "
                f"audio {cold['audio_duration_s']:.3f}s | "
                f"chunks {cold['chunks']} | bytes {cold['bytes_total']} | "
                f"throughput {cold['throughput_bytes_s']:.1f} B/s | "
                f"RTF {cold['rtf']:.3f}"
            )

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                ext = "pcm" if response_format == "pcm" else "wav"
                cold_name = f"{name.replace(' ', '_')}_cold.{ext}"
                (output_dir / cold_name).write_bytes(cold_audio)

            # Warm runs
            warm_runs_data = []
            for i in range(warm_runs):
                warm_audio, run_metrics = post_tts_stream(
                    base_url=base_url,
                    text=text,
                    response_format=response_format,
                    chunk_size=chunk_size,
                    sample_rate=sample_rate,
                )
                warm_runs_data.append(run_metrics)
                print(
                    f"    Run {i+1}: "
                    f"TTFB {run_metrics['first_chunk_time_s']:.3f}s | "
                    f"total {run_metrics['total_time_s']:.3f}s | "
                    f"RTF {run_metrics['rtf']:.3f}"
                )

                if output_dir:
                    ext = "pcm" if response_format == "pcm" else "wav"
                    run_name = f"{name.replace(' ', '_')}_warm_{i+1}.{ext}"
                    (output_dir / run_name).write_bytes(warm_audio)

            warm_stats = {}
            for metric in metric_names:
                values = [run[metric] for run in warm_runs_data]
                warm_stats[metric] = summarize_metric(values)

            prompt_result["warm"] = {
                "runs": warm_runs,
                "stats": warm_stats,
            }

            print(
                "  Warm summary: "
                f"TTFB median {warm_stats['first_chunk_time_s']['median']:.3f}s | "
                f"total median {warm_stats['total_time_s']['median']:.3f}s | "
                f"RTF median {warm_stats['rtf']['median']:.3f}"
            )
            print(
                "  Warm total time stats: "
                f"p95 {warm_stats['total_time_s']['p95']:.3f}s | "
                f"min {warm_stats['total_time_s']['min']:.3f}s | "
                f"max {warm_stats['total_time_s']['max']:.3f}s"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            prompt_result["error"] = str(e)

        results["prompts"].append(prompt_result)

    print(f"\n{'-'*70}")
    print("SUMMARY")
    print(f"{'-'*70}")

    successful = [p for p in results["prompts"] if "warm" in p]
    if successful:
        med_total = [p["warm"]["stats"]["total_time_s"]["median"] for p in successful]
        med_ttfb = [p["warm"]["stats"]["first_chunk_time_s"]["median"] for p in successful]
        med_rtf = [p["warm"]["stats"]["rtf"]["median"] for p in successful]

        results["summary"] = {
            "successful_prompts": len(successful),
            "total_prompts": len(PROMPTS),
            "avg_warm_median_total_time_s": _round(sum(med_total) / len(med_total)),
            "avg_warm_median_ttfb_s": _round(sum(med_ttfb) / len(med_ttfb)),
            "avg_warm_median_rtf": _round(sum(med_rtf) / len(med_rtf)),
        }

        print(f"Successful: {len(successful)}/{len(PROMPTS)} prompts")
        print(f"Avg warm median total time: {results['summary']['avg_warm_median_total_time_s']:.3f}s")
        print(f"Avg warm median TTFB: {results['summary']['avg_warm_median_ttfb_s']:.3f}s")
        print(f"Avg warm median RTF: {results['summary']['avg_warm_median_rtf']:.3f}")
    else:
        results["summary"] = {
            "successful_prompts": 0,
            "total_prompts": len(PROMPTS),
        }
        print("No successful prompts.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Benchmark Script")
    parser.add_argument(
        "--url",
        default="http://localhost:8880",
        help="Server base URL (default: http://localhost:8880)",
    )
    parser.add_argument(
        "--label",
        default="CURRENT_BACKEND_STREAMING",
        help="Label for this benchmark run",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=5,
        help="Number of warm runs per prompt (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--response-format",
        choices=["pcm", "wav"],
        default="pcm",
        help="Streaming response format (default: pcm)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="iter_content chunk size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="PCM sample rate in Hz used for duration calc (default: 24000)",
    )

    args = parser.parse_args()

    if args.warm_runs < 1:
        parser.error("--warm-runs must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.response_format == "pcm" and args.sample_rate < 1:
        parser.error("--sample-rate must be >= 1 for PCM")

    results = bench(
        base_url=args.url,
        label=args.label,
        warm_runs=args.warm_runs,
        output_dir=args.output_dir,
        response_format=args.response_format,
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

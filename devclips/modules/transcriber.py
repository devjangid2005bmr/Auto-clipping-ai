
from __future__ import annotations

import os
import json
import time
import subprocess
import multiprocessing as mp

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DOWNLOAD_DIR = PROJECT_DIR / "storage" / "downloads"
TRANSCRIPT_DIR = PROJECT_DIR / "storage" / "transcripts"
AUDIO_DIR = TRANSCRIPT_DIR / "audio"
CHUNK_DIR = TRANSCRIPT_DIR / "chunks"

load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

CHUNK_SECONDS = 180
OVERLAP_SECONDS = 2

# Start with 2 workers. Benchmark 1 vs 2.
MAX_WORKERS = 2

# Each worker loads its own model.
# Keep this low on an 8GB RAM system.


# ==================================================
# AUDIO EXTRACTION
# ==================================================

def extract_audio(video_path: Path, audio_path: Path):

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]

    print("\nExtracting audio...")

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("Audio extraction completed.")


# ==================================================
# GET AUDIO DURATION
# ==================================================

def get_audio_duration(audio_path: Path) -> float:

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


# ==================================================
# SPLIT AUDIO INTO CHUNKS
# ==================================================

def split_audio(audio_path: Path, duration: float):

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old chunks for this video.
    for old_file in CHUNK_DIR.glob("chunk_*.wav"):
        old_file.unlink()

    chunks = []

    start = 0.0
    index = 0

    while start < duration:

        end = min(start + CHUNK_SECONDS, duration)

        # Include overlap on both sides where possible.
        actual_start = max(0, start - OVERLAP_SECONDS)
        actual_end = min(duration, end + OVERLAP_SECONDS)

        chunk_path = CHUNK_DIR / f"chunk_{index:03d}.wav"

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(actual_start),
            "-i", str(audio_path),
            "-t", str(actual_end - actual_start),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(chunk_path)
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        chunks.append({
            "index": index,
            "path": str(chunk_path),
            "offset": actual_start,
            "core_start": start,
            "core_end": end,
            "duration": actual_end - actual_start
        })

        print(
            f"Created chunk {index + 1}: "
            f"{start:.1f}s - {end:.1f}s"
        )

        start = end
        index += 1

    return chunks


# ==================================================
# WORKER FUNCTION
# Each process loads its own Whisper model.
# ==================================================

def transcribe_chunk(chunk):

    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=2 if DEVICE == "cpu" else 0
    )

    segments, info = model.transcribe(
        chunk["path"],
        language="hi",
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=(
            "Ye Hindi aur Hinglish interview hai. "
            "Hindi words Devanagari mein aur English words "
            "Roman English mein likho. Names accurately "
            "transcribe karo."
        )
    )

    result_words = []

    for segment in segments:

        if not segment.words:
            continue

        for word in segment.words:

            # Convert chunk-relative timestamp
            # to original full-video timestamp.
            global_start = chunk["offset"] + float(word.start)
            global_end = chunk["offset"] + float(word.end)

            # Keep only words inside this chunk's core interval.
            # This removes overlap duplicates.
            midpoint = (global_start + global_end) / 2

            if not (
                chunk["core_start"] <= midpoint < chunk["core_end"]
            ):
                continue

            result_words.append({
                "word": word.word.strip(),
                "start": round(global_start, 3),
                "end": round(global_end, 3)
            })

    return {
        "index": chunk["index"],
        "words": result_words,
        "language": info.language
    }


# ==================================================
# MERGE RESULTS
# ==================================================

def merge_results(results):

    all_words = []

    for result in results:
        all_words.extend(result["words"])

    # Sort by original video timestamp.
    all_words.sort(key=lambda item: (item["start"], item["end"]))

    # Final duplicate protection.
    cleaned = []

    for word in all_words:

        if not word["word"]:
            continue

        if cleaned:
            previous = cleaned[-1]

            if (
                word["word"].lower() == previous["word"].lower()
                and abs(word["start"] - previous["start"]) < 0.35
            ):
                continue

        cleaned.append(word)

    return cleaned


# ==================================================
# BUILD SEGMENTS FROM WORDS
# ==================================================

def build_segments(words, max_gap=1.0):

    segments = []
    current = []

    for word in words:

        if current:
            gap = word["start"] - current[-1]["end"]

            if gap > max_gap:
                segments.append(current)
                current = []

        current.append(word)

    if current:
        segments.append(current)

    output = []

    for index, group in enumerate(segments, 1):

        output.append({
            "segment_id": index,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(w["word"] for w in group),
            "words": group
        })

    return output


# ==================================================
# MAIN TRANSCRIPTION PIPELINE
# ==================================================

def transcribe_video(video_path: Path):

    start_time = time.time()

    video_path = video_path.resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video_id = video_path.stem

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_DIR / f"{video_id}.wav"
    output_path = TRANSCRIPT_DIR / f"{video_id}.json"

    if not audio_path.exists():
        extract_audio(video_path, audio_path)
    else:
        print("\nUsing existing audio.")

    duration = get_audio_duration(audio_path)

    print(f"\nAudio duration: {duration / 60:.2f} minutes")
    print(f"Chunk duration: {CHUNK_SECONDS} seconds")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Model: {MODEL_NAME}")

    chunks = split_audio(audio_path, duration)

    print(f"\nTotal chunks: {len(chunks)}")
    print("\nStarting parallel transcription...\n")

    results = []

    # Spawn avoids inheriting loaded native model state.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context
    ) as executor:

        future_map = {
            executor.submit(transcribe_chunk, chunk): chunk
            for chunk in chunks
        }

        completed = 0

        for future in as_completed(future_map):

            chunk = future_map[future]

            result = future.result()
            results.append(result)

            completed += 1

            print(
                f"\nCompleted chunk "
                f"{completed}/{len(chunks)} "
                f"(chunk {chunk['index'] + 1}) | "
                f"Words: {len(result['words'])}"
            )

    # Restore chronological order.
    results.sort(key=lambda item: item["index"])

    words = merge_results(results)
    segments = build_segments(words)

    elapsed = round(time.time() - start_time, 2)

    if not words:
        raise RuntimeError("No words detected.")

    final_data = {
        "video_id": video_id,
        "source_video": str(video_path),
        "audio_path": str(audio_path),
        "duration_seconds": duration,
        "language": results[0]["language"],
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "workers": MAX_WORKERS,
        "total_chunks": len(chunks),
        "total_words": len(words),
        "total_segments": len(segments),
        "processing_seconds": elapsed,
        "transcribed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "words": words,
        "segments": segments
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            final_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 55)
    print("PARALLEL TRANSCRIPTION COMPLETED")
    print("=" * 55)
    print(f"Chunks       : {len(chunks)}")
    print(f"Workers      : {MAX_WORKERS}")
    print(f"Total words  : {len(words)}")
    print(f"Total time   : {elapsed} seconds")
    print(f"Output       : {output_path}")
    print("=" * 55)

    return final_data


# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    videos = list(DOWNLOAD_DIR.glob("*.mp4"))

    if not videos:
        raise SystemExit("No downloaded MP4 videos found.")

    print("\nAvailable videos:\n")

    for index, video in enumerate(videos, 1):
        print(f"{index}. {video.name}")

    choice = input("\nSelect video (Enter = latest): ").strip()

    if choice:
        selected = videos[int(choice) - 1]
    else:
        selected = max(
            videos,
            key=lambda p: p.stat().st_mtime
        )

    transcribe_video(selected)

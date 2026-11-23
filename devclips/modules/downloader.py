
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_DIR / "storage" / "downloads"

# Supported YouTube domains
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtube-nocookie.com",
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def validate_youtube_url(url: str) -> None:
    """Validate that the supplied URL belongs to YouTube."""
    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with https:// or http://")

    hostname = (parsed.hostname or "").lower()

    if hostname not in YOUTUBE_DOMAINS:
        raise ValueError("Please provide a valid YouTube URL.")

    if hostname.endswith("youtu.be") and not parsed.path.strip("/"):
        raise ValueError("Invalid YouTube video URL.")


def format_duration(seconds: int | float | None) -> str:
    """Convert seconds into HH:MM:SS."""
    if seconds is None:
        return "Unknown"

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_downloaded_file(info: dict, video_id: str, ydl) -> Path:
    """Find the final video file after FFmpeg merging."""
    candidates = []

    # yt-dlp commonly provides the final filepath here.
    if info.get("filepath"):
        candidates.append(Path(info["filepath"]))

    # Check requested download paths.
    for item in info.get("requested_downloads") or []:
        if item.get("filepath"):
            candidates.append(Path(item["filepath"]))

    # Check yt-dlp's prepared filename.
    try:
        candidates.append(Path(ydl.prepare_filename(info)))
    except Exception:
        pass

    # Check candidate paths and their MP4 equivalents.
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate

        mp4_candidate = candidate.with_suffix(".mp4")
        if mp4_candidate.is_file():
            return mp4_candidate

    # Final fallback: search by YouTube video ID.
    matches = [
        path for path in DOWNLOAD_DIR.glob(f"{video_id}.*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if matches:
        return max(matches, key=lambda path: path.stat().st_size)

    raise FileNotFoundError(
        "Download completed, but the final video file could not be located."
    )


def download_youtube_video(
    url: str,
    max_duration_minutes: int = 180,
) -> dict:
    """
    Download one YouTube video and save its metadata.

    Returns a dictionary containing the downloaded file path
    and video information.
    """
    url = url.strip()
    validate_youtube_url(url)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    max_duration_seconds = max_duration_minutes * 60

    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "quiet": False,
        "no_warnings": False,
        "overwrites": False,
    }

    print("\nFetching YouTube video information...")

    with yt_dlp.YoutubeDL(options) as ydl:
        # First fetch metadata without downloading.
        info = ydl.extract_info(url, download=False)

        if not info:
            raise RuntimeError("Could not retrieve video information.")

        duration = info.get("duration")

        if duration is None:
            raise ValueError("Video duration could not be determined.")

        if duration > max_duration_seconds:
            raise ValueError(
                f"Video is too long ({format_duration(duration)}). "
                f"Maximum allowed duration is "
                f"{max_duration_minutes} minutes."
            )

        video_id = info.get("id")

        if not video_id:
            raise RuntimeError("YouTube video ID could not be determined.")

        print(f"\nTitle: {info.get('title', 'Unknown')}")
        print(f"Duration: {format_duration(duration)}")
        print(f"Starting download...\n")

        # Download using the already fetched information.
        downloaded_info = ydl.process_ie_result(info, download=True)

        video_path = find_downloaded_file(
            downloaded_info,
            video_id,
            ydl,
        )

    file_size = video_path.stat().st_size

    metadata = {
        "video_id": video_id,
        "title": downloaded_info.get("title"),
        "source_url": url,
        "duration_seconds": duration,
        "duration_formatted": format_duration(duration),
        "width": downloaded_info.get("width"),
        "height": downloaded_info.get("height"),
        "resolution": (
            f"{downloaded_info.get('width')}x{downloaded_info.get('height')}"
            if downloaded_info.get("width")
            and downloaded_info.get("height")
            else "Unknown"
        ),
        "extension": video_path.suffix.lstrip("."),
        "filesize_bytes": file_size,
        "filesize_mb": round(file_size / (1024 * 1024), 2),
        "file_path": str(video_path.resolve()),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save metadata beside the downloaded video.
    metadata_path = DOWNLOAD_DIR / f"{video_id}.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("DOWNLOAD COMPLETED")
    print("=" * 55)
    print(f"Title      : {metadata['title']}")
    print(f"Duration   : {metadata['duration_formatted']}")
    print(f"Resolution : {metadata['resolution']}")
    print(f"Size       : {metadata['filesize_mb']} MB")
    print(f"Video      : {metadata['file_path']}")
    print(f"Metadata   : {metadata_path}")
    print("=" * 55)

    return metadata


if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()

    try:
        result = download_youtube_video(video_url)
        print("\nDownloader returned successfully.")

    except Exception as error:
        print(f"\nDOWNLOAD ERROR: {error}")

#!/usr/bin/env python3
"""Download EchoSync neural model weights into the local models/ layout.

Idempotent: existing non-empty files are skipped. Any partial/failed download
raises so Docker builds abort instead of shipping a broken model cache.

Sources:
    VAD      : snakers4/silero-vad          (silero_vad.onnx, ~2.3 MB)
    Whisper  : Systran/faster-whisper-base.en via faster-whisper HF cache
    TTS      : thewh1teagle/kokoro-onnx GitHub release `model-files`
               (kokoro-v0_19.onnx, voices.bin)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# (url, destination Path, minimum acceptable bytes)
VAD_MODEL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    MODELS_DIR / "vad" / "silero_vad.onnx",
    1_000_000,
)

KOKORO_MODEL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
    MODELS_DIR / "tts" / "kokoro-v0_19.onnx",
    100_000_000,
)

VOICES_BIN = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin",
    MODELS_DIR / "tts" / "voices.bin",
    1_000_000,
)

MODEL_TARGETS = [VAD_MODEL, KOKORO_MODEL, VOICES_BIN]

HEADERS = {"User-Agent": "Mozilla/5.0 (EchoSync model fetcher)"}


def download(url: str, dest: Path, min_bytes: int) -> None:
    """Download a single model file, skipping an existing valid copy."""
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers=HEADERS)

    print(f"[get ] {url}")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)

    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded {url} too small ({size} bytes) — aborting")

    tmp.replace(dest)
    print(f"[done] {dest.name} ({size} bytes)")


def warm_whisper() -> None:
    """Prime the CTranslate2 faster-whisper model cache under models/stt."""
    cache_file = MODELS_DIR / "stt" / "CACHEDIR.TAG"
    if (MODELS_DIR / "stt" / "models--Systran--faster-whisper-base.en").is_dir():
        print("[skip] faster-whisper base.en already cached under models/stt")
        return

    print("[get ] faster-whisper base.en (HuggingFace cache)")
    from faster_whisper import WhisperModel

    WhisperModel(
        model_size_or_path="base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS_DIR / "stt"),
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("Signature: 8a477f597d28d172789f06886806bc550635d4d2\n")
    print("[done] faster-whisper base.en cached")


def main() -> int:
    for url, dest, min_bytes in MODEL_TARGETS:
        download(url, dest, min_bytes)
    warm_whisper()
    print("Model cache ready under", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download EchoSync neural model weights into the local models/ layout.

Idempotent: existing non-empty files are skipped. Any partial/failed download
raises so Docker builds abort instead of shipping a broken model cache.

Sources:
    VAD      : snakers4/silero-vad          (silero_vad.onnx, ~2.3 MB)
    Whisper  : Systran/faster-whisper-{WHISPER_MODEL_NAME} via faster-whisper HF cache
               (default medium.en; override via .env for smaller/faster models)
    TTS      : thewh1teagle/kokoro-onnx GitHub release `model-files`
               (kokoro-v0_19.onnx, voices.bin)
"""

from __future__ import annotations

import os
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


def _whisper_model_name() -> str:
    """Resolve WHISPER_MODEL_NAME from env, then .env, then base.en fallback.

    The shell environment often lacks the .env vars (which pydantic-settings
    loads at app runtime), so read the project .env file directly when the
    variable is not exported.
    """
    env_val = os.environ.get("WHISPER_MODEL_NAME")
    if env_val:
        return env_val
    env_file = MODELS_DIR.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("WHISPER_MODEL_NAME="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or "base.en"
    return "base.en"


def warm_whisper() -> None:
    """Prime the CTranslate2 faster-whisper model cache under models/stt."""
    # Respect the model configured in .env (WHISPER_MODEL_NAME) so the cache
    # always mirrors what the runtime loads. Falls back to base.en.
    model_name = _whisper_model_name()
    cache_name = f"models--Systran--faster-whisper-{model_name}"
    cache_file = MODELS_DIR / "stt" / "CACHEDIR.TAG"
    if (MODELS_DIR / "stt" / cache_name).is_dir():
        print(f"[skip] faster-whisper {model_name} already cached under models/stt")
        return

    print(f"[get ] faster-whisper {model_name} (HuggingFace cache)")
    from faster_whisper import WhisperModel

    WhisperModel(
        model_size_or_path=model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS_DIR / "stt"),
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("Signature: 8a477f597d28d172789f06886806bc550635d4d2\n")
    print(f"[done] faster-whisper {model_name} cached")


def _tts_engine() -> str:
    """Resolve TTS_ENGINE from env, then .env, then kokoro fallback."""
    env_val = os.environ.get("TTS_ENGINE")
    if env_val:
        return env_val.lower().strip()
    env_file = MODELS_DIR.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("TTS_ENGINE="):
                return line.split("=", 1)[1].strip().strip('"').strip("'").lower() or "kokoro"
    return "kokoro"


def main() -> int:
    tts = _tts_engine()
    download(VAD_MODEL[0], VAD_MODEL[1], VAD_MODEL[2])
    if tts != "edge":
        download(KOKORO_MODEL[0], KOKORO_MODEL[1], KOKORO_MODEL[2])
        download(VOICES_BIN[0], VOICES_BIN[1], VOICES_BIN[2])
    else:
        print("[skip] TTS_ENGINE=edge active; skipping Kokoro-82M download")
    warm_whisper()
    print("Model cache ready under", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

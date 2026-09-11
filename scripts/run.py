#!/usr/bin/env python3
"""Unified launcher for EchoSync AI.

Verifies prerequisites, environment keys, and local model weights,
then boots the FastAPI server (serving backend and frontend) and opens
the client workbench in the default browser.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure child processes and uvicorn reloader have ROOT_DIR on PYTHONPATH
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    f"{ROOT_DIR}:{existing_pythonpath}" if existing_pythonpath else str(ROOT_DIR)
)


def print_step(msg: str) -> None:
    print(f"\033[1;34m[EchoSync]\033[0m {msg}")


def print_error(msg: str) -> None:
    print(f"\033[1;31m[EchoSync Error]\033[0m {msg}", file=sys.stderr)


def check_environment() -> bool:
    """Verify .env file and GEMINI_API_KEY exist."""
    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        example_env = ROOT_DIR / ".env.example"
        if example_env.is_file():
            print_step("Creating .env from .env.example...")
            env_file.write_text(example_env.read_text())
        else:
            print_error("Missing .env file. Create .env with GEMINI_API_KEY.")
            return False

    # Read .env to verify key presence
    content = env_file.read_text()
    has_key = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val and val != "your_gemini_api_key_here":
                has_key = True
            break

    if not has_key and not os.environ.get("GEMINI_API_KEY"):
        print_error("GEMINI_API_KEY is unset or placeholder in .env.")
        print_error("Obtain key from https://aistudio.google.com/ and set in .env")
        return False

    return True


def check_and_download_models() -> bool:
    """Ensure neural model weights exist locally. If missing, invoke download_models.py."""
    vad_model = ROOT_DIR / "models" / "vad" / "silero_vad.onnx"
    stt_dir = ROOT_DIR / "models" / "stt"
    tts_model = ROOT_DIR / "models" / "tts" / "kokoro-v0_19.onnx"
    tts_voices = ROOT_DIR / "models" / "tts" / "voices.bin"

    models_ready = (
        vad_model.is_file() and stt_dir.is_dir() and tts_model.is_file() and tts_voices.is_file()
    )

    if models_ready:
        print_step("Local neural model cache verified.")
        return True

    print_step("Model weights missing. Downloading weights via scripts/download_models.py...")
    download_script = ROOT_DIR / "scripts" / "download_models.py"
    if not download_script.is_file():
        print_error(f"Missing downloader script at {download_script}")
        return False

    ret = subprocess.call([sys.executable, str(download_script)], cwd=str(ROOT_DIR))
    return ret == 0


def wait_for_server(url: str, timeout_sec: float = 40.0) -> bool:
    """Poll healthz endpoint until server is ready."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(0.4)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoSync AI unified launcher")
    parser.add_argument(
        "--host", default=None, help="Host binding (default: from config or 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port binding (default: from config or 8000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Skip opening browser automatically"
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    args = parser.parse_args()

    print_step("Starting EchoSync Voice Agent launcher...")

    if not check_environment():
        return 1

    if not check_and_download_models():
        print_error("Failed to verify or download models.")
        return 1

    # Resolve port and host
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("PORT", "8000"))

    # Spawn browser opener daemon thread if requested
    if not args.no_browser:
        import contextlib
        import threading

        target_url = f"http://localhost:{port}"

        def open_browser() -> None:
            health_url = f"http://127.0.0.1:{port}/healthz"
            if wait_for_server(health_url, timeout_sec=40.0):
                print_step(f"Server ready. Opening {target_url} in browser...")
                with contextlib.suppress(OSError):
                    webbrowser.open(target_url)

        t = threading.Thread(target=open_browser, daemon=True)
        t.start()

    print_step(f"Booting backend & frontend on http://{host}:{port}...")

    # Launch uvicorn via python API
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=os.environ.get("LOG_LEVEL", "INFO").lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

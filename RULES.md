# RULES.md — Automated Agent Operational Guidelines

## EchoSync AI Development Standard & System Constraints

---

## 1. Core Operating Principles

1. **Step-by-Step Execution Protocol:**
* Never execute multiple broad milestones in a single turn.
* Work strictly unit-by-unit (e.g., config -> VAD -> STT -> LLM -> TTS -> API -> Docker).
* After implementing a specific file, module, or test suite, stop and request user confirmation before moving to the next component.


2. **Editor & Tooling Constraints:**
* When referencing or instructing file edits via CLI, strictly use `code <filepath>`. Never suggest or use `nano`, `vim`, or `micro`.
* The Python package and environment manager for this project is strictly `uv`. Do not invoke raw `pip`, `poetry`, or `conda`.
* Host platform target: Arch Linux / CachyOS. All system command recommendations must remain compatible with this environment.
* Strict typography standard: Do not insert emojis anywhere in source code, commit messages, documentation, or terminal summaries.


3. **No Unrequested Scaffolding:**
* Do not generate placeholder "TODO" files or hallucinate mock dependencies unless explicitly required by the active task.
* Keep modules tightly scoped to their single responsibility.



---

## 2. Python & Architecture Standards

1. **Python Version & Typing:**
* Python 3.11+ target.
* Strict static type typing on all public functions, classes, and coroutines (`typing` / `collections.abc`).
* No untyped `**kwargs` or generic `dict` returns where a defined Pydantic schema or TypedDict can be used.


2. **Asyncio & Event Loop Hygiene:**
* Never run blocking synchronous CPU or I/O work directly inside FastAPI router handlers or async worker coroutines.
* Offload CTranslate2 (`faster-whisper`), ONNX Runtime (`silero-vad`, `kokoro`), and filesystem operations to worker threads via `asyncio.to_thread` or an explicit `ThreadPoolExecutor`.
* Enforce bounded queues: all inbound audio buffers must use `asyncio.Queue(maxsize=...)` to enforce backpressure and prevent unbounded memory growth.


3. **Dependency Injection & Configuration:**
* All configurations must be sourced via Pydantic v2 `BaseSettings` (`src/config.py`).
* Hardcoding model names, sample rates, chunk sizes, ports, or API keys directly in source files is strictly prohibited.
* Model weights and checkpoints must resolve to a configurable local cache path (`/app/models/` or local `.cache/`) rather than fetching dynamically at startup.



---

## 3. Audio & ML Pipeline Constraints

1. **Audio Data Formats:**
* Input format: Single-channel (mono), 16 kHz sample rate, 16-bit signed Linear PCM (`int16`, little-endian).
* Frame size: Inbound chunks must match 512 samples ($32\text{ ms}$) per frame for direct compatibility with Silero-VAD ONNX tensors.
* Synthesis output: 24 kHz mono PCM or streaming chunk frames.


2. **Quantization & Resource Ceilings:**
* STT inference (`faster-whisper`) must explicitly specify `device="cpu"` and `compute_type="int8"`.
* Thread pooling for CTranslate2 must be explicitly pinned to avoid CPU starvation against the main ASGI event loop.


3. **Resilience & Fault Tolerance:**
* External LLM calls (Google AI Studio Gemini API) must be wrapped using `tenacity.AsyncRetrying` with exponential backoff and jitter, handling HTTP 429 and network disconnects gracefully.
* WebSockets must handle client disconnects cleanly: cancel pending async tasks, flush in-memory frame queues, and invoke garbage collection hooks.



---

## 4. Testing, Code Quality & Verification Gates

1. **Static Analysis:**
* Code must pass `ruff check .` and `ruff format --check .` without errors.
* Imports must follow standard isort grouping enforced by Ruff.


2. **Automated Unit & Integration Tests:**
* Every core module (`vad.py`, `stt.py`, `llm.py`, `tts.py`) must have a corresponding test suite in `tests/`.
* Use `pytest` and `pytest-asyncio` for asynchronous test routines.
* Network-dependent services (e.g., Gemini API) must be mockable in unit test suites to allow offline CI test runs.


3. **Step Completion Requirement:**
* A step is only considered complete once the code is written, linted, and verified with a passing test or reproducible terminal command.
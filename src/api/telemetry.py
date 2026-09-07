"""Structured JSON logging and telemetry instrumentation for EchoSync pipeline."""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import psutil

# Initialize root logger with JSON stream formatting
logger = logging.getLogger("echosync.pipeline")


@dataclass(slots=True)
class PipelineMetrics:
    """Quantitative timing metrics for a completed voice dialogue turn."""

    vad_silence_ms: float = 0.0
    stt_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    total_rtt_ms: float = 0.0


@dataclass(slots=True)
class HostMetrics:
    """Host process compute and memory resource metrics."""

    memory_rss_mb: float
    cpu_percent: float


def get_host_metrics() -> HostMetrics:
    """Query current process RSS memory in megabytes and CPU utilization percent."""
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    cpu_pct = process.cpu_percent(interval=None)
    return HostMetrics(memory_rss_mb=round(rss_mb, 2), cpu_percent=round(cpu_pct, 2))


def log_turn_telemetry(
    session_id: str,
    turn_id: str,
    metrics: PipelineMetrics,
    transcript: str,
    level: str = "INFO",
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit RFC 8259 compliant structured single-line JSON log entry to stdout."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "logger": "echosync.pipeline",
        "session_id": session_id,
        "turn_id": turn_id,
        "event": "pipeline_turn_complete",
        "metrics": asdict(metrics),
        "transcript": transcript,
        "host_metrics": asdict(get_host_metrics()),
    }
    if extra:
        payload["extra"] = extra

    log_line = json.dumps(payload)
    sys.stdout.write(log_line + "\n")
    sys.stdout.flush()

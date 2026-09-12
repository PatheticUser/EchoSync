"""Prometheus collectors for EchoSync operational dashboards.

Process-wide counters, gauges, and histograms gathered from the HTTP request
layer, the WebSocket audio gateway, and per-turn pipeline telemetry. The
collectors register with the default ``prometheus_client`` registry at import
time and are scraped by the ``/metrics`` route that
``prometheus-fastapi-instrumentator`` mounts in :func:`src.main.create_app`.

Metric names use the ``echosync_`` namespace. HTTP request totals, durations,
and per-status error counts are provided by the instrumentator itself.
Latency percentiles (e.g. RTT p95) are derived in PromQL from the histograms
via ``histogram_quantile(0.95, rate(<name>_bucket[5m]))``.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- WebSocket gateway ------------------------------------------------------

#: Currently open WebSocket audio sessions. Raised on ``accept`` and lowered in
#: the endpoint's ``finally`` block, so "closed" derives as total - active.
active_ws_connections = Gauge(
    "echosync_ws_connections_active",
    "Currently open WebSocket audio sessions.",
)

#: Total accepted WebSocket connections since process start.
ws_connections_total = Counter(
    "echosync_ws_connections_total",
    "Total accepted WebSocket audio connections.",
)

# --- Per-turn pipeline telemetry --------------------------------------------

#: Completed voice dialogue turns since process start (turns/min = rate over time).
turns_total = Counter(
    "echosync_turns_total",
    "Completed voice dialogue turns.",
)

#: Per-turn round-trip latency. Telemetry records milliseconds, converted here
#: to Prometheus base units (seconds). Default buckets suit voice-agent RTTs
#: from sub-second STT/TTS stages up to multi-second conversational turns.
turn_rtt_seconds = Histogram(
    "echosync_turn_rtt_seconds",
    "Round-trip time of a completed dialogue turn, in seconds.",
)

stt_seconds = Histogram(
    "echosync_stt_seconds",
    "Speech-to-text duration per completed turn, in seconds.",
)

llm_ttft_seconds = Histogram(
    "echosync_llm_ttft_seconds",
    "LLM time-to-first-token per completed turn, in seconds.",
)

tts_first_chunk_seconds = Histogram(
    "echosync_tts_first_chunk_seconds",
    "TTS time-to-first-audio-chunk per completed turn, in seconds.",
)


def record_turn_metrics(
    stt_ms: float,
    llm_ttft_ms: float,
    tts_first_chunk_ms: float,
    total_rtt_ms: float,
) -> None:
    """Observe one completed turn into the turn counter and latency histograms.

    Millisecond timings from :class:`src.api.telemetry.PipelineMetrics` are
    converted to seconds before being recorded.
    """
    turns_total.inc()
    turn_rtt_seconds.observe(total_rtt_ms / 1000.0)
    stt_seconds.observe(stt_ms / 1000.0)
    llm_ttft_seconds.observe(llm_ttft_ms / 1000.0)
    tts_first_chunk_seconds.observe(tts_first_chunk_ms / 1000.0)

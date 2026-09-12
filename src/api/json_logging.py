"""Single-line JSON logging for EchoSync app loggers.

Every ``echosync.*`` logger emits RFC 8259 JSON on one physical line so
production log drains (Axiom / Better Stack) see a single schema::

    {"timestamp": "2026-09-12T09:30:00.123456+00:00", "level": "INFO",
     "logger": "echosync.main", "message": "..."}

Caller-supplied ``extra={...}`` kwargs become top-level JSON fields.

Scope deliberately leaves two streams untouched:

- Uvicorn's own loggers (``uvicorn``, ``uvicorn.access``, ``uvicorn.error``)
  keep uvicorn's default text configuration, driven by ``LOG_LEVEL`` as before.
- The pipeline metrics event stream (``log_turn_telemetry``) writes directly to
  stdout with its own stable payload keys; it is not routed through this
  formatter.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# LogRecord bookkeeping attributes: never surfaced as JSON payload fields.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JSONFormatter(logging.Formatter):
    """Serialize a :class:`logging.LogRecord` as a single-line JSON object.

    Canonical schema: ``{"timestamp", "level", "logger", "message"}`` plus any
    caller-supplied ``extra`` record attributes as additional fields.
    Timestamps are ISO-8601 UTC (aware, ``+00:00``), matching the timestamp
    format already emitted by the pipeline telemetry stream.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {}

        # Caller-supplied extras (record.__dict__) become JSON fields; private
        # and stdlib bookkeeping attributes are excluded.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        # Canonical schema keys always win over extras.
        payload["timestamp"] = datetime.fromtimestamp(record.created, UTC).isoformat()
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["message"] = record.getMessage()

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # default=str tolerates non-serializable extras; newlines remain escaped
        # so every entry stays one physical line.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_app_logging(level: str | int = "INFO") -> None:
    """Attach the JSON formatter to the ``echosync`` logger (idempotent).

    Configuration targets the parent ``echosync`` logger so every child app
    logger (``echosync.main``, ``echosync.router``, ``echosync.pipeline``)
    inherits the handler. ``propagate=False`` guarantees a single JSON line per
    record and keeps uvicorn's root-level text handler (installed by uvicorn's
    default ``dictConfig``) from double-emitting app records. Uvicorn's own
    loggers, the root logger, and ``log_turn_telemetry`` are left untouched.
    """
    app_logger = logging.getLogger("echosync")
    app_logger.setLevel(level)
    app_logger.propagate = False

    has_json_handler = any(
        isinstance(handler, logging.StreamHandler) and isinstance(handler.formatter, JSONFormatter)
        for handler in app_logger.handlers
    )
    if not has_json_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        app_logger.addHandler(handler)

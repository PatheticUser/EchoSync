"""Tests for the T3.1 unified single-line JSON logging formatter."""

import json
import logging

from src.api.json_logging import JSONFormatter, configure_app_logging


def _record(
    name: str, level: int = logging.INFO, msg: str = "", args: tuple = ()
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_json_formatter_emits_canonical_schema() -> None:
    line = JSONFormatter().format(_record("echosync.router", msg="Audio queue full"))
    payload = json.loads(line)
    assert set(payload) >= {"timestamp", "level", "logger", "message"}
    assert payload["level"] == "INFO"
    assert payload["logger"] == "echosync.router"
    assert payload["message"] == "Audio queue full"
    assert "\n" not in line  # single physical line per record


def test_json_formatter_extra_kwargs_become_fields() -> None:
    record = _record("echosync.pipeline", level=logging.WARNING, msg="Turn aborted")
    record.session_id = "ses_123"
    record.queue_size = 5
    payload = json.loads(JSONFormatter().format(record))
    assert payload["session_id"] == "ses_123"
    assert payload["queue_size"] == 5


def test_json_formatter_message_args_and_isoutc_timestamp() -> None:
    payload = json.loads(
        JSONFormatter().format(_record("echosync.main", msg="Frame #%d ready", args=(42,)))
    )
    assert payload["message"] == "Frame #42 ready"
    assert payload["timestamp"].endswith("+00:00")


def test_configure_app_logging_is_idempotent() -> None:
    app_logger = logging.getLogger("echosync")
    app_logger.handlers.clear()  # deterministic slate for this assertion
    configure_app_logging()
    configure_app_logging()
    json_handlers = [
        handler
        for handler in app_logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and isinstance(handler.formatter, JSONFormatter)
    ]
    assert len(json_handlers) == 1
    assert app_logger.propagate is False

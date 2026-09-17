"""Unit tests for built-in tool execution and live context formatting."""

import pytest

from src.core.tools import (
    calculate,
    execute_tool,
    format_live_context,
    get_current_time,
    get_system_metrics,
)


def test_get_current_time() -> None:
    """Verify time string format contains weekday, month, and time."""
    time_str = get_current_time()
    assert isinstance(time_str, str)
    assert len(time_str) > 10
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert any(day in time_str for day in weekdays)


def test_get_system_metrics() -> None:
    """Verify system telemetry dictionary fields."""
    metrics = get_system_metrics()
    assert isinstance(metrics, dict)
    assert "os" in metrics
    assert "cpu_percent" in metrics
    assert "memory_rss_mb" in metrics
    assert metrics["memory_rss_mb"] > 0


def test_calculate_basic_arithmetic() -> None:
    """Verify safe mathematical evaluation."""
    assert calculate("2 + 2") == "4"
    assert calculate("10 * 5 - 4") == "46"
    assert calculate("100 / 4") == "25"
    assert calculate("2 ** 8") == "256"
    assert calculate("sqrt(144)") == "12"


def test_calculate_rejects_dangerous_expressions() -> None:
    """Verify code injection and arbitrary execution attempts fail safely."""
    assert "Error" in calculate("__import__('os').system('ls')")
    assert "Error" in calculate("open('/etc/passwd')")


def test_execute_tool() -> None:
    """Verify tool execution via registry."""
    res = execute_tool("get_current_time")
    assert isinstance(res, str)

    with pytest.raises(KeyError):
        execute_tool("non_existent_tool")


def test_format_live_context() -> None:
    """Verify live context string contains time and host information."""
    ctx = format_live_context()
    assert "[Live Context]" in ctx
    assert "Current Time:" in ctx
    assert "Host:" in ctx

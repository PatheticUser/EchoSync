"""Built-in tool execution and real-time context injection for EchoSync."""

import ast
import datetime
import math
import operator
import platform
from typing import Any

import psutil

# Allowed operators for safe mathematical evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def get_current_time() -> str:
    """Return the current local date, weekday, and 12-hour formatted time."""
    now = datetime.datetime.now(datetime.UTC).astimezone()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def get_system_metrics() -> dict[str, Any]:
    """Return host system telemetry including CPU percent, memory RSS, and OS."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=None)
    now = datetime.datetime.now(datetime.UTC).astimezone()
    return {
        "os": platform.system(),
        "release": platform.release(),
        "cpu_percent": cpu_pct,
        "memory_rss_mb": round(mem_mb, 1),
        "host_uptime_s": round(now.timestamp() - psutil.boot_time(), 0),
    }


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression string without using eval()."""

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return SAFE_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            return SAFE_OPERATORS[op_type](_eval_node(node.operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("Only simple function calls are allowed")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")
            args = [_eval_node(arg) for arg in node.args]
            func = SAFE_FUNCTIONS[func_name]
            return func(*args) if callable(func) else func
        elif isinstance(node, ast.Name):
            if node.id in SAFE_FUNCTIONS:
                return SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Undefined identifier: {node.id}")
        else:
            raise TypeError(f"Unsupported expression element: {type(node)}")

    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


# Exposed tool registry mapping function names to callable implementations
VOICE_TOOLS = {
    "get_current_time": get_current_time,
    "get_system_metrics": get_system_metrics,
    "calculate": calculate,
}


def execute_tool(tool_name: str, **kwargs: Any) -> Any:
    """Execute a registered tool by name with arguments."""
    if tool_name not in VOICE_TOOLS:
        raise KeyError(f"Tool {tool_name} not found in VOICE_TOOLS registry")
    return VOICE_TOOLS[tool_name](**kwargs)


def format_live_context() -> str:
    """Generate concise live contextual grounding for system prompts.

    Injected into every LLM turn with 0ms network latency overhead so the model
    can immediately and accurately answer time, date, and system state queries.
    """
    time_str = get_current_time()
    try:
        stats = get_system_metrics()
        stats_str = f"CPU: {stats['cpu_percent']}%, Memory: {stats['memory_rss_mb']} MB"
    except Exception:  # noqa: BLE001
        stats_str = "Status: operational"

    return f"[Live Context]\nCurrent Time: {time_str}\nHost: {stats_str}\n"

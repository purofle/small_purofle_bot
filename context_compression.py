"""Context budgeting and loss-minimizing history compression.

The OpenAI client does not expose a portable tokenizer for every configured
model.  The estimator below is intentionally conservative and is used only
to decide when history needs to be compacted.  The API remains the authority
on the exact token count.
"""

from __future__ import annotations

import json
from typing import Any


DEFAULT_CONTEXT_TOKEN_LIMIT = 256_000
DEFAULT_RESPONSE_TOKEN_RESERVE = 16_000
MESSAGE_OVERHEAD_TOKENS = 12


def estimate_tokens(value: Any) -> int:
    """Estimate tokens conservatively for strings and OpenAI message values."""
    if value is None:
        return 0
    if isinstance(value, str):
        ascii_chars = sum(ord(char) < 128 for char in value)
        other_chars = len(value) - ascii_chars
        # CJK and other non-ASCII text commonly use about one token per char.
        return max(1, (ascii_chars + 3) // 4 + other_chars)
    if isinstance(value, (int, float, bool)):
        return 1
    if isinstance(value, list):
        return sum(estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return estimate_tokens(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return estimate_tokens(str(value))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    return MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _tool_chain_starts_at(history: list[dict[str, Any]], index: int) -> bool:
    """Avoid retaining a tool result without its assistant tool request."""
    return history[index].get("role") == "tool"


def _truncate_text_to_tokens(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text
    if token_budget <= 8:
        return "[truncated]"

    suffix = "\n[content truncated]"
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle] + suffix) <= token_budget:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def _truncate_message(message: dict[str, Any], token_budget: int) -> dict[str, Any]:
    """Create a compact representation when one history item exceeds budget."""
    content_budget = max(8, token_budget - MESSAGE_OVERHEAD_TOKENS - 8)
    truncated = {
        "role": message.get("role", "unknown"),
        "content": _truncate_text_to_tokens(_message_text(message), content_budget),
    }
    if message.get("role") == "tool" and message.get("tool_call_id"):
        truncated["tool_call_id"] = message["tool_call_id"]
    return truncated


def compress_history(
    history: list[dict[str, Any]],
    *,
    available_tokens: int,
    summary_label: str = "Earlier conversation summary",
) -> tuple[list[dict[str, Any]], bool]:
    """Fit history into a token budget, returning (history, was_compressed).

    Recent messages are preferred. Older messages are represented by a
    bounded transcript summary so user preferences and tool outcomes remain
    visible instead of being silently discarded.
    """
    if available_tokens <= 0:
        return [], bool(history)

    total = sum(estimate_message_tokens(message) for message in history)
    if total <= available_tokens:
        return history, False

    summary_budget = max(64, min(available_tokens // 3, 8_192))
    recent_budget = max(0, available_tokens - summary_budget - MESSAGE_OVERHEAD_TOKENS)
    recent: list[dict[str, Any]] = []
    used = 0
    for message in reversed(history):
        cost = estimate_message_tokens(message)
        if recent and used + cost > recent_budget:
            break
        if not recent and cost > recent_budget:
            recent.append(_truncate_message(message, recent_budget))
            break
        recent.append(message)
        used += cost
    recent.reverse()

    boundary = len(history) - len(recent)
    # A tool result is only valid together with the preceding assistant tool
    # call. Keep the complete chain if it fits; otherwise omit its results.
    if boundary < len(history) and _tool_chain_starts_at(history, boundary):
        assistant_boundary = boundary - 1
        expanded = history[assistant_boundary:]
        expanded_cost = sum(estimate_message_tokens(message) for message in expanded)
        if assistant_boundary >= 0 and expanded_cost <= recent_budget:
            boundary = assistant_boundary
        else:
            while boundary < len(history) and _tool_chain_starts_at(history, boundary):
                boundary += 1
    recent = history[boundary:]
    older = history[:boundary]
    if not older:
        return recent, True

    recent_tokens = sum(estimate_message_tokens(message) for message in recent)
    summary_content_budget = min(summary_budget, available_tokens - recent_tokens - MESSAGE_OVERHEAD_TOKENS)
    if summary_content_budget <= estimate_tokens(summary_label) + 2:
        return recent, True

    lines: list[str] = []
    used_summary = estimate_tokens(summary_label)
    for message in older:
        line = f"{message.get('role', 'unknown')}: {_message_text(message)}"
        cost = estimate_tokens(line)
        if used_summary + cost > summary_content_budget:
            lines.append("[older context truncated]")
            break
        lines.append(line)
        used_summary += cost

    summary_content = f"{summary_label}:\n" + "\n".join(lines)
    summary = {
        "role": "system",
        "content": _truncate_text_to_tokens(summary_content, summary_content_budget),
    }
    return [summary, *recent], True

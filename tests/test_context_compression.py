import unittest

from context_compression import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    compress_history,
    estimate_message_tokens,
)


class ContextCompressionTests(unittest.TestCase):
    def test_default_limit_is_256k(self) -> None:
        self.assertEqual(DEFAULT_CONTEXT_TOKEN_LIMIT, 256_000)

    def test_short_history_is_unchanged(self) -> None:
        history = [{"role": "user", "content": "hello"}]
        compressed, changed = compress_history(history, available_tokens=100)
        self.assertFalse(changed)
        self.assertIs(compressed, history)

    def test_old_history_is_summarized_and_recent_history_retained(self) -> None:
        history = [
            {"role": "user", "content": "old " * 30},
            {"role": "assistant", "content": "old answer " * 30},
            {"role": "user", "content": "latest"},
        ]
        compressed, changed = compress_history(history, available_tokens=100)
        self.assertTrue(changed)
        self.assertEqual(compressed[-1], history[-1])
        self.assertEqual(compressed[0]["role"], "system")
        self.assertLessEqual(sum(estimate_message_tokens(item) for item in compressed), 100)

    def test_tool_result_keeps_assistant_tool_call(self) -> None:
        history = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "follow up"},
        ]
        compressed, changed = compress_history(history, available_tokens=45)
        self.assertTrue(changed)
        roles = [item["role"] for item in compressed]
        self.assertNotIn("tool", roles[:1])
        if "tool" in roles:
            tool_index = roles.index("tool")
            self.assertEqual(roles[tool_index - 1], "assistant")

    def test_oversized_tool_chain_is_not_split(self) -> None:
        history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "arguments": "x" * 500}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "final answer"},
        ]
        compressed, changed = compress_history(history, available_tokens=45)
        self.assertTrue(changed)
        roles = [item["role"] for item in compressed]
        self.assertNotIn("tool", roles)
        self.assertEqual(compressed[-1], history[-1])
        self.assertLessEqual(sum(estimate_message_tokens(item) for item in compressed), 45)


if __name__ == "__main__":
    unittest.main()

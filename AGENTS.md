# small purofle bot
When you change or add any new features, you MUST update this file.
## What is small purofle bot
This is a Telegram group chatbot that can help you with various tasks.
## Architecture
- `tools/`: includes all the tools that the bot can use to perform tasks
- `dynamic_tools/`: includes all the tools that can be dynamically loaded by the bot. Sometimes, the bot may write a tool by itself and load it dynamically.
- `tool_regsistry.py`: The following tool system will be introduced.
- `llm_core/`: LLM-related utilities.
  - `prompt.py`: layered system prompt builder. It separates the prompt into a **static prompt** (constant personality, chat style, safety, tool-writing guide and behavior rules) and a **dynamic prompt** (current environment, user info and available tools). The final system prompt is assembled by `Prompt.build(context)`.
- `tests/`: unit tests, runnable with `python -m unittest discover -s tests -v`.

### Dependency Security
Dependencies are managed with `uv`. The lock file must keep `aiohttp>=3.14.1`
and `idna>=3.15`; these minimum versions are enforced through uv constraints
in `pyproject.toml` to avoid known Dependabot vulnerabilities. After changing
dependencies, run `uv lock`, `uv sync`, and the unit test suite.

### Tool System
Each tool MUST have a `__tools_name__` and a `__tools_description__`. The bot will use the `tools_name` to call the tool and the `tools_description` to understand what the tool does.
the arguments of the tool MUST have a `__tools_parameters__` global variable, which is a dictionary that contains the parameters of the tool.
The bot will use the `tools_parameters` to understand what parameters the tool needs.
### Chat Context
Each user should have their own chat context. At the same time, some group messages will be also included in the chat context. Agents should decide whether
to include the group messages in the chat context or not.
### Memory

### Context Compression
The bot enforces a default input context budget of `256000` tokens in
`context_compression.py`. When the estimated system prompt, tools, current
message, and history exceed that budget (leaving room for the response), old
history is represented by a bounded summary and the newest messages are
retained. The limit and response reserve are configurable on `AIBot` via
`context_token_limit` and `response_token_reserve`, or at runtime with the
`CONTEXT_TOKEN_LIMIT` and `RESPONSE_TOKEN_RESERVE` environment variables.

import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import Bot
from aiogram.types import Message

ToolFn = Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self) -> None:
        # name -> {"type":"function","function":{...}}
        self.specs: Dict[str, dict] = {}
        # name -> async fn(ctx, **kwargs)
        self.fns: Dict[str, ToolFn] = {}

    @property
    def schemas(self) -> list[dict]:
        return list(self.specs.values())

    def register(self, name: str, description: str, parameters: dict, fn: ToolFn) -> None:
        self.specs[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self.fns[name] = fn


def _load_module_from_path(module_name: str, file: Path):
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tools(tools_dir: str | Path) -> ToolRegistry:
    tools_dir = Path(tools_dir)
    if not tools_dir.exists() or not tools_dir.is_dir():
        raise FileNotFoundError(f"tools_dir not found or not a directory: {tools_dir}")

    reg = ToolRegistry()

    for file in sorted(tools_dir.glob("*.py")):
        if file.name.startswith("_"):
            continue

        # 用唯一 module name，避免重复 import/冲突
        module_name = f"_dynamic_tools_{file.stem}"

        module = _load_module_from_path(module_name, file)

        name = getattr(module, "__tools_name__", None)
        desc = getattr(module, "__tools_description__", None)
        params = getattr(module, "__tools_parameters__", None)

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{file}: __tools_name__ must be non-empty str")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(f"{file}: __tools_description__ must be non-empty str")
        if not isinstance(params, dict):
            raise ValueError(f"{file}: __tools_parameters__ must be a dict (JSON Schema)")

        fn = getattr(module, name, None)
        if fn is None:
            raise ValueError(f"{file}: function `{name}` not found")
        if not inspect.iscoroutinefunction(fn):
            raise ValueError(f"{file}: function `{name}` must be async")

        # 轻度校验：至少要能接 ctx
        sig = inspect.signature(fn)
        if len(sig.parameters) < 1:
            raise ValueError(f"{file}: `{name}` must accept ctx as first argument")

        reg.register(name=name, description=desc, parameters=params, fn=fn)

    return reg


@dataclass
class ToolContext:
    bot: Bot
    message: Message
    superuser_id: int

    @property
    def is_superuser(self) -> bool:
        return self.message.from_user is not None and self.message.from_user.id == self.superuser_id


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.restricted_tools = {"run_shell_command", "read_file", "write_file"}

    async def execute(self, ctx: ToolContext, tool_name: str, tool_arguments_json: Optional[str]) -> Any:
        fn = self.registry.fns.get(tool_name)
        if fn is None:
            return {"error": "tool_not_found", "tool": tool_name}

        if tool_name in self.restricted_tools and not ctx.is_superuser:
            logging.warning(
                "Rejected restricted tool call: tool=%s user_id=%s",
                tool_name,
                getattr(ctx.message.from_user, "id", None),
            )
            return {"error": "permission_denied", "tool": tool_name}

        try:
            kwargs = json.loads(tool_arguments_json) if tool_arguments_json else {}
        except json.JSONDecodeError as e:
            logging.warning("Invalid tool arguments JSON for %s: %s", tool_name, e)
            return {"error": "invalid_tool_arguments_json", "tool": tool_name}

        if not isinstance(kwargs, dict):
            return {"error": "tool_arguments_must_be_object", "tool": tool_name}

        try:
            return await fn(ctx, **kwargs)
        except TypeError as e:
            logging.warning("Tool %s TypeError: %s", tool_name, e)
            return {"error": "tool_argument_error", "tool": tool_name, "detail": str(e)}
        except Exception as e:
            logging.exception("Tool %s failed", tool_name)
            return {"error": "tool_runtime_error", "tool": tool_name, "detail": str(e)}

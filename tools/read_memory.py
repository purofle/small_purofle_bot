from pathlib import Path

from pydantic import BaseModel, ConfigDict

__tools_name__ = "read_memory"
__tools_description__ = "Read bot memory from local memory.txt."

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_FILE = _PROJECT_ROOT / "memory.txt"
_MAX_RETURN_CHARS = 12000


class ReadMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


__tools_parameters__ = ReadMemoryArgs.model_json_schema()


async def read_memory(ctx, **kwargs):
    ReadMemoryArgs.model_validate(kwargs)

    if not _MEMORY_FILE.exists():
        return {"exists": False, "content": ""}

    content = _MEMORY_FILE.read_text(encoding="utf-8")
    if len(content) > _MAX_RETURN_CHARS:
        content = content[-_MAX_RETURN_CHARS:]

    return {"exists": True, "content": content}

from datetime import datetime, UTC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__tools_name__ = "write_memory"
__tools_description__ = "Write content to local memory.txt."

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_FILE = _PROJECT_ROOT / "memory.txt"


class WriteMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=8000, description="Memory content to write")
    mode: Literal["append", "overwrite"] = Field(
        "append",
        description="append adds one record, overwrite replaces whole file",
    )
    with_timestamp: bool = Field(
        True,
        description="Prefix record with UTC timestamp when mode is append",
    )


__tools_parameters__ = WriteMemoryArgs.model_json_schema()


async def write_memory(ctx, **kwargs):
    args = WriteMemoryArgs.model_validate(kwargs)

    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "overwrite":
        _MEMORY_FILE.write_text(args.content, encoding="utf-8")
    else:
        prefix = ""
        if args.with_timestamp:
            prefix = f"[{datetime.now(UTC).isoformat()}] "

        line = f"{prefix}{args.content}".strip()
        with _MEMORY_FILE.open("a", encoding="utf-8") as f:
            if _MEMORY_FILE.stat().st_size > 0:
                f.write("\n")
            f.write(line)

    return {"ok": True, "path": str(_MEMORY_FILE), "mode": args.mode}

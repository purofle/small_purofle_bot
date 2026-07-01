from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__tools_name__ = "read_file"
__tools_description__ = "Read a text file under project root."

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MAX_CHARS = 12000


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, max_length=800, description="Relative file path under project root")
    max_chars: int = Field(
        _DEFAULT_MAX_CHARS,
        ge=200,
        le=50000,
        description="Maximum characters returned",
    )


__tools_parameters__ = ReadFileArgs.model_json_schema()


def _resolve_path(rel_path: str) -> Path:
    target = (_PROJECT_ROOT / rel_path).resolve()
    if target != _PROJECT_ROOT and _PROJECT_ROOT not in target.parents:
        raise ValueError("path must stay inside project root")
    return target


async def read_file(ctx, **kwargs):
    args = ReadFileArgs.model_validate(kwargs)
    target = _resolve_path(args.path)

    if not target.exists():
        return {"exists": False, "path": str(target), "content": ""}
    if target.is_dir():
        return {"exists": True, "is_dir": True, "path": str(target), "content": ""}

    content = target.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(content) > args.max_chars:
        content = content[: args.max_chars]
        truncated = True

    return {
        "exists": True,
        "is_dir": False,
        "path": str(target),
        "content": content,
        "truncated": truncated,
    }

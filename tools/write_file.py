from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__tools_name__ = "write_file"
__tools_description__ = "Create or modify a text file under project root."

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, max_length=800, description="Relative file path under project root")
    content: str = Field(..., max_length=100000, description="Text content to write")
    mode: Literal["overwrite", "append"] = Field("overwrite", description="overwrite replaces file, append appends")
    create_parents: bool = Field(True, description="Create parent directories when missing")


__tools_parameters__ = WriteFileArgs.model_json_schema()


def _resolve_path(rel_path: str) -> Path:
    target = (_PROJECT_ROOT / rel_path).resolve()
    if target != _PROJECT_ROOT and _PROJECT_ROOT not in target.parents:
        raise ValueError("path must stay inside project root")
    return target


async def write_file(ctx, **kwargs):
    args = WriteFileArgs.model_validate(kwargs)
    target = _resolve_path(args.path)

    if args.create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    elif not target.parent.exists():
        raise ValueError("parent directory does not exist")

    if target.exists() and target.is_dir():
        raise ValueError("path points to a directory")

    if args.mode == "overwrite":
        target.write_text(args.content, encoding="utf-8")
    else:
        with target.open("a", encoding="utf-8") as f:
            f.write(args.content)

    return {
        "ok": True,
        "path": str(target),
        "mode": args.mode,
        "size_bytes": target.stat().st_size,
    }

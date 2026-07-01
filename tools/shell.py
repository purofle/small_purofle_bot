import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__tools_name__ = "run_shell_command"
__tools_description__ = "Run a shell command on the bot host and return stdout/stderr. Restricted to highest-privilege user only."

_MAX_OUTPUT_CHARS = 8000
_DEFAULT_TIMEOUT_SECONDS = 15
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ShellArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., min_length=1, max_length=2000, description="Shell command to execute")
    timeout_seconds: int = Field(
        _DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=60,
        description="Execution timeout in seconds",
    )
    workdir: str | None = Field(
        None,
        description="Optional working directory under project root",
    )


__tools_parameters__ = ShellArgs.model_json_schema()


def _resolve_workdir(workdir: str | None) -> Path:
    if not workdir:
        return _PROJECT_ROOT

    target = (_PROJECT_ROOT / workdir).resolve()
    if target != _PROJECT_ROOT and _PROJECT_ROOT not in target.parents:
        raise ValueError("workdir must stay inside project root")

    if not target.exists() or not target.is_dir():
        raise ValueError("workdir does not exist or is not a directory")

    return target


async def run_shell_command(ctx, **kwargs):
    args = ShellArgs.model_validate(kwargs)
    cwd = _resolve_workdir(args.workdir)

    proc = await asyncio.create_subprocess_shell(
        args.command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=args.timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout, stderr = await proc.communicate()

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if len(stdout_text) > _MAX_OUTPUT_CHARS:
        stdout_text = stdout_text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"
    if len(stderr_text) > _MAX_OUTPUT_CHARS:
        stderr_text = stderr_text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"

    return {
        "ok": proc.returncode == 0 and not timed_out,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "cwd": str(cwd),
    }

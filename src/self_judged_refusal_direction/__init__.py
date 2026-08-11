import os
import sysconfig
from pathlib import Path


def _configure_native_jit() -> None:
    include = sysconfig.get_path("include")
    if include is not None and not (Path(include) / "Python.h").is_file():
        os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


_configure_native_jit()


def main() -> None:
    from self_judged_refusal_direction.cli import main as cli_main

    cli_main()


__all__ = ["main"]

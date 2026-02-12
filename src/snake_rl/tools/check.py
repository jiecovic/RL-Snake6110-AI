# src/snake_rl/tools/check.py
from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    python = sys.executable
    _run([python, "-m", "ruff", "check", "src/snake_rl"])
    _run([python, "-m", "pyright"])


if __name__ == "__main__":
    main()

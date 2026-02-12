# src/snake_rl/rl/eval/table.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from snake_rl.rl.metrics import Metrics


def _fmt_float(value: float | None, *, width: int, prec: int = 3) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{float(value):>{width}.{prec}f}"


def _fmt_int(value: int | None, *, width: int) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{int(value):>{width}d}"


def _fmt_step(value: int | None, *, width: int) -> str:
    if value is None:
        return f"{'-':>{width}}"
    v = int(value)
    if v >= 1_000_000_000:
        s = f"{v / 1_000_000_000:.2f}B"
    elif v >= 1_000_000:
        s = f"{v / 1_000_000:.2f}M"
    elif v >= 1_000:
        s = f"{v / 1_000:.2f}k"
    else:
        s = str(v)
    return f"{s:>{width}}"


def _fmt_text(value: str | None, *, width: int) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{str(value):>{width}}"


_COLS: list[tuple[str, int]] = [
    ("step", 9),
    ("eps", 4),
    ("det", 3),
    ("mean_r", 9),
    ("std_r", 8),
    ("len", 7),
    ("mean_score", 10),
    ("win_rate", 8),
    ("wins", 5),
    ("BR", 2),
    ("BS", 2),
    ("BW", 2),
    ("time", 25),
]


@dataclass(slots=True)
class EvalTablePrinter:
    header_every: int = 20
    header_every_s: float = 300.0
    _rows: int = 0
    _last_header_t: float = field(default_factory=perf_counter)

    def _header_cells(self) -> list[str]:
        return [f"{name:>{width}}" for name, width in _COLS]

    def _rule(self) -> str:
        parts = ["-" * width for _name, width in _COLS]
        return "+-" + "-+-".join(parts) + "-+"

    def header(self) -> str:
        cells = self._header_cells()
        return "| " + " | ".join(cells) + " |"

    def _should_header(self) -> bool:
        if self._rows == 0:
            return True
        if int(self.header_every) > 0 and (self._rows % int(self.header_every)) == 0:
            return True
        now = perf_counter()
        return float(self.header_every_s) > 0 and (now - float(self._last_header_t)) >= float(
            self.header_every_s
        )

    def _row_cells(
        self,
        *,
        metrics: dict[str, Any],
        timesteps: int,
        episodes: int,
        deterministic: bool,
        best_flags: dict[str, bool] | None = None,
    ) -> list[str]:
        mean_r = metrics.get(Metrics.EP_RETURN_MEAN)
        std_r = metrics.get(Metrics.EP_RETURN_STD)
        mean_len = metrics.get(Metrics.EP_LENGTH_MEAN)
        mean_score = metrics.get(Metrics.EP_SCORE_MEAN)
        win_rate = metrics.get(Metrics.EP_WIN_RATE)
        wins = metrics.get(Metrics.EP_WINS)
        flags = dict(best_flags or {})

        def _flag(key: str) -> str:
            return "*" if bool(flags.get(key, False)) else ""

        det = "Y" if deterministic else "N"
        return [
            _fmt_step(int(timesteps), width=9),
            _fmt_int(int(episodes), width=4),
            _fmt_text(det, width=3),
            _fmt_float(None if mean_r is None else float(mean_r), width=9, prec=3),
            _fmt_float(None if std_r is None else float(std_r), width=8, prec=3),
            _fmt_float(None if mean_len is None else float(mean_len), width=7, prec=2),
            _fmt_float(None if mean_score is None else float(mean_score), width=10, prec=3),
            _fmt_float(None if win_rate is None else float(win_rate), width=8, prec=3),
            _fmt_int(None if wins is None else int(wins), width=5),
            f"{_flag('reward'):>2}",
            f"{_flag('score'):>2}",
            f"{_flag('win'):>2}",
        ]

    def emit(
        self,
        *,
        metrics: dict[str, Any],
        timesteps: int,
        episodes: int,
        deterministic: bool,
        best_flags: dict[str, bool] | None = None,
        printer: Callable[[str], None],
    ) -> None:
        show_header = self._should_header()
        if show_header:
            printer(self._rule())
            printer(self.header())
            printer(self._rule())
            self._last_header_t = perf_counter()
        wall_time = metrics.get("wall_time")
        cells = self._row_cells(
            metrics=metrics,
            timesteps=timesteps,
            episodes=episodes,
            deterministic=deterministic,
            best_flags=best_flags,
        )
        time_cell = wall_time if isinstance(wall_time, str) and wall_time else "-"
        cells.append(f"{time_cell:>25}")

        line = "| " + " | ".join(cells) + " |"
        printer(line)
        self._rows += 1

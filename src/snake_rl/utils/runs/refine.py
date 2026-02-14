# src/snake_rl/utils/runs/refine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snake_rl.utils.runs.checkpoints import pick_checkpoint
from snake_rl.utils.runs.resume import resolve_resume_context


@dataclass(frozen=True)
class RefineContext:
    refine_arg: str
    run_dir: Path | None
    config_path: Path | None
    checkpoint_path: Path
    source_label: str


def _normalize_refine_from(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    aliases = {
        "reward": "best_reward",
        "score": "best_score",
        "win": "best_win",
    }
    return aliases.get(s, s)


def resolve_refine_context(
    refine: str,
    *,
    refine_from: str | None,
    runs_root: Path,
    legacy_root: Path | None = None,
) -> RefineContext:
    """
    Resolve refine context (run dir/config + checkpoint) from a refine target.

    refine_from:
      - latest, final
      - best_reward, best_score, best_win, best
      - path/to/checkpoint.zip
    """
    ctx = resolve_resume_context(
        refine,
        runs_root=runs_root,
        legacy_root=legacy_root,
    )

    norm = _normalize_refine_from(refine_from)
    # If refine_from is an explicit checkpoint path, use it.
    if norm is not None:
        p = Path(norm).expanduser()
        if p.suffix.lower() == ".zip" and p.is_file():
            return RefineContext(
                refine_arg=refine,
                run_dir=ctx.run_dir,
                config_path=ctx.config_path,
                checkpoint_path=p.resolve(),
                source_label="checkpoint",
            )

    if ctx.run_dir is None:
        # refine points directly to a checkpoint; no run_dir context.
        return RefineContext(
            refine_arg=refine,
            run_dir=None,
            config_path=None,
            checkpoint_path=ctx.checkpoint_path,
            source_label="checkpoint",
        )

    which = norm or "latest"
    if which not in {
        "latest",
        "final",
        "best",
        "best_reward",
        "best_score",
        "best_win",
    }:
        raise ValueError(
            "refine_from must be one of: latest, final, best, best_reward, best_score, best_win"
        )

    checkpoint = pick_checkpoint(run_dir=ctx.run_dir, which=which)
    return RefineContext(
        refine_arg=refine,
        run_dir=ctx.run_dir,
        config_path=ctx.config_path,
        checkpoint_path=checkpoint,
        source_label=str(which),
    )


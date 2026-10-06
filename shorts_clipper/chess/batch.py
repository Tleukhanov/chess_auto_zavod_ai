"""Batch a library of PGNs into chess shorts.

`scripts/make_chess_clip.py` renders one clip from one game and stops. That is
the right shape for a demo and the wrong shape for a library: an unattended run
over a folder of games has to survive what a folder of games actually contains --
a truncated export, a file that is not PGN at all, a single file holding five
games -- and it has to answer "which moments have I already cut?".

Three decisions define this module:

1. **Selection is global, not per-game.** ``--count N`` picks the best N
   moments across every game in the run, so a second folder with one 900cp
   blunder beats a folder of ten games that never dropped a piece. Candidates
   are ranked by the centipawn loss the mover actually took, which is the same
   number :mod:`shorts_clipper.chess.analysis` sorts by.
2. **Dedup is keyed on the moment, not on the script.** Reusing the hash
   machinery from :mod:`shorts_clipper.pipeline.stock_dedup` (same on-disk
   format, same ``load_used``/``record_used`` calls) but hashing
   ``fen_before + uci`` instead of caption text: the same blunder reached from
   two different exports of one game is the same moment and must be cut once,
   while two games that happen to share a move number are different moments.
3. **A source that cannot be read is a *result*, not an exception.** The batch
   records the reason and -- unless ``continue_on_error`` is set -- stops, which
   is the documented contract for every other batch in this repo.

Failure semantics follow `README.md` ("Batch multi-source runs"): fail fast by
default, ``--continue-on-error`` keeps going, and the run exits non-zero when
anything failed either way.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import re
from pathlib import Path

from shorts_clipper.chess import analysis, clip
from shorts_clipper.core.settings import Settings
from shorts_clipper.pipeline.stock_dedup import load_used, record_used, script_hash

log = logging.getLogger(__name__)

PGN_SUFFIXES = (".pgn",)
DEFAULT_CHESS_MUSIC = "dark_industrial_loop.wav"
DEFAULT_SEED = 7


class ChessBatchError(Exception):
    """Hard failure in batch mode: a source failed and the run aborted."""


@dataclasses.dataclass(frozen=True)
class Moment:
    """One renderable decision plus the game it came from."""

    decision: analysis.Decision
    source: Path
    game_index: int = 0

    @property
    def key(self) -> str:
        """Dedup digest: the position before the move plus the move itself."""
        return script_hash(f"{self.decision.fen_before} {self.decision.uci}")

    @property
    def label(self) -> str:
        headers = self.decision.header or {}
        white = str(headers.get("White") or "white")
        black = str(headers.get("Black") or "black")
        return f"{white} vs {black}"

    @property
    def slug(self) -> str:
        """Filesystem-safe ``players-m12-san`` name for the rendered clip."""
        headers = self.decision.header or {}
        white = _slug(str(headers.get("White") or "white"))
        black = _slug(str(headers.get("Black") or "black"))
        san = _slug(self.decision.san)
        return f"{white}-{black}_m{self.decision.move_number}_{san}"


@dataclasses.dataclass
class SourceResult:
    """Outcome of a single PGN file inside a batch."""

    index: int
    source: str
    ok: bool
    error: str | None = None
    candidates: int = 0
    skipped_used: int = 0
    games_in_file: int = 0
    clips: list[Path] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BatchResult:
    """Aggregate outcome of a :func:`run_batch` call."""

    ok: bool
    results: list[SourceResult]
    total: int
    succeeded: int
    failed: int
    clips: list[Path] = dataclasses.field(default_factory=list)
    skipped_duplicate: int = 0

    @property
    def has_engine(self) -> bool:
        return analysis.has_engine()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Lowercase ASCII-ish slug; non-Latin names collapse rather than vanish."""
    out = _SLUG_RE.sub("-", str(text or "").strip().lower()).strip("-")
    return out or "game"


def moment_key(decision: analysis.Decision) -> str:
    """Dedup digest for a single decision.

    Position + move, deliberately: the same blunder re-exported from a
    different site is still the same moment, and two unrelated games that both
    blunder on move 12 are not.
    """
    return script_hash(f"{decision.fen_before} {decision.uci}")


def used_path(settings: Settings | None = None) -> Path:
    """Path of the chess dedup history."""
    if settings is None:
        return Path("data/chess_used.json")
    return Path(settings.chess_used_path)


def load_used_moments(path: str | Path) -> set[str]:
    """Digests already cut, empty on a missing or corrupt history file."""
    return load_used(path)


def clear_used(path: str | Path) -> int:
    """Forget every recorded moment. Returns how many digests were dropped.

    Dedup is a promise across runs, so removing the file is the only honest way
    to say "cut these again" -- re-recording the same digests would leave the
    next run exactly where it was.
    """
    target = Path(path)
    before = load_used(target)
    try:
        target.unlink()
    except FileNotFoundError:
        return 0
    except OSError as exc:
        log.warning("could not clear chess dedup at %s: %s", target, exc)
        return 0
    log.info("cleared %d chess moment(s) from %s", len(before), target)
    return len(before)


def collect_sources(paths: list[str] | tuple[str, ...]) -> list[Path]:
    """Expand the CLI paths into the PGN files to run, sorted and deduplicated.

    A directory contributes every ``*.pgn`` beneath it; an explicitly named file
    is always kept, even with an odd extension, because the user named it and
    the parser -- not the filename -- decides whether it is a game.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            return
        seen.add(key)
        out.append(candidate)

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(
                child for child in p.rglob("*")
                if child.is_file() and child.suffix.lower() in PGN_SUFFIXES
            )
            for child in found:
                _add(child)
        else:
            _add(p)
    return out


def _read_game(path: Path) -> tuple[object | None, int, str | None]:
    """Return ``(game, games_in_file, error)`` for one PGN file.

    Only the *first* game in a file is analysed, and the caller is told how many
    there were so it can say so: a multi-game export is common, and silently
    cutting game one of five reads as "that was all of them".
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, 0, f"unreadable: {exc.strerror or exc}"

    game = analysis.load_pgn(text)
    if game is None:
        return None, 0, "not a PGN: no moves found"

    total = _count_games(text)
    return game, total, None


def _count_games(text: str) -> int:
    """How many games *text* holds, at least 1.

    python-chess has no ``read_games``; a stream is drained by calling
    ``read_game`` until it returns None. A count that cannot be established
    degrades to 1 -- the common case -- rather than failing the file.
    """
    chess = analysis._chess_module()  # noqa: SLF001 - shared import probe
    if chess is None:
        return 1
    import io

    handle = io.StringIO(text)
    total = 0
    try:
        while True:
            if chess.pgn.read_game(handle) is None:
                break
            total += 1
    except Exception:
        log.debug("could not count games", exc_info=True)
    return max(total, 1)


def resolve_music(
    music: str | Path | None,
    no_music: bool,
    settings: Settings,
) -> Path | None:
    """Pick the bed: explicit path, else the default track in ``music_dir``.

    A missing file is reported by the caller rather than swallowed here: a clip
    that ships silent because a bed was misspelled looks identical to a clip
    that was meant to be silent.
    """
    if no_music:
        return None
    candidate = Path(music) if music is not None else Path(settings.music_dir) / DEFAULT_CHESS_MUSIC
    if not candidate.is_file():
        log.warning("no music bed at %s", candidate)
        return None
    return candidate


def render_moment(
    moment: Moment,
    out_dir: Path,
    music: Path | None,
    index: int,
    seed: int = DEFAULT_SEED,
) -> Path | None:
    """Render one moment: silent clip, then the bed mixed in.

    ``index`` only offsets the bed selection so two clips from one game do not
    start on the same bar.
    """
    plan = clip.plan_for_decision(moment.decision)
    if not plan.frames:
        log.error("no frame plan for %s", moment.label)
        return None

    slug = moment.slug
    silent = clip.render_clip(plan, out_dir / f"{slug}_silent.mp4")
    if silent is None:
        return None

    offset = 0.0
    if music is not None:
        try:
            from shorts_clipper.captions.music import energetic_offset

            offset = energetic_offset(music, random.Random(seed + index), plan.duration)
        except Exception:
            log.debug("energetic_offset failed for %s", music, exc_info=True)

    final = clip.add_bed(silent, music, out_dir / f"{slug}.mp4", start_seconds=offset)
    # The silent intermediate is not a deliverable; leaving it behind would fill
    # the output folder with duplicates the operator has to recognise by name.
    try:
        Path(silent).unlink()
    except OSError:
        pass
    return Path(final)


def plan_batch(
    sources: list[Path],
    *,
    count: int,
    critical_cp: int,
    min_ply: int,
    used: set[str],
) -> tuple[list[Moment], list[SourceResult], int]:
    """Analyse every source and pick the best *count* moments overall.

    Returns ``(selected, results, skipped_duplicate)``. Candidates are ranked
    by mover loss across *all* games, which is what makes ``--count`` a run-wide
    budget rather than a per-game quota.
    """
    results: list[SourceResult] = []
    candidates: list[Moment] = []
    skipped = 0

    for i, path in enumerate(sources, 1):
        game, games_in_file, error = _read_game(path)
        if error is not None:
            results.append(SourceResult(index=i, source=str(path), ok=False, error=error))
            continue
        if games_in_file > 1:
            log.info(
                "%s holds %d games; analysing the first only", path.name, games_in_file
            )

        decisions = analysis.find_decisions(
            game, critical_cp=critical_cp, min_ply=min_ply, limit=max(count, 1)
        )
        result = SourceResult(index=i, source=str(path), ok=True, games_in_file=games_in_file)
        fresh = 0
        for dec in decisions:
            moment = Moment(decision=dec, source=path, game_index=games_in_file)
            if moment.key in used:
                result.skipped_used += 1
                skipped += 1
                log.info("already cut, skipping: %s — %s", path.name, dec.caption())
                continue
            candidates.append(moment)
            fresh += 1
        result.candidates = fresh
        results.append(result)

    candidates.sort(key=lambda m: m.decision.mover_loss_cp, reverse=True)
    return candidates[: max(count, 1)], results, skipped


def run_batch(
    sources: list[str] | tuple[str, ...],
    *,
    settings: Settings | None = None,
    count: int = 1,
    critical_cp: int | None = None,
    min_ply: int | None = None,
    output_dir: str | Path | None = None,
    music: str | Path | None = None,
    no_music: bool = False,
    seed: int = DEFAULT_SEED,
    continue_on_error: bool = False,
    used_file: str | Path | None = None,
) -> BatchResult:
    """Cut up to *count* chess shorts from a PGN, a directory, or a list of both.

    ``sources`` may mix single PGN files and directories. Files are analysed
    before anything is rendered, so the selection is a run-wide ranking rather
    than a per-game quota.

    Raises :class:`ChessBatchError` when a source fails and
    ``continue_on_error`` is False, matching ``run_batch`` in the clip
    pipeline; the caller is expected to report it and exit non-zero.
    """
    settings = settings or Settings()
    critical_cp = (
        int(critical_cp) if critical_cp is not None else int(settings.chess_critical_cp)
    )
    min_ply = int(min_ply) if min_ply is not None else int(settings.chess_min_ply)

    files = collect_sources(sources)
    if not files:
        raise ChessBatchError(f"no .pgn files found in: {', '.join(str(s) for s in sources)}")

    out_dir = Path(output_dir) if output_dir else Path(settings.chess_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = Path(used_file) if used_file else used_path(settings)
    used = load_used_moments(history)
    if used:
        log.info("dedup: %d moment(s) already cut (%s)", len(used), history)

    bed = resolve_music(music, no_music, settings)

    selected, results, skipped = plan_batch(
        files,
        count=count,
        critical_cp=critical_cp,
        min_ply=min_ply,
        used=used,
    )

    # Fail fast *before* rendering: a source that could not be read is reported
    # and the run aborts, exactly as documented for the other batch commands.
    # Checking after the scan (rather than mid-scan) keeps the whole library
    # assessed in one pass, so the abort message can say how many files were
    # unusable rather than only naming the first.
    broken = [r for r in results if not r.ok]
    if broken and not continue_on_error:
        first = broken[0]
        raise ChessBatchError(
            f"[{first.index}/{len(results)}] {first.source} FAILED: {first.error}"
        )

    by_source: dict[str, SourceResult] = {r.source: r for r in results}
    written: list[Path] = []

    for index, moment in enumerate(selected, 1):
        result = by_source[str(moment.source)]
        try:
            path = render_moment(moment, out_dir, bed, index, seed=seed)
        except Exception as exc:  # a render blow-up is a failed source, not a crash
            log.debug("render failed for %s", moment.source, exc_info=True)
            path = None
            result.ok = False
            result.error = f"render failed: {exc}"
        if path is None or not Path(path).is_file():
            if result.ok:
                result.ok = False
                result.error = "ffmpeg render failed"
            if not continue_on_error:
                raise ChessBatchError(f"{moment.source} FAILED: {result.error}")
            continue
        result.clips.append(Path(path))
        written.append(Path(path))
        record_used(history, moment.key)

    failed = [r for r in results if not r.ok]
    return BatchResult(
        ok=not failed,
        results=results,
        total=len(results),
        succeeded=len(results) - len(failed),
        failed=len(failed),
        clips=written,
        skipped_duplicate=skipped,
    )
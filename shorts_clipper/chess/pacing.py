"""Highlight pacing: skim the game, then slow down where it was decided.

Treating every ply equally is the wrong edit. In a real game a hundred moves
are shuffling and one of them threw the position away, so giving each the same
two seconds spends the viewer's attention where nothing happens.

So the pace is inverted: the bulk of the game passes quickly, and only a short
window around the decisive move is held long enough to read. The measurement
that motivated this: a 114-ply game with an 11-ply slow window gave those 11
plies 1.9x the time of the other 103 combined -- not the 20x the format is
actually about. Hence a narrower window, a shorter hold, and a faster skim.

Frames are sampled rather than one-per-move: at skim speed a frame every few
plies reads as the board flying past, which is the intent, and keeps a
long game to a few dozen renders.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Plies per second while skimming. 18 is fast enough to read as "this is just
# context" and slow enough that individual moves are still perceptible.
SKIM_PLIES_PER_SECOND = 18.0
# Plies per frame while skimming: more means fewer renders and a coarser blur.
SKIM_PLIES_PER_FRAME = 4
# Seconds per ply inside the window. Long enough to actually read the position.
SLOW_SECONDS_PER_PLY = 1.35
# Plies of slow motion either side of the decisive move.
SLOW_WINDOW_PLIES = 2
# Never render more than this many frames; a very long game must still fit.
MAX_FRAMES = 44


def _chess():
    try:
        import chess
        import chess.engine  # noqa: F401

        return chess
    except ImportError:
        return None


def pace_game(
    game,
    target_ply: int,
    *,
    skim_plies_per_second: float = SKIM_PLIES_PER_SECOND,
    skim_plies_per_frame: int = SKIM_PLIES_PER_FRAME,
    slow_seconds_per_ply: float = SLOW_SECONDS_PER_PLY,
    slow_window: int = SLOW_WINDOW_PLIES,
    max_frames: int = MAX_FRAMES,
):
    """Return (frames, holds) for the whole game with a slow window at *target_ply*.

    *frames* is a list of :class:`FrameSpec`, *holds* the matching seconds. Each
    frame shows the position **after** the last ply it covers, so the board never
    shows a move that has not been spoken yet.
    """
    chess = _chess()
    if chess is None or game is None:
        return [], []

    moves = list(game.mainline_moves())
    total = len(moves)
    if not total:
        return [], []

    target = max(0, min(int(target_ply), total - 1))
    slow_from = max(0, target - slow_window)
    slow_to = min(total - 1, target + slow_window)

    # Sample the ply indices we will render a frame for.
    samples: list[tuple[int, bool]] = []  # (last ply covered, is_slow)
    i = 0
    while i < total:
        if slow_from <= i <= slow_to:
            samples.append((i, True))
            i += 1
            continue
        if i > slow_to:
            # Past the window: plain fast step. Falling through to the
            # crossing-safe branch here re-appended slow_from - 1 forever, which
            # collapsed a 114-ply game to 8 frames.
            step = max(1, int(skim_plies_per_frame))
            end = min(i + step - 1, total - 1)
            samples.append((end, False))
            i = end + 1
            continue
        # Before the window: a skim step must not cross into it, or one frame
        # would have to hold at both speeds.
        step = max(1, int(skim_plies_per_frame))
        end = min(i + step - 1, slow_from - 1, total - 1)
        if end < i:
            samples.append((i, False))
            i += 1
        else:
            samples.append((end, False))
            i = end + 1

    if len(samples) > max_frames:
        slow_samples = [s for s in samples if s[1]]
        keep = max(1, max_frames - len(slow_samples))
        fast_idx = [n for n, (_, s) in enumerate(samples) if not s]
        # Round the stride up: slicing with [::step] yields ceil(len/step)
        # items, so a floored step overshot the cap (22 frames for a cap of 20).
        step = max(1, -(-len(fast_idx) // keep))
        thinned = [samples[n] for n in fast_idx[::step]] + slow_samples
        thinned.sort(key=lambda item: item[0])
        samples = thinned

    # Walk the board once, snapshotting each sampled position.
    from shorts_clipper.chess.board import FrameSpec

    wanted = {ply for ply, _ in samples}
    frames: list[FrameSpec] = []
    holds: list[float] = []
    board = chess.Board()
    slow_count = 0
    for ply in range(total):
        board.push(moves[ply])
        if ply not in wanted:
            continue
        slow = next(s for p, s in samples if p == ply)
        if slow:
            slow_count += 1
            holds.append(slow_seconds_per_ply)
            caption = _slow_caption(chess, board, ply, target)
        else:
            holds.append(skim_plies_per_frame / skim_plies_per_second)
            caption = _skim_caption(ply + 1, total)
        frames.append(
            FrameSpec(
                fen=board.fen(),
                top_text=caption,
                bottom_text="",
                accent="bad" if slow else "neutral",
            )
        )
    return frames, holds


def _skim_caption(ply_done: int, total: int) -> str:
    return f"{ply_done} / {total}"


def _slow_caption(chess, board, ply: int, target: int) -> str:
    if ply == target:
        return f"вот этот ход — {ply // 2 + 1}…"
    return f"{ply // 2 + 1}…"


def plan_for_highlight(
    game,
    decision,
    *,
    skim_plies_per_second: float = SKIM_PLIES_PER_SECOND,
    skim_plies_per_frame: int = SKIM_PLIES_PER_FRAME,
    slow_seconds_per_ply: float = SLOW_SECONDS_PER_PLY,
    slow_window: int = SLOW_WINDOW_PLIES,
):
    """A whole-game clip that skims and then slows at the decisive move."""
    from shorts_clipper.chess.clip import ChessClipPlan

    frames, holds = pace_game(
        game,
        getattr(decision, "ply", 0),
        skim_plies_per_second=skim_plies_per_second,
        skim_plies_per_frame=skim_plies_per_frame,
        slow_seconds_per_ply=slow_seconds_per_ply,
        slow_window=slow_window,
    )
    if not frames:
        return ChessClipPlan(frames=[])
    return ChessClipPlan(frames=frames, holds=holds)
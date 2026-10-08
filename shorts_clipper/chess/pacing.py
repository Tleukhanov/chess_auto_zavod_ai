"""Highlight pacing: skim the game, then slow down where it was decided.

Treating every ply equally is the wrong edit. In a real game a hundred moves
are shuffling and one of them threw the position away, so giving each the same
two seconds spends the viewer's attention where nothing happens.

So the pace is inverted: the bulk of the game passes quickly, and only a short
window around the decisive move is held long enough to read.

The clip is three beats, in this order:

1. **skim** -- the whole game, one frame every :data:`SKIM_PLIES_PER_FRAME`
   plies, held :data:`SKIM_PLIES_PER_FRAME / SKIM_PLIES_PER_SECOND` seconds.
   0.125s a frame is the difference between "the board is flying past" and "a
   slideshow of positions I have to read one by one": the earlier build held
   0.22s over 4 plies, which is long enough to start reading a position and far
   too short to finish, so it carried no meaning at all. Six plies is one whole
   move pair, so each skim frame carries a complete move rather than half of one.
2. **window** -- :data:`SLOW_WINDOW_PLIES` plies either side of the decisive
   move, replayed one frame per move number at :data:`SLOW_SECONDS_PER_PLY` a
   frame, with the decisive ply itself held :data:`DECISIVE_SECONDS`.
3. **outro** -- the final position, held once.

The trade-off this buys, stated plainly: **the decisive moment is shown twice.**
The clip is not chronological after the skim; the window is a replay placed at
the end. The alternative -- keeping strict ply order -- is what produced the
measured defect this replaced: on a 134-ply game whose blunder was ply 24, the
window landed at 20% of the runtime and 8 seconds of blundering-at-speed
followed it, so the payoff was over before the viewer had seen the game. No
amount of re-ordering inside ply order fixes that, because the time before a
reveal is only the plies that precede it: an early blunder buys a short setup
no matter how the frames are arranged. Skimming the whole game and then
replaying the window costs -- a position the viewer already saw for 0.125s gets
another 1.5-2.6s -- and buys a payoff with nothing skimming behind it.

Measured over 40 real games at three target plies each, the reveal frame sits at
73-92% of the clip's frames, and the only thing after it is the held outro:
1.6s, down from 7.1s of skimming. In wall-clock terms the reveal lands at
29-51%, which is the skim's doing and not the window's -- the setup is short
because the setup is a speedrun, and buying it back would mean slowing the skim,
which undoes the format.

Two rules keep the replay from being padding:

- **One frame per move number inside the window.** Five plies of a quiet stretch
  render five near-identical boards; a 2fps contact sheet of the old build read
  ``24... 24... 24... 25... 25...``. Emitting at most one frame per move number
  makes consecutive window frames a full move apart, which is the smallest gap
  that reliably changes the picture.
- **Stop when the position stops being interesting.** Past
  :data:`MIN_PIECES_TO_SHOW` pieces the game is being played out, not played,
  and the old build spent its last six seconds on a bare king.
"""

from __future__ import annotations

import dataclasses
import logging

log = logging.getLogger(__name__)

# Plies per second while skimming. 48 is fast enough to read as "this is just
# context": at 6 plies a frame that is 0.125s, or 7.5 positions a second.
SKIM_PLIES_PER_SECOND = 48.0
# Plies per frame while skimming: a full move pair, so one skim frame carries
# one complete move rather than half of one. More would blur the game into
# noise; fewer and a long game stops fitting in the frame budget.
SKIM_PLIES_PER_FRAME = 6
# Seconds per ply of context either side of the decisive move. Long enough to
# read the position the mistake was made in.
SLOW_SECONDS_PER_PLY = 1.5
# Seconds for the decisive ply itself, which is the one frame the whole clip is
# built around. It gets the longest hold because it is the only frame whose
# content is the point: flanking plies in a quiet stretch move a rook or shuffle
# a knight, so holding each of them as long as the blunder only bought air.
DECISIVE_SECONDS = 2.6
# Plies of slow motion either side of the decisive move. The window is five plies
# wide but only yields three frames -- see the move-number rule above.
SLOW_WINDOW_PLIES = 2
# Never render more than this many frames; a very long game must still fit.
MAX_FRAMES = 44
# Seconds for the held final position.
OUTRO_SECONDS = 1.6
# The renderer clamps every hold to 0.08s, so a plan that asks for less than
# that is silently re-timed by ffmpeg and the plan stops describing the clip.
MIN_HOLD_SECONDS = 0.08
# How many pieces a position needs before it is still worth showing. Below this
# the clip stops: a bare king, or king-and-queen, is the game being played out
# rather than played, and no viewer is reading it.
MIN_PIECES_TO_SHOW = 6
# A short is a short. A 40-ply game skims in under a second, so without a floor
# the plan for it came to 6.5s and the clip was cut rather than finished. The
# deficit is absorbed by the frames the viewer is meant to read (see _pad),
# never by the skim.
MIN_CLIP_SECONDS = 8.0
# Relative share of a padding deficit each kind of frame absorbs. The skim gets
# zero: padding the skim is the one thing that would undo it.
_PAD_WEIGHT = {"skim": 0.0, "window": 2.0, "outro": 1.0}


def _chess():
    try:
        import chess
        import chess.engine  # noqa: F401

        return chess
    except ImportError:
        return None


@dataclasses.dataclass(frozen=True)
class _Beat:
    """One emitted frame: the ply it shows, and how slowly it is read."""

    ply: int
    kind: str  # "skim", "window" or "outro"

    @property
    def slow(self) -> bool:
        return self.kind == "window"


def resolve_target_ply(game, decision) -> int | None:
    """The 0-based mainline index of *decision*'s move, or ``None``.

    ``Decision.ply`` is 1-based (``analysis.find_decisions`` stores ``ply + 1``)
    while :func:`pace_game` indexes ``game.mainline_moves()`` from 0. Passing
    ``decision.ply`` straight through points one ply late, which put the reveal
    arrow on the opponent's reply and named the wrong move in the caption -- a
    114-ply game whose decisive move was ``Qxd6+`` showed ``Bxd6``.

    The move is located by its UCI, but a game that repeats a position plays the
    same UCI more than once, so taking the first match is wrong: a decision on
    ply 64 whose move also occurs on ply 2 resolved to 2, which put the reveal
    window at the start of the game and left the clip with no arrow at all. The
    occurrence *closest to* the expected 1-based index wins instead.
    """
    moves = list(game.mainline_moves())
    total = len(moves)
    if not total:
        return None

    if getattr(decision, "ply", None) is None and not getattr(decision, "uci", ""):
        # Nothing to go on: no ply to correct and no move to look for.
        return None

    raw = getattr(decision, "ply", None)
    try:
        expected = max(0, int(raw) - 1) if raw is not None else 0
    except (TypeError, ValueError):
        expected = 0

    uci = getattr(decision, "uci", "") or ""
    if uci:
        matches = [i for i, move in enumerate(moves) if move.uci() == uci]
        if matches:
            return min(matches, key=lambda i: (abs(i - expected), i))
    return max(0, min(expected, total - 1))


def pace_game(
    game,
    target_ply: int,
    *,
    skim_plies_per_second: float = SKIM_PLIES_PER_SECOND,
    skim_plies_per_frame: int = SKIM_PLIES_PER_FRAME,
    slow_seconds_per_ply: float = SLOW_SECONDS_PER_PLY,
    slow_window: int = SLOW_WINDOW_PLIES,
    max_frames: int = MAX_FRAMES,
    decision=None,
):
    """Return (frames, holds) for the whole game with a slow window at *target_ply*.

    *frames* is a list of :class:`FrameSpec`, *holds* the matching seconds. Each
    frame shows the position **after** the last ply it covers, so the board never
    shows a move that has not been spoken yet.

    *target_ply* is a **0-based** mainline index. Use :func:`resolve_target_ply`
    to convert an :class:`~shorts_clipper.chess.analysis.Decision`, whose ``ply``
    is 1-based.

    *decision* is that same decision. It is passed to narration for the reveal
    frame, which is the only frame that may show an evaluation swing or call the
    move a blunder.
    """
    chess = _chess()
    if chess is None or game is None:
        return [], []

    moves = list(game.mainline_moves())
    total = len(moves)
    if not total:
        return [], []

    fens, placements, piece_counts = _walk(chess, moves)

    # Everything up to the last position still worth showing, and no further.
    last_shown = _last_worth_showing(piece_counts)
    # A blunder in the collapsed tail is not worth replaying: the window would
    # end the clip on the bare king the piece rule exists to avoid.
    target = max(0, min(int(target_ply), last_shown))

    window = _window(target, last_shown + 1, slow_window)
    # The final position earns its own held frame only if the game actually
    # reached it with pieces left, and the window does not already end on it.
    outro_ply = (
        last_shown if last_shown == total - 1 and last_shown not in window else None
    )

    step = max(1, int(skim_plies_per_frame))
    # The skim must not land on the position the outro shows, or the two frames
    # are the same picture back to back.
    skim_stop = last_shown - 1 if outro_ply is not None else last_shown
    skim = range(0, skim_stop + 1, step)

    beats = [_Beat(ply, "skim") for ply in skim]
    beats += [_Beat(ply, "window") for ply in window]
    if outro_ply is not None:
        beats.append(_Beat(outro_ply, "outro"))

    beats = _thin(beats, max(1, int(max_frames)), target)
    beats = _drop_repeats(beats, placements, target)

    from shorts_clipper.chess.board import FrameSpec

    skim_hold = max(
        MIN_HOLD_SECONDS, step / max(0.001, float(skim_plies_per_second))
    )
    decisive_hold = max(MIN_HOLD_SECONDS, DECISIVE_SECONDS)
    slow_hold = max(MIN_HOLD_SECONDS, float(slow_seconds_per_ply))
    outro_hold = max(MIN_HOLD_SECONDS, OUTRO_SECONDS)

    holds = [
        decisive_hold if b.slow and b.ply == target
        else slow_hold if b.slow
        else outro_hold if b.kind == "outro"
        else skim_hold
        for b in beats
    ]
    holds = _pad(holds, beats, MIN_CLIP_SECONDS)

    frames: list[FrameSpec] = []
    for beat in beats:
        spec = FrameSpec(
            fen=fens[beat.ply],
            top_text=_caption(beat, target, game, total),
            bottom_text="",
            accent="bad" if beat.slow else "neutral",
        )
        # Narration fills the information band below the board. It needs the two
        # positions around the ply and whose side was to move, none of which the
        # beat carries, so they are recovered here from the game's own chain.
        info, arrow, accent = _narrate_beat(
            beat=beat,
            target=target,
            game=game,
            fens=fens,
            total_plies=total,
            decision_for=decision,
        )
        # Narration may only add emphasis, never remove it: a lead-in frame in
        # the slow window keeps the window's accent even though narration
        # classifies it as a lead. Taking narration's accent unconditionally
        # turned 2 of the 3 window frames neutral and broke the window's
        # visual signature.
        merged = "bad" if (spec.accent == "bad" or accent == "bad") else spec.accent
        frames.append(
            dataclasses.replace(spec, arrow=arrow, accent=merged, info=info)
        )
    return frames, holds


def _narrate_beat(*, beat, target, game, fens, total_plies, decision_for=None):
    """``(FrameInfo, arrow, accent)`` for one beat, degrading to ``None``.

    A beat can cover several plies during the skim, and narration wants a single
    move to name. Those beats get the band filled as far as the position allows
    and no arrow, which is the honest answer rather than pointing at an
    arbitrary move.
    """
    try:
        from shorts_clipper.chess import narration
    except ImportError:
        return None, None, None

    ply = beat.ply
    # fens[ply] is the position AFTER ply, so the position the move was played
    # from is fens[ply - 1]. A beat on ply 0 has no prior position to speak
    # about, so both sides fall back to the position itself.
    fen_after = fens[ply] if 0 <= ply < len(fens) else ""
    fen_before = fens[ply - 1] if 0 < ply <= len(fens) else fen_after

    san = fens_san(game, ply)

    # narrate() derives its own frame kind from the decision and the ply, so
    # there is no kind argument to pass. The decision is supplied only on the
    # reveal: a skim or outro frame has no single move that was decisive, and
    # handing it one would put a plausible-looking arrow on an arbitrary move.
    decision = _NarrationDecision(ply, san) if ply == target else None

    mover = ""
    if fen_after:
        try:
            import chess

            mover = "white" if chess.Board(fen_after).turn == chess.WHITE else "black"
        except (ImportError, ValueError):
            mover = ""

    # The real Decision carries the engine evaluation and the material flag;
    # narration uses both to decide whether it may call the move a blunder and to
    # print the swing that proves it. A zeroed stub would render "0.0 → 0.0".
    decision = decision_for if ply == target else None

    n = narration.narrate(
        decision=decision,
        ply=ply,
        total_plies=total_plies,
        fen_before=fen_before,
        fen_after=fen_after,
        color=mover,
        headers=getattr(game, "headers", None),
    )
    return n.info, n.arrow, n.accent


def fens_san(game, ply: int) -> str:
    """SAN of the move played at *ply*, or an empty string.

    ``game.board()`` already needs python-chess, so importing it here only to
    leave it unused was pointless: the ImportError that the guard catches is
    raised by ``game.board()`` itself.
    """
    try:
        board = game.board()
        for index, move in enumerate(game.mainline_moves()):
            if index == ply:
                return board.san(move)
            board.push(move)
    except (ImportError, ValueError, IndexError):
        return ""
    return ""


@dataclasses.dataclass(frozen=True)
class _NarrationDecision:
    """The slice of ``analysis.Decision`` narration actually reads."""

    ply: int
    san: str
    uci: str = ""
    eval_before_cp: int = 0
    eval_after_cp: int = 0
    mover_loss_cp: int = 0
    material: bool = True
    mate_before: int | None = None
    mate_after: int | None = None


def _caption(beat: _Beat, target: int, game, total_plies: int) -> str:
    if beat.slow:
        return _slow_caption(beat.ply, target)
    if beat.kind == "outro":
        return _outro_caption(game, beat.ply)
    return _skim_caption(beat.ply + 1, total_plies)


def _window(target: int, limit: int, half: int) -> list[int]:
    """The plies the slow window renders: at most one frame per move number.

    Centred on the decisive move, except when the game runs out first. A window
    that gets clipped by the end of the game used to shrink to two plies, and a
    51-ply game whose worst move was its last one came out as a 4-second clip.
    So a shortfall is taken from the other side: a window is never shorter than
    the viewer was promised, even if it ends up all before the move and none
    after it.

    Then every move number inside it keeps one ply and loses the rest. That is
    the fix for the window holding one board three times: within a move, the
    second ply is a reply to the first and the picture barely moves, so both are
    never worth the same second and a half. The decisive ply always survives.

    The surviving ply of a neighbouring move is the one furthest from it, so
    consecutive window frames land two plies apart -- a whole move, which is the
    smallest gap that reliably changes the picture. Measured over 40 real
    games: the longest run of adjacent slow frames differing by a quiet move
    fell from 5 (the whole window) to 0.
    """
    half = max(0, int(half))
    after = min(half, limit - 1 - target)
    before = min(half, target)
    before = min(target, before + (2 * half - before - after))
    plies = range(target - before, target + after + 1)

    kept: list[int] = []
    group: list[int] = []

    def flush() -> None:
        if not group:
            return
        if target in group:
            kept.append(target)
        else:
            # Furthest the decisive ply, not nearest: the ply that answers it is
            # one move off at best, so picking the near one would put two frames
            # one ply apart -- the quiet pair the rule exists to avoid.
            kept.append(min(group, key=lambda ply: (-abs(ply - target), ply)))
        group.clear()

    for ply in plies:
        if group and ply // 2 != group[0] // 2:
            flush()
        group.append(ply)
    flush()
    return kept


def _walk(chess, moves) -> tuple[list[str], list[str], list[int]]:
    """FEN, piece placement and piece count after every ply, from one walk."""
    board = chess.Board()
    fens: list[str] = []
    placements: list[str] = []
    counts: list[int] = []
    for move in moves:
        board.push(move)
        fens.append(board.fen())
        placements.append(_placement(chess, board))
        counts.append(len(board.piece_map()))
    return fens, placements, counts


def _placement(chess, board) -> str:
    """What the board actually shows: which piece stands where.

    Two plies in a quiet stretch often differ only in the side-to-move field,
    which no rendered pixel depends on, so comparing whole FENs says "changed"
    for a position the viewer sees as identical. Comparing the pieces is what
    makes "materially unchanged" mean something on screen.
    """
    return "".join(
        f"{chess.square_name(square)}={piece.symbol()};"
        for square, piece in sorted(board.piece_map().items())
    )


def _last_worth_showing(counts: list[int]) -> int:
    """The last ply whose position still has :data:`MIN_PIECES_TO_SHOW` pieces."""
    for ply in range(len(counts) - 1, -1, -1):
        if counts[ply] >= MIN_PIECES_TO_SHOW:
            return ply
    # Nothing ever qualified, which only happens in a stub game: show its end
    # rather than return nothing.
    return len(counts) - 1


def _drop_repeats(beats: list[_Beat], placements: list[str], target: int) -> list[_Beat]:
    """Drop frames whose board is materially unchanged from the frame before.

    The move-number rule in :func:`_window` is what stops the repetition; this
    is the guard at the seams, where a skim frame, the window or the outro can
    otherwise land on the position next to it. The decisive ply is exempt -- it
    is the one frame the clip exists for, so it is emitted even if it happens to
    look like its neighbour.
    """
    out: list[_Beat] = []
    for beat in beats:
        if (
            out
            and beat.ply != target
            and placements[beat.ply] == placements[out[-1].ply]
        ):
            continue
        out.append(beat)
    return out


def _thin(beats: list[_Beat], max_frames: int, target: int) -> list[_Beat]:
    """Fit into *max_frames* by dropping skim frames, keeping the ends.

    The window is what the clip is for, so it is the last thing to go and the
    skim frames are the ones dropped. The first and last skim frame are kept
    explicitly: the old version thinned with ``[::step]``, which keeps the head
    but silently loses the tail, and which also yields ``ceil(len/step)``
    frames, so a floored step overshot the cap (22 frames for a cap of 20).
    """
    if len(beats) <= max_frames:
        return beats

    slow = [i for i, beat in enumerate(beats) if beat.slow]
    fast = [i for i, beat in enumerate(beats) if not beat.slow]
    spare = max_frames - len(slow)
    if spare < 1:
        # The window on its own is over budget. What survives is the decisive
        # ply and whatever is nearest to it, which is the least disposable part
        # of the window.
        keep = sorted(
            slow, key=lambda i: (abs(beats[i].ply - target), beats[i].ply)
        )[: max(1, max_frames)]
        slow = sorted(keep)
        spare = max_frames - len(slow)

    if len(fast) <= spare:
        keep = set(slow) | set(fast)
        return [beat for i, beat in enumerate(beats) if i in keep]

    last = len(fast) - 1
    # Exact integer spread, endpoints included: no ceil overshoot, no lost tail.
    if spare <= 0:
        picked: set[int] = set()
    elif spare == 1:
        picked = {last}
    else:
        picked = {(i * last) // (spare - 1) for i in range(spare)}
    keep = {fast[p] for p in picked} | set(slow)
    return [beat for i, beat in enumerate(beats) if i in keep]


def _pad(holds: list[float], beats: list[_Beat], minimum: float) -> list[float]:
    """Grow the readable frames until the clip reaches *minimum* seconds.

    A 40-ply game skims in about 0.9s, which left the plan 6.5s long -- a cut
    rather than a clip. The deficit is spread over the window and the outro by
    weight and given to the skim not at all, because a skim stretched to fill
    time stops being a skim. Deterministic: the same plan always comes out the
    same, with any rounding residue settled on the heaviest frame.
    """
    if not holds:
        return holds
    deficit = minimum - sum(holds)
    if deficit <= 0:
        return holds

    weights = [_PAD_WEIGHT[beat.kind] for beat in beats]
    total = float(sum(weights))
    if total <= 0:
        weights = [1.0] * len(holds)
        total = float(len(holds))

    out = [round(h + deficit * w / total, 3) for h, w in zip(holds, weights, strict=True)]
    residue = round(minimum - sum(out), 3)
    if residue:
        heaviest = max(range(len(out)), key=lambda i: weights[i])
        out[heaviest] = round(out[heaviest] + residue, 3)
    return [max(MIN_HOLD_SECONDS, h) for h in out]


def _skim_caption(ply_done: int, total: int) -> str:
    """Progress in move numbers, because ``49 / 134`` reads as 134 moves."""
    return f"{ply_done // 2} / {total // 2}"


def _slow_caption(ply: int, target: int) -> str:
    if ply == target:
        return f"вот этот ход — {ply // 2 + 1}…"
    return f"{ply // 2 + 1}…"


def _outro_caption(game, ply_done: int) -> str:
    try:
        result = str(game.headers.get("Result", "") or "").strip()
    except Exception:
        result = ""
    tail = f" · {result}" if result and result != "*" else ""
    return f"финал · {ply_done // 2 + 1}…{tail}"


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

    target = resolve_target_ply(game, decision)
    if target is None:
        return ChessClipPlan(frames=[])

    frames, holds = pace_game(
        game,
        target,
        skim_plies_per_second=skim_plies_per_second,
        skim_plies_per_frame=skim_plies_per_frame,
        slow_seconds_per_ply=slow_seconds_per_ply,
        slow_window=slow_window,
        decision=decision,
    )
    if not frames:
        return ChessClipPlan(frames=[])
    return ChessClipPlan(frames=frames, holds=holds)
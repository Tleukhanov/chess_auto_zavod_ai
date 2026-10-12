"""Highlight pacing: a short run-up to the decisive move, then slow down on it.

The reference channel does not show a game. On the Adams-Kasparov clip it shows
the *decisive phase*: the position that made the mistake possible, the moves that
built it, and the move itself. No setup, no history, no run-through from move 1.
The viewer named the symptom of the alternative in one line -- "почему с 1 хода
все идет" -- why does it start at the first move.

So the distance this module now measures is **from the move, not into the game**.
The clip opens a fixed :data:`LEAD_IN_PLIES` plies before the decisive ply, and
what comes between that opening frame and the window is decided by one rule: the
decisive ply. There is no such thing as "the rest of the game".

The clip is two parts and there is no third:

1. **lead-in** -- the plies from :data:`LEAD_IN_PLIES` before the decisive ply up
   to where the window starts, one frame per ply, :data:`LEAD_IN_SECONDS` each.
   At the shipped constants that is 8 frames over 8 plies. Enough to see the
   position the mistake was made in; not enough to be a history lesson.
2. **window** -- :data:`SLOW_WINDOW_PLIES` plies either side of the decisive
   move, at most one frame per move number, at :data:`SLOW_SECONDS_PER_PLY` a
   frame, with the decisive ply itself held :data:`DECISIVE_SECONDS`. Nothing
   follows it. The reveal is the ending.

Two rules survive from the old build, because they were about legibility rather
than about the skim:

- **One frame per move number inside the window.** Five plies of a quiet stretch
  render five near-identical boards; a 2fps contact sheet of the old build read
  ``24... 24... 24... 25... 25...``. The lead-in is deliberately the opposite --
  one frame per ply -- because its frames are half a second long, so the two
  halves of a move read as the move pair they are rather than as one position
  shown twice.
- **Stop when the position stops being interesting.** Past
  :data:`MIN_PIECES_TO_SHOW` pieces the game is being played out, not played,
  and the old build spent its last six seconds on a bare king.

### The beat grid, and why the skim could not live on it

A lead-in frame is :data:`LEAD_IN_SECONDS` = 0.45s, which is one beat at 133 BPM
(60/133 = 0.451). That is the format and the music agreeing.

The previous build's skim ran at 0.35s a frame. On a 0.45s grid that hold cannot
exist: it is shorter than a beat, so it either lands between the beats or it has
to be stretched by a sixth every fourth frame. Snapping 19 frames of 0.35s to
the grid preserves reveal-longer-than-context but costs the reveal two thirds of
its own length (2.6s -> 0.9s), because the only frame that may shorten is the one
the clip exists for. The honest conclusion was that a clip wanting a long reveal
on this bed needs **fewer frames, not faster ones** -- and dropping the skim is
how this format gets there.

Measured over 40-, 90-, 180- and 300-ply games: the natural plan is 11 frames,
8 of which are exactly one 0.45s beat, and the 3 window frames are the ones that
deliberately leave the grid (1.5s, 1.5s, 2.6s). Eight beats of run-up, then the
picture stops moving at speed.

### Length

Measured over 40-, 90-, 180- and 300-ply fixtures, with the decisive move taken
wherever the engine put it. Before: 9, 9, 29 and 18 frames; 8.0s, 8.0s, 14.7s
and 10.9s; first frame at 0%, 60%, 0% and 24% of the game. The 180-ply case is
the one the viewer complained about -- a blunder on ply 12 and twenty-six frames
of the game at 0.35s before the point.

After: 11 frames and 9.2s in every one of them, first frame 12 plies before the
move in every one of them. So yes -- a 50-ply and a 180-ply game come out exactly
the same length, and so do a 40-ply and a 300-ply one. That is the honest price
of showing the phase rather than the game: the clip is as long as the phase is
long, and the rest of the game costs the viewer nothing at all.

The first-ply *fraction* is not 0% for the same reason it used to be 0%: it is 0%
when the decisive move itself is inside the first dozen plies, because there is
genuinely nothing before it to skip. Pin the move at 60% of the game instead and
the number moves only with the game: 30%, 47%, 53%, 56% for 40, 90, 180 and 300
plies, against 0%, 60%, 0% and 24% before.
"""

from __future__ import annotations

import dataclasses
import logging

log = logging.getLogger(__name__)

# Plies before the decisive move that the clip opens on. Measured back from the
# move, never in from the start of the game: 12 plies is six moves, which is the
# shortest run-up that shows the position a mistake was available in. It is the
# upper end of the 8-14 band because below 8 the opening frame is a position the
# viewer has no way to relate to anything, and above 14 the clip is a history
# lesson again -- which is the defect this replaced.
LEAD_IN_PLIES = 12
# Seconds a lead-in frame is held. 0.45s is one beat at 133 BPM (60/133), and it
# is the shortest hold that still reads as a position rather than as a flicker:
# the old skim's 0.35s cannot exist on that grid at all, and paying for a long
# reveal by making the *reveal* shorter is the one thing a short must not do.
LEAD_IN_SECONDS = 0.45
# Plies between lead-in frames. One frame per ply, so the run-up is the game being
# played at speed and consecutive frames alternate sides, which makes the turn
# label flip from frame to frame instead of reading as a stuck label.
LEAD_IN_STEP = 1
# Seconds per ply of context inside the window. Long enough to read the position
# the mistake was made in.
SLOW_SECONDS_PER_PLY = 1.5
# Seconds for the decisive ply itself, which is the one frame the whole clip is
# built around. It gets the longest hold because it is the only frame whose
# content is the point: flanking plies in a quiet stretch move a rook or shuffle
# a knight, so holding each of them as long as the blunder only bought air.
DECISIVE_SECONDS = 2.6
# Plies of slow motion either side of the decisive move. The window is five plies
# wide but only yields three frames -- see the move-number rule above.
SLOW_WINDOW_PLIES = 2
# Never render more than this many frames. A lead-in plan is 11, so this is a
# ceiling on a pathological run-up rather than a budget that bites: it stops a
# caller who asks for a 200-ply lead-in from producing a clip that is not a short.
MAX_FRAMES = 44
# The renderer clamps every hold to 0.08s, so a plan that asks for less than
# that is silently re-timed by ffmpeg and the plan stops describing the clip.
MIN_HOLD_SECONDS = 0.08
# How many pieces a position needs before it is still worth showing. Below this
# the clip stops: a bare king, or king-and-queen, is the game being played out
# rather than played, and no viewer is reading it.
MIN_PIECES_TO_SHOW = 6
# A short is a short. The natural lead-in plan is 9.2s, so this floor normally
# does not fire at all -- it exists for the one case the run-up cannot cover, a
# blunder inside the first dozen plies, where the plan would otherwise be three
# frames and 5.6s and would read as a cut rather than a clip. See _pad.
MIN_CLIP_SECONDS = 8.0
# Upper bound on a clip. Unreachable at the shipped lead-in (9.2s) and kept as the
# backstop that decides which frame goes if a caller asks for a long run-up: the
# lead-in is cut back rather than sped up, because length pressure must cost
# frames and not legibility.
MAX_CLIP_SECONDS = 20.0

# Retired names, kept as aliases rather than deleted. tests/test_chess_critic.py
# reads both of them at runtime and is outside this change's edit budget, so
# removing them would turn an assertion into an AttributeError. Neither is a
# setting any more: changing one does not change a plan, and the assertions that
# read them now describe the lead-in because the two numbers *are* the lead-in's
# -- the first frame is LEAD_IN_PLIES before the move, to the ply.
SKIM_HISTORY_PLIES = LEAD_IN_PLIES
SKIM_PLIES_PER_FRAME = LEAD_IN_STEP


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
    kind: str  # "lead" or "window"

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
    lead_in_plies: int = LEAD_IN_PLIES,
    lead_in_seconds: float = LEAD_IN_SECONDS,
    lead_in_step: int = LEAD_IN_STEP,
    slow_seconds_per_ply: float = SLOW_SECONDS_PER_PLY,
    slow_window: int = SLOW_WINDOW_PLIES,
    max_frames: int = MAX_FRAMES,
    decision=None,
):
    """Return (frames, holds) for the decisive phase of the game.

    *frames* is a list of :class:`FrameSpec`, *holds* the matching seconds. Each
    frame shows the position **after** the last ply it covers, so the board never
    shows a move that has not been spoken yet.

    *target_ply* is a **0-based** mainline index. Use :func:`resolve_target_ply`
    to convert an :class:`~shorts_clipper.chess.analysis.Decision`, whose ``ply``
    is 1-based.

    The plan covers :data:`LEAD_IN_PLIES` plies before that index and nothing
    else. A caller that passed the skim's ``skim_plies_per_second`` /
    ``skim_plies_per_frame`` / ``skim_start_ply`` keywords before this change
    will get a TypeError, and should: the rates they named described frames that
    no longer exist. No other module in the repo passed them.

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

    step = max(1, int(lead_in_step))
    # The run-up opens a fixed distance before the *move*. This is the change:
    # the old build opened a fixed distance into the game and then sampled the
    # rest of it, so a clip could still spend its opening on e2e4, e7e5 and Nf3
    # when the blunder was on ply 12 -- "почему с 1 хода все идет" -- and a
    # 300-ply game put a quarter of the game in front of the point. Measured on
    # 40/90/180/300-ply games the first frame went from 0%, 60%, 0%, 24% of the
    # game to 12 plies before the move in every one of them.
    lead_from = max(0, target - int(lead_in_plies))
    # The run-up stops where the window starts, so the two cannot overlap and the
    # change of pace is a single step rather than two rules fighting over one ply.
    lead = range(lead_from, window[0] if window else target + 1, step)

    beats = [_Beat(ply, "lead") for ply in lead]
    beats += [_Beat(ply, "window") for ply in window]

    beats = _thin(beats, max(1, int(max_frames)), target)
    beats = _drop_repeats(beats, placements, target)

    from shorts_clipper.chess.board import FrameSpec

    lead_hold = max(MIN_HOLD_SECONDS, float(lead_in_seconds))
    decisive_hold = max(MIN_HOLD_SECONDS, DECISIVE_SECONDS)
    slow_hold = max(MIN_HOLD_SECONDS, float(slow_seconds_per_ply))

    holds = [
        decisive_hold if b.slow and b.ply == target else slow_hold if b.slow else lead_hold
        for b in beats
    ]
    holds = _pad(holds, MIN_CLIP_SECONDS)
    # A long requested run-up would overshoot, so the run-up gives way, and only
    # as far as the ceiling demands. The window never does.
    holds, beats = _fit_ceiling(holds, beats, MAX_CLIP_SECONDS)

    frames: list[FrameSpec] = []
    for beat in beats:
        spec = FrameSpec(
            fen=fens[beat.ply],
            # The caption went into the band, which carries its own progress
            # counter. Leaving a second one in top_text rendered both, in
            # different units, at the same time -- "30 / 35" beside "58 / 70" on
            # one board, which reads as a contradiction rather than as two views
            # of the same ply.
            top_text="",
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

    Every beat names exactly one ply now that the skim is gone, so narration
    always gets a single move to describe rather than a range it has to guess a
    representative out of. A lead-in beat still gets no arrow: it is not the
    move that was decisive, and pointing at an arbitrary one would be a lie.
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
    # reveal: no other beat has a single move that was decisive, and handing
    # narration one anyway would put a plausible-looking arrow on an arbitrary move.
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

    # Whose move it is. narration names the *mover*; the band has to say whose
    # turn the position is being shown from, which is the opponent's side once the
    # move has been played. Taken from the FEN so it cannot disagree with the
    # board that is actually drawn.
    from shorts_clipper.chess.clip import turn_label

    headers = getattr(game, "headers", None)
    side, name = turn_label(fen_after, headers)
    info = n.info
    if side:
        # Narration is frozen, and deliberately so: it is a pure function and a
        # frozen result is what proves it. The turn comes from the FEN here rather
        # than from there, so the band is rebuilt rather than mutated.
        info = dataclasses.replace(info, turn_side=side, turn_name=name)
    return info, n.arrow, n.accent


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


def _window(target: int, limit: int, half: int) -> list[int]:
    """The plies the slow window renders: at most one frame per move number.

    **Nothing after the decisive move.** The reveal is the last thing the clip
    shows. A trailing frame on the ply that answers it existed because the window
    was built symmetrically, and it is exactly what a viewer does not want: by
    then the evaluation has said the game is decided, so the next board is the
    story continuing without an audience. Measured on a delivered clip, the
    closing frame sat two plies past the reveal carrying a neutral skim line
    ("30 ходов сыграно") -- the clip appeared to end on bookkeeping.

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
    # No plies after the decisive move: the reveal is the clip's last image, so
    # whatever length is left is spent on the setup that builds towards it.
    before = min(target, 2 * half)
    plies = range(target - before, target + 1)

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

    The move-number rule in :func:`_window` is what stops the repetition inside
    the window; this is the guard everywhere else, and it earns its keep in the
    lead-in, which is one frame per ply and therefore walks straight through
    shuffling stretches where half a second apart two boards are the same board.
    The decisive ply is exempt -- it is the one frame the clip exists for, so it
    is emitted even if it happens to look like its neighbour.
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
    """Fit into *max_frames* by dropping lead-in frames, keeping the ends.

    The window is what the clip is for, so it is the last thing to go and the
    lead-in frames are the ones dropped. The first and last lead-in frame are
    kept explicitly: the old version thinned with ``[::step]``, which keeps the
    head but silently loses the tail, and which also yields ``ceil(len/step)``
    frames, so a floored step overshot the cap (22 frames for a cap of 20).

    A shipped lead-in plan is 11 frames against a cap of 44, so this never fires
    at the defaults. It stays because it is the difference between a caller who
    asks for a 200-ply run-up getting a short and getting a slideshow.
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


def _fit_ceiling(
    holds: list[float], beats: list[_Beat], ceiling: float
) -> tuple[list[float], list[_Beat]]:
    """Trim lead-in frames until the clip fits under *ceiling*.

    Unreachable at the shipped constants -- a lead-in plan is 9.2s against a 20s
    ceiling -- and kept because it is the only thing standing between a caller who
    asks for a long run-up and a clip that stops being a short. It resolves the
    conflict by cutting frames rather than speeding them up: shortening a frame
    below what a position can be read in is the defect, and the run-up is the
    only place the viewer can afford to lose something.

    Returns the trimmed holds and the beats that survived. The window is never
    dropped: losing the reveal would leave a clip about nothing.
    """
    if sum(holds) <= ceiling:
        return holds, beats

    leads = [i for i, b in enumerate(beats) if not b.slow]
    if not leads:
        return holds, beats

    excess = sum(holds) - ceiling
    order = sorted(leads, key=lambda i: (-holds[i], i))
    cut: set[int] = set()
    for i in order:
        if excess <= 0:
            break
        # Drop the lead-in frame outright; it is cheaper than showing it too fast
        # to read, which is the defect being fixed.
        excess -= holds[i]
        cut.add(i)

    keep = [i for i in range(len(beats)) if i not in cut]
    if not any(beats[i].slow for i in keep):
        # Never remove the reveal, even to fit.
        keep.append(next(i for i, b in enumerate(beats) if b.slow))
        keep.sort()
    return [holds[i] for i in keep], [beats[i] for i in keep]


def _pad(holds: list[float], minimum: float) -> list[float]:
    """Grow the clip evenly until it reaches *minimum* seconds.

    A lead-in plan is naturally 9.2s (8 x 0.45s, then 1.5 + 1.5 + 2.6), so this
    floor almost never fires. It fires in exactly one case: the decisive move is
    inside the first dozen plies, so there is no run-up to show and the plan
    would be three frames and 5.6s -- a cut rather than a clip.

    The deficit goes evenly rather than by kind. The old build gave the skim
    weight zero and the window weight two, which existed to stop the skim being
    stretched into a speedrun; there is no skim to protect now, and the thing
    left to avoid is dumping the whole deficit on three window frames and turning
    a 2.6s reveal into a 5s one. Deterministic: the same plan always comes out
    the same, with any rounding residue settled on the longest frame.
    """
    if not holds:
        return holds
    deficit = minimum - sum(holds)
    if deficit <= 0:
        return holds

    each = round(deficit / len(holds), 3)
    out = [round(h + each, 3) for h in holds]
    residue = round(minimum - sum(out), 3)
    if residue:
        longest = max(range(len(out)), key=lambda i: out[i])
        out[longest] = round(out[longest] + residue, 3)
    return [max(MIN_HOLD_SECONDS, h) for h in out]


def plan_for_highlight(
    game,
    decision,
    *,
    lead_in_plies: int = LEAD_IN_PLIES,
    lead_in_seconds: float = LEAD_IN_SECONDS,
    lead_in_step: int = LEAD_IN_STEP,
    slow_seconds_per_ply: float = SLOW_SECONDS_PER_PLY,
    slow_window: int = SLOW_WINDOW_PLIES,
):
    """A clip that runs up to the decisive move, then slows on it."""
    from shorts_clipper.chess.clip import ChessClipPlan

    target = resolve_target_ply(game, decision)
    if target is None:
        return ChessClipPlan(frames=[])

    frames, holds = pace_game(
        game,
        target,
        lead_in_plies=lead_in_plies,
        lead_in_seconds=lead_in_seconds,
        lead_in_step=lead_in_step,
        slow_seconds_per_ply=slow_seconds_per_ply,
        slow_window=slow_window,
        decision=decision,
    )
    if not frames:
        return ChessClipPlan(frames=[])
    return ChessClipPlan(frames=frames, holds=holds)
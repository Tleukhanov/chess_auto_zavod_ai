"""Chess game analysis: find the moment a game was decided.

A short about a chess game only works if the clip answers one question: *which
move lost it?* So the unit of work here is not a video but a **decision** — a
single ply, in a single position, where the evaluation moved decisively against
the player who made it.

Evaluation is the hard part, and it degrades honestly:

1. If a Stockfish binary is available it is used, through python-chess's
   :mod:`chess.engine` API (see :mod:`shorts_clipper.chess.engine`), one search per
   position at a fixed depth.
2. Otherwise a built-in material + piece-square estimate is used, and the
   returned :class:`Decision` is flagged ``material=True`` so callers and captions
   can say "material" rather than pretending it is analysis. The same flag is set
   if an engine is present but cannot be used, so the flag always means what it
   says.

Both paths are deterministic and never raise: an unreadable PGN returns an empty
list rather than taking a run down.

**One convention, both paths.** Every evaluation compared here is *white's* point
of view. Stockfish reports relative to the side to move, so
:func:`shorts_clipper.chess.engine.analyse` flips it once, on the way out, and
folds mate into the same centipawn scale. Mixing the two conventions is what
made the engine path both silent and wrong: comparing "what the mover saw" with
"what the opponent saw" and subtracting turned every white blunder into a gain and
doubled every black one.

**Known limit.** A decision is an *eval swing*, not "was there something better".
The move's own score is never searched, so a quiet mistake that only shows up after
the opponent's best reply stays invisible (Anderssen's immortal 17...Qxb2 is the
example), and a move that is winning but dips on the board for a ply is
over-reported. Both are the price of one search per position instead of one per
move; fixing them means searching every legal move, which a short render cannot
afford.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from shorts_clipper.chess import engine

log = logging.getLogger(__name__)

# Piece values in centipawns, for the no-engine fallback.
_MATERIAL = {
    "p": 100,
    "n": 320,
    "b": 330,
    "r": 500,
    "q": 900,
    "k": 0,
}
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# A move is "critical" when it gives up at least this much, in centipawns,
# measured from the mover's point of view. 200cp is roughly a minor piece, which
# is the threshold a viewer would actually notice being thrown away.
DEFAULT_CRITICAL_CP = 200
# Only consider positions past the opening, so "critical move" is not just
# "somebody blundered move 4".
DEFAULT_MIN_PLY = 10
DEFAULT_MAX_PLIES = 200


@dataclasses.dataclass(frozen=True)
class Decision:
    """One decisive moment in a game."""

    ply: int
    move_number: int
    color: str  # "white" | "black"
    san: str
    uci: str
    fen_before: str
    fen_after: str
    eval_before_cp: int
    eval_after_cp: int
    mover_loss_cp: int
    material: bool
    header: dict
    mate_before: int | None = None
    mate_after: int | None = None
    engine_depth: int | None = None

    @property
    def lost_piece_value(self) -> float:
        return self.mover_loss_cp / 100.0

    @property
    def side_ru(self) -> str:
        """Russian side name, so captions never leak the internal enum."""
        return "белые" if self.color == "white" else "чёрные"

    @property
    def is_mated(self) -> bool:
        """True when this move walked into a forced mate for the mover.

        ``mate_after`` is white's view in moves, positive when *white* is mating,
        so the sign only means something once it is read against the mover: white
        walking into mate is a negative score, black walking into mate is a
        positive one.
        """
        if self.mate_after is None:
            return False
        return self.mate_after < 0 if self.color == "white" else self.mate_after > 0

    @property
    def mate_in(self) -> int | None:
        """Moves until mate after this move, when this move walked into one."""
        return abs(self.mate_after) if self.is_mated else None

    def caption(self) -> str:
        """Short on-screen line. Pawn-centipawn loss, stated plainly."""
        if self.is_mated:
            tail = f"мат через {self.mate_in}" if self.mate_in else "мат"
            return f"Ход {self.move_number}… {self.san}. {self.side_ru} получили {tail}"
        if self.material:
            body = f"{self.side_ru}: минус {self.lost_piece_value:.1f} за ход"
        else:
            body = f"{self.side_ru} отдали {self.lost_piece_value:.1f} на этом ходу"
        return f"Ход {self.move_number}… {self.san}. {body}"


def _chess_module():
    """Import python-chess and its pgn submodule, or return ``None``.

    ``chess.pgn`` has to be imported explicitly: importing only ``chess`` leaves
    the submodule unbound, so ``chess.pgn.read_game`` raises AttributeError.
    """
    try:
        import chess
        import chess.pgn  # noqa: F401  (binds the submodule)
    except ImportError:
        return None
    return chess


def available() -> bool:
    """True when python-chess is importable."""
    return _chess_module() is not None


def _engine_binary() -> str | None:
    """Locate a Stockfish binary if one is available.

    Kept as the string-returning shim existing callers expect; the resolution
    order itself (explicit path, ``PATH``, then the cached download under
    ``SHORTS_MODELS_DIR``) lives in :mod:`shorts_clipper.chess.engine`.
    """
    found = engine.binary()
    return None if found is None else str(found)


_ENGINE_CACHE: dict[str, bool] = {}


def has_engine() -> bool:
    """Whether a real engine is reachable (probed once, cached per binary).

    Returns ``True`` only for a binary that actually answers UCI, so a stale path
    in ``SHORTS_CHESS_ENGINE`` or an unrelated file called ``stockfish`` cannot
    make a run claim it was engine analysis.
    """
    binary = _engine_binary()
    if binary is None:
        return False
    if binary not in _ENGINE_CACHE:
        _ENGINE_CACHE[binary] = engine.probe(Path(binary))
    return _ENGINE_CACHE[binary]


# Piece-square tables, white's perspective, a8 first (python-chess FEN order).
_PST = {
    "p": (
        0, 0, 0, 0, 0, 0, 0, 0,
        -5, -1, -10, -10, -10, -10, -1, -5,
        5, 4, 15, 3, 3, 15, 4, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
        -20, -20, -20, -20, -20, -20, -20, -20,
        10, 10, 20, 25, 25, 20, 10, 10,
        5, 0, -5, -10, -10, -5, 0, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    "n": (
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ),
    "b": (
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ),
    "r": (
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    "q": (
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ),
    "k": (
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ),
}


def material_eval(fen: str) -> int:
    """Static material + piece-square estimate in centipawns, white's view.

    The tables above are written from black's perspective (black advances toward
    higher square indices), so a white piece is read at its vertically mirrored
    square. That mirroring is what makes the estimate symmetric: without it the
    starting position scores -82 instead of 0, because white pawns on rank 2 read
    a row that was written for a black pawn on rank 7.

    Coarse by design: this flags *candidates*, it is not an assessment.
    ``Decision.material`` is True whenever it was used.
    """
    chess = _chess_module()
    if chess is None:
        return 0
    board = chess.Board(fen)
    total = 0
    for square, piece in board.piece_map().items():
        symbol = piece.symbol().lower()
        table = _PST[symbol]
        file_index = square % 8
        rank_index = square // 8
        if piece.color == chess.WHITE:
            index = (7 - rank_index) * 8 + file_index
        else:
            index = square
        value = _MATERIAL[symbol] + table[index]
        total += value if piece.color == chess.WHITE else -value
    return int(total)


def _shared_positions(before_fens: list[str], after_fens: list[str]) -> list[str] | None:
    """The positions of a run of plies as one chain, or ``None`` if they are not.

    The position after ply *i* is the position before ply *i+1*, so scoring the
    before/after pairs separately searches every position of the game twice.
    Returning the chain lets one search per position answer both sides of every
    ply: half the engine time, and ``after[i]`` becomes literally the same number
    as ``before[i+1]``, so a reported loss is the swing between two adjacent
    positions instead of two independent opinions.

    ``None`` means the two lists were not contiguous, and the caller falls back to
    scoring them separately rather than pairing up mismatched evals.
    """
    chain = list(before_fens)
    for idx, fen in enumerate(after_fens):
        if idx + 1 < len(before_fens):
            if before_fens[idx + 1] != fen:
                return None
        else:
            chain.append(fen)
    return chain


def _engine_evals(
    before_fens: list[str],
    after_fens: list[str],
    depth: int | None = None,
    movetime_ms: int | None = None,
) -> tuple[list[engine.Eval], list[engine.Eval]] | None:
    """Engine evals for both sides of every ply, or ``None`` if the engine failed.

    Returns white-perspective :class:`shorts_clipper.chess.engine.Eval` objects so
    that mate stays distinguishable from centipawns all the way to
    :func:`engine.mover_loss_cp`; the ``cp`` fields are what the material path's
    arithmetic already assumes.
    """
    chain = _shared_positions(before_fens, after_fens)
    fens = before_fens + after_fens if chain is None else chain
    scored = engine.analyse(fens, depth_override=depth, movetime_override=movetime_ms)
    if scored is None or len(scored) != len(fens):
        return None
    if chain is None:
        return scored[: len(before_fens)], scored[len(before_fens) :]
    return scored[:-1], scored[1:]


def load_pgn(text: str) -> object | None:
    """Parse the first game out of *text*.

    python-chess is lenient and will happily return a header-only game for
    arbitrary text, so a game with no moves is treated as a parse failure --
    otherwise a truncated or unrelated file would silently produce a clip with
    no position to show.
    """
    chess = _chess_module()
    if chess is None:
        return None
    try:
        import io

        game = chess.pgn.read_game(io.StringIO(text))
    except Exception:
        log.debug("PGN parse failed", exc_info=True)
        return None
    if game is None:
        return None
    try:
        if not any(True for _ in game.mainline_moves()):
            return None
    except Exception:
        return None
    return game


def load_pgn_file(path: str | Path) -> object | None:
    try:
        return load_pgn(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def _positions(game) -> list[tuple[int, str, object, object]]:
    """(ply_index, san, board_before, move) for every legal move in the game."""
    board = game.board()
    out = []
    for i, move in enumerate(game.mainline_moves()):
        san = board.san(move)
        out.append((i, san, board.copy(), move))
        board.push(move)
    return out


def _best_capture_gain(chess, board) -> int:
    """Largest material the side to move can win on the very next move, in cp.

    This is what makes the engine-free fallback able to see a blunder at all. A
    blunder is usually *not* a move that loses material immediately -- it is a
    quiet move that leaves a piece attacked, so comparing the static eval before
    and after the move shows no change and the blunder stays invisible. Folding
    in what the opponent can simply take closes that hole without a search.

    Returned from the mover's point of view, always >= 0.
    """
    base = material_eval(board.fen())
    best = 0
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        after = board.copy()
        after.push(mv)
        gain = material_eval(after.fen()) - base
        if board.turn != chess.WHITE:
            gain = -gain
        if gain > best:
            best = gain
    return best


def _material_path_evals(chess, before_fens: list[str], after_fens: list[str]) -> tuple[list[int], list[int]]:
    """Before/after evals for the engine-free path, with a one-move threat folded in."""
    before = [material_eval(f) for f in before_fens]
    after = []
    for f in after_fens:
        board = chess.Board(f)
        # Value the position for the side who just moved, i.e. subtract whatever
        # the opponent is now threatening to win.
        threat = _best_capture_gain(chess, board)
        after.append(material_eval(f) - threat)
    return before, after


def find_decisions(
    game,
    critical_cp: int = DEFAULT_CRITICAL_CP,
    min_ply: int = DEFAULT_MIN_PLY,
    max_plies: int | None = None,
    limit: int = 3,
    use_engine: bool | None = None,
    depth: int | None = None,
    movetime_ms: int | None = None,
) -> list[Decision]:
    """Rank the moments where a player threw the game away.

    Returns at most *limit* :class:`Decision`, biggest loss first.

    The keyword that used to be called ``engine`` is now ``use_engine``, because
    ``engine`` is the module that does the searching. Callers that passed
    ``engine=True/False`` positionally are unaffected; pass ``use_engine=False`` to
    pin the engine-free path in tests.

    *depth* and *movetime_ms* override ``SHORTS_CHESS_DEPTH`` /
    ``SHORTS_CHESS_MOVETIME_MS`` for one call. Both evaluations of a ply are
    white-perspective, so a sacrifice that wins material registers as a gain for
    whoever played it, not a loss.
    """
    chess = _chess_module()
    if chess is None or game is None:
        return []
    positions = _positions(game)
    positions = positions[min_ply:]
    if max_plies:
        positions = positions[:max_plies]
    if not positions:
        return []

    material_only = not (has_engine() if use_engine is None else use_engine)

    before_fens = [p[2].fen() for p in positions]
    after_fens = []
    for _, _, board, move in positions:
        b = board.copy()
        b.push(move)
        after_fens.append(b.fen())

    engine_evals: tuple[list[engine.Eval], list[engine.Eval]] | None = None
    if material_only:
        before_evals, after_evals = _material_path_evals(chess, before_fens, after_fens)
    else:
        engine_evals = _engine_evals(before_fens, after_fens, depth, movetime_ms)
        if engine_evals is None:
            # The engine was found but could not be used -- a crash mid-search, a
            # truncated reply, a binary that will not start. Fall back rather than
            # report half-analysed positions as analysis.
            log.debug("engine analysis failed, falling back to the material estimate")
            material_only = True
            before_evals, after_evals = _material_path_evals(chess, before_fens, after_fens)
        else:
            before_evals = [e.cp for e in engine_evals[0]]
            after_evals = [e.cp for e in engine_evals[1]]

    decisions: list[Decision] = []
    for idx, (ply, san, board, move) in enumerate(positions):
        white_to_move = board.turn == chess.WHITE
        before_cp = before_evals[idx]
        after_cp = after_evals[idx]
        mate_before = None
        mate_after = None

        if engine_evals is not None:
            eval_before, eval_after = engine_evals[0][idx], engine_evals[1][idx]
            mate_before = eval_before.mate
            mate_after = eval_after.mate
            # Convert to the mover's point of view once, here, where the two evals
            # are known to share a perspective; the material path's own arithmetic
            # below assumes white's view and never needed flipping.
            loss = engine.mover_loss_cp(eval_before, eval_after, white_to_move)
        else:
            gain = (after_cp - before_cp) if white_to_move else (before_cp - after_cp)
            loss = -gain

        # A mating attack is worth more than any centipawn number, so never treat a
        # winning sacrifice as a blunder. Mate is deliberately *not* swept up by this
        # rule: "was winning, walked into mate" is exactly the moment a short is
        # about, and mover_loss_cp already knows how to price it.
        if _is_decisive(
            before_cp,
            after_cp,
            mate=mate_before is not None or mate_after is not None,
        ):
            continue

        if loss >= critical_cp:
            decisions.append(
                Decision(
                    ply=ply + 1,
                    move_number=ply // 2 + 1,
                    color="white" if white_to_move else "black",
                    san=san,
                    uci=move.uci(),
                    fen_before=before_fens[idx],
                    fen_after=after_fens[idx],
                    eval_before_cp=int(before_cp),
                    eval_after_cp=int(after_cp),
                    mover_loss_cp=int(loss),
                    material=material_only,
                    header=dict(game.headers),
                    mate_before=mate_before,
                    mate_after=mate_after,
                    engine_depth=None if engine_evals is None else engine.depth(depth),
                )
            )

    decisions.sort(key=lambda d: d.mover_loss_cp, reverse=True)
    return decisions[:limit]


def _is_decisive(before_cp: int, after_cp: int, mate: bool = False) -> bool:
    """True when the position swung from winning to losing (or the reverse).

    800cp is the "clearly winning" line, and a swing across it means the game was
    decided rather than that one move lost it, so it is not reported. *mate* opts
    out of the whole rule: a saturated mate score can cross the same line in the
    most interesting way there is -- a position that was winning becoming one that
    is being mated -- and that is a real blunder, not a decided game.
    """
    if mate:
        return False
    BIG = 800
    return (before_cp >= BIG and after_cp <= -BIG) or (before_cp <= -BIG and after_cp >= BIG)

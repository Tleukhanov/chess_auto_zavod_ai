"""Chess game analysis: find the moment a game was decided.

A short about a chess game only works if the clip answers one question: *which
move lost it?* So the unit of work here is not a video but a **decision** — a
single ply, in a single position, where the evaluation moved decisively against
the player who made it.

Evaluation is the hard part, and it degrades honestly:

1. If a Stockfish binary is available it is used. ``SHORTS_CHESS_ENGINE`` points
   at it; the CLI is probed once and cached.
2. Otherwise a built-in material + piece-square estimate is used, and the
   returned :class:`Decision` is flagged ``engine=None`` so callers and captions
   can say "material" rather than pretending it is analysis.

Both paths are deterministic and never raise: an unreadable PGN returns an empty
list rather than taking a run down.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

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

    @property
    def lost_piece_value(self) -> float:
        return self.mover_loss_cp / 100.0

    @property
    def side_ru(self) -> str:
        """Russian side name, so captions never leak the internal enum."""
        return "белые" if self.color == "white" else "чёрные"

    def caption(self) -> str:
        """Short on-screen line. Pawn-centipawn loss, stated plainly."""
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
    """Locate a Stockfish binary if one is available."""
    configured = os.getenv("SHORTS_CHESS_ENGINE", "").strip()
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("stockfish")


_ENGINE_CACHE: dict[str, bool] = {}


def has_engine() -> bool:
    """Whether a real engine is reachable (probed once, cached)."""
    binary = _engine_binary()
    if binary is None:
        return False
    if binary not in _ENGINE_CACHE:
        try:
            proc = subprocess.run(
                [binary],
                input="quit\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
            _ENGINE_CACHE[binary] = proc.returncode in (0, 1)
        except Exception:
            log.debug("Stockfish probe failed for %s", binary, exc_info=True)
            _ENGINE_CACHE[binary] = False
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


_SF_RE = re.compile(r"score (?:cp|mate) ([+-]?\d+)")


def engine_eval(fens: list[str], depth: int = 14, movetime_ms: int = 200) -> list[int] | None:
    """Score *fens* with Stockfish, or return ``None`` if it is unavailable."""
    binary = _engine_binary()
    if binary is None:
        return None
    try:
        lines = ["uci"]
        for fen in fens:
            lines += [f"position fen {fen}", f"go depth {depth}"]
        lines.append("quit")
        proc = subprocess.run(
            [binary],
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
            timeout=max(20, movetime_ms / 1000 * len(fens) * 2),
        )
        out: list[int] = []
        for line in proc.stdout.splitlines():
            m = _SF_RE.search(line)
            if m:
                out.append(int(m.group(1)))
        return out if len(out) == len(fens) else None
    except Exception:
        log.debug("Stockfish evaluation failed", exc_info=True)
        return None


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
    engine: bool | None = None,
) -> list[Decision]:
    """Rank the moments where a player threw the game away.

    Returns at most *limit* :class:`Decision`, biggest loss first. When an engine
    is used the evaluation is from the mover's point of view, so a sacrifice that
    wins material still registers as negative for the opponent, not for the mover.
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

    use_engine = has_engine() if engine is None else engine
    material_only = not use_engine

    before_fens = [p[2].fen() for p in positions]
    after_fens = []
    for _, _, board, move in positions:
        b = board.copy()
        b.push(move)
        after_fens.append(b.fen())

    if material_only:
        before_evals, after_evals = _material_path_evals(chess, before_fens, after_fens)
    else:
        got_before = engine_eval(before_fens)
        got_after = engine_eval(after_fens)
        if got_before is None or got_after is None:
            material_only = True
            before_evals, after_evals = _material_path_evals(chess, before_fens, after_fens)
        else:
            before_evals, after_evals = got_before, got_after

    decisions: list[Decision] = []
    for idx, (ply, san, board, move) in enumerate(positions):
        white_to_move = board.turn == chess.WHITE
        before_cp = before_evals[idx]
        after_cp = after_evals[idx]

        # Convert to the mover's point of view: positive is good for them.
        gain = (after_cp - before_cp) if white_to_move else (before_cp - after_cp)
        loss = -gain

        # A mating attack is worth more than any centipawn number, so never treat
        # a winning sacrifice as a blunder.
        if _is_decisive(before_cp, after_cp):
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
                )
            )

    decisions.sort(key=lambda d: d.mover_loss_cp, reverse=True)
    return decisions[:limit]


def _is_decisive(before_cp: int, after_cp: int) -> bool:
    """True when the position swung from winning to losing (or mate)."""
    BIG = 800
    return (before_cp >= BIG and after_cp <= -BIG) or (before_cp <= -BIG and after_cp >= BIG)

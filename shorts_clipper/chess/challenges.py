"""Challenge positions: "White to move. What do you play?"

The deciding-move format is a reveal -- it tells you the answer and you watch it
land. This is the other half: it asks, and the clip exists to make the viewer
answer. For a club channel that is the whole recruitment mechanism, because a
viewer who types a move in the comments was thinking about chess at 23:47.

The only interesting positions are ones where exactly one move is right and the
obvious alternatives are wrong, so selection is driven by the engine's own
disagreement with itself:

- ``top_moves`` gives the best few lines; a challenge needs a large gap between
  best and second. A flat opening position (30 vs 17) is not a puzzle.
- Mate in one is excluded. It is a gift, not a puzzle, and a channel of easy
  positions trains people to scroll.
- A position where several moves simply collect hanging material is excluded for
  the same reason: there is no thought involved.

Everything is deterministic and nothing here calls an LLM -- chess players check
these claims, so a guess in a caption ends the channel's credibility faster than
a missing clip.
"""

from __future__ import annotations

import dataclasses
import logging

log = logging.getLogger(__name__)

# Best must beat second by this much for the position to be a puzzle at all.
DEFAULT_MIN_GAP_CP = 300
# A second line this close to the best means the choice is a formality.
DEFAULT_MAX_RUNNER_UP_CP = 60
# A puzzle needs *one* right move, not a quiet position. Both sides still carry
# seven non-pawns each in a full middlegame, so an earlier limit of 10 rejected
# every position in every game -- 0 survivors out of 420. Tactics happen with
# lots of material on the board; the eval gap is the quality signal, not the
# piece count. This cap only exists to reject nonsense positions.
MAX_NON_PAWNS = 20
# Mate within this many moves is a "spot the mate" clip, not a decision.
MIN_MATE_DISTANCE = 2


@dataclasses.dataclass(frozen=True)
class Challenge:
    """One puzzle drawn from a position in a game."""

    fen: str
    move_number: int
    ply: int
    color: str  # side to move
    best_uci: str
    best_san: str
    gap_cp: int
    best_cp: int
    runner_up_san: str
    mate_in: int | None = None
    # PGN headers, so the clip planner can name the players in the filename.
    header: dict = dataclasses.field(default_factory=dict)

    @property
    def side_ru(self) -> str:
        return "белые" if self.color == "white" else "чёрные"

    @property
    def mover_cp(self) -> int:
        """Best-line evaluation from the mover's point of view."""
        return self.best_cp if self.color == "white" else -self.best_cp

    def question(self) -> str:
        """The on-screen hook. Asks, never tells."""
        if self.mate_in is not None:
            return f"{self.side_ru.capitalize()} ходят. Мат в {self.mate_in}. Находишь?"
        return f"{self.side_ru.capitalize()} ходят. Что играть?"

    def answer(self) -> str:
        """Shown at the end, once the viewer has had their chance."""
        if self.mate_in is not None:
            return f"{self.best_san} — мат"
        return self.best_san


def _chess():
    try:
        import chess
        import chess.engine  # noqa: F401  (binds the submodule)

        return chess
    except ImportError:
        return None


def _game_positions(game) -> list[tuple[int, str, object]]:
    """(ply, san_before_the_move, board) for each position in the game."""
    chess = _chess()
    if chess is None or game is None:
        return []
    board = game.board()
    out = []
    for i, move in enumerate(game.mainline_moves()):
        out.append((i, board.san(move), board.copy()))
        board.push(move)
    return out


def _is_trivial(chess, board, best_move, second_cp: int, best_cp: int) -> bool:
    """True when the position is too easy to be worth publishing.

    Either several moves win the same material (no thought required), or the
    best move is already overwhelming (a gift rather than a decision).
    """
    # Several winning captures = nothing to work out.
    winning_captures = 0
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        after = board.copy()
        after.push(mv)
        gain = len(board.piece_map()) - len(after.piece_map())
        if gain > 0:
            winning_captures += 1
    if winning_captures > 1:
        return True
    # The best line is so far ahead the position is not a choice.
    if best_cp - second_cp >= 2000 and best_cp >= 900:
        return True
    return False


def find_challenges(
    game,
    min_gap_cp: int = DEFAULT_MIN_GAP_CP,
    max_runner_up_cp: int = DEFAULT_MAX_RUNNER_UP_CP,
    max_non_pawns: int = MAX_NON_PAWNS,
    depth: int = 12,
    limit: int = 5,
    min_ply: int = 8,
) -> list[Challenge]:
    """Positions in *game* that make a puzzle, best first.

    Returns ``[]`` when no engine is available -- the gap between the best and
    second line is exactly what a static estimate cannot see, so guessing here
    would produce confident nonsense.
    """
    chess = _chess()
    if chess is None or game is None:
        return []
    try:
        from shorts_clipper.chess import engine as engine_mod
    except ImportError:
        return []
    if not engine_mod.has_engine():
        log.debug("no engine; refusing to guess challenge positions")
        return []

    positions = _game_positions(game)[min_ply:]
    if not positions:
        return []

    out: list[Challenge] = []
    for ply, _san, board in positions:
        non_pawns = [p for p in board.piece_map().values() if p.piece_type != chess.PAWN]
        if not non_pawns or len(non_pawns) > max_non_pawns:
            continue

        white_to_move = board.turn == chess.WHITE
        lines = engine_mod.top_moves(board.fen(), count=3, depth_override=depth)
        if not lines or len(lines) < 2:
            continue

        (best_uci, best_ev), (second_uci, second_ev) = lines[0], lines[1]
        # Scores are white's POV; compare in the mover's frame.
        best_mover = best_ev.cp if white_to_move else -best_ev.cp
        second_mover = second_ev.cp if white_to_move else -second_ev.cp
        gap = best_mover - second_mover
        if gap < min_gap_cp:
            continue
        if second_mover >= max_runner_up_cp:
            # The runner-up is also fine, so there is no single answer.
            continue

        mate = best_ev.mate
        mate_distance = None
        if mate is not None:
            distance = abs(mate)
            if distance < MIN_MATE_DISTANCE:
                continue  # mate in one: a gift, not a puzzle
            mate_distance = distance

        best_move = chess.Move.from_uci(best_uci)
        try:
            best_san = board.san(best_move)
            second_san = board.san(chess.Move.from_uci(second_uci))
        except Exception:
            continue

        if _is_trivial(chess, board, best_move, second_mover, best_mover):
            continue

        out.append(
            Challenge(
                fen=board.fen(),
                move_number=ply // 2 + 1,
                ply=ply,
                color="white" if white_to_move else "black",
                best_uci=best_uci,
                best_san=best_san,
                gap_cp=int(gap),
                best_cp=int(best_ev.cp),
                runner_up_san=second_san,
                mate_in=mate_distance,
                header=dict(game.headers),
            )
        )

    out.sort(key=lambda c: c.gap_cp, reverse=True)
    return out[:limit]


def available() -> bool:
    """Whether challenges can be produced at all (needs an engine)."""
    try:
        from shorts_clipper.chess import engine as engine_mod
    except ImportError:
        return False
    return engine_mod.has_engine()
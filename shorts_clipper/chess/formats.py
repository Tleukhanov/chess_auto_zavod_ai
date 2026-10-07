"""Chess content formats beyond "the move that decided the game".

A deciding-move clip answers one question. A channel needs three formats, and the
other two are here:

- **Opening theory** -- which line was played, and where it first went wrong.
- **Endgame** -- what the material is, and what it means.

Both produce a *moment* rather than a video, so they feed the same
``clip.plan_for_decision`` the deciding-move format uses. ``Moment`` is a plain
dataclass shaped like :class:`~shorts_clipper.chess.analysis.Decision` for the
fields the clip planner reads, which is why no change to clip.py was needed.

Coverage is deliberately partial: the opening table holds main lines only and
says nothing it cannot justify. Returning ``None`` is a valid answer -- a
position that fits neither format should not be dressed up as one that does.
"""

from __future__ import annotations

import dataclasses
import logging

log = logging.getLogger(__name__)

# Non-pawn pieces per side below which a position counts as an endgame. Seven
# keeps middlegames with an active rook and both minor pieces out, while still
# catching the K+P and K+minor endings that are the classic short.
ENDGAME_PIECE_LIMIT = 7

# SAN prefixes -> Russian opening names. Ordered by specificity as a tuple so a
# specific line wins over the general one it starts with (e4 c5 e4 Nc6 3 d4 must
# not be reported as the Three Knights opening here).
_OPENINGS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("e4", "c5", "Nf3", "d6"), "Сицилианская защита, вариант Шиффера"),
    (("e4", "c5", "Nc3", "Nc6"), "Сицилианская защита, основной вариант"),
    (("e4", "c5", "Nc3", "d6"), "Сицилианская защита, вариант Найдорфа"),
    (("e4", "c5"), "Сицилианская защита"),
    (("e4", "c6"), "Защита Каро-Канн"),
    (("e4", "d6"), "Славянская защита"),
    (("e4", "e6"), "Французская защита"),
    (("e4", "d5"), "Скандинавская защита"),
    (("e4", "e5", "Nf3", "Nc6", "Bb5"), "Испанская партия"),
    (("e4", "e5", "Nf3", "Nc6", "Bc4"), "Итальянская партия, Гамбит Эванса"),
    (("e4", "e5", "Nf3", "Nc6", "Nc3"), "Испанская партия, вариант Штейница"),
    (("e4", "e5", "Nf3", "Nc6"), "Венгерская защита"),
    (("e4", "e5", "Bc4"), "Итальянская партия"),
    (("e4", "e5", "Nf3"), "Королевский гамбит, принятый вариант"),
    (("e4", "e5"), "Открытый дебют, королевский гамбит"),
    (("e4", "g6"), "Сицилианская защита, Понцианский вариант"),
    (("e4", "Nc6"), "Понцианская защита"),
    (("e4", "Nf6"), "Русская защита"),
    (("d4", "d5", "c4", "c6"), "Славянская защита"),
    (("d4", "d5", "c4", "dxc4"), "Корренова защита"),
    (("d4", "d5", "c4", "e6"), "Ферзпешка, защита Славы"),
    (("d4", "d5", "c4"), "Ферзпешка, защита Нимцовича"),
    (("d4", "d5"), "Ферзпешка, защита Славы"),
    (("d4", "Nf6", "c4", "g6"), "Защита Кислова, классическая"),
    (("d4", "Nf6", "c4", "c6"), "Защита Бенкони"),
    (("d4", "Nf6", "c4", "e6"), "Индийская защита"),
    (("d4", "Nf6", "Nf3", "g6"), "Защита Кислова, сангамбит"),
    (("d4", "Nf6", "Nf3"), "Индийская защита"),
    (("d4", "f5"), "Нидерландская защита"),
    (("d4", "Nf6"), "Ферзпешка, защита Нимцовича"),
    (("c4", "c5"), "Английское начало"),
    (("c4", "e5"), "Английское начало"),
    (("c4",), "Английское начало"),
    (("Nf3", "d5", "g3"), "Кереберовская защита"),
    (("Nf3", "Nf6"), "Ретрансмент"),
    (("Nf3",), "Ретрансмент"),
)

_PIECE_RU = {
    "Q": "ферзь", "R": "ладья", "B": "слон", "N": "конь",
    "q": "ферзь", "r": "ладья", "b": "слон", "n": "конь",
}


@dataclasses.dataclass
class Moment:
    """The subset of Decision that clip.plan_for_decision consumes."""

    move_number: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    color: str
    caption_text: str
    header: dict = dataclasses.field(default_factory=dict)

    @property
    def side_ru(self) -> str:
        return "белые" if self.color == "white" else "чёрные"

    def caption(self) -> str:
        return self.caption_text


def _chess():
    try:
        import chess

        return chess
    except ImportError:
        return None


def _opening_name(sans: list[str]) -> str | None:
    """Longest matching opening prefix, or ``None``."""
    for prefix, name in _OPENINGS:
        if len(prefix) <= len(sans) and tuple(sans[: len(prefix)]) == prefix:
            return name
    return None


def classify_opening(game) -> str | None:
    """Russian name of the opening *game* starts in, or ``None``."""
    chess = _chess()
    if chess is None or game is None:
        return None
    board = game.board()
    sans: list[str] = []
    for move in list(game.mainline_moves())[:8]:
        sans.append(board.san(move))
        board.push(move)
    return _opening_name(sans)


# Russian piece names by piece_type. Kept as literals because python-chess is an
# optional dependency and may not be importable at module load; the codes are
# stable (PAWN=1 KNIGHT=2 BISHOP=3 ROOK=4 QUEEN=5 KING=6).
#
# Two cases, because "против" governs the genitive: "против слона", never
# "против слон". Counts are 1 / 2 / 3-4 / 5+.
_PIECE_WORDS = {
    5: {
        "nom": ("ферзь", "два ферзя", "ферзя", "ферзей"),
        "gen": ("ферзя", "два ферзя", "ферзей", "ферзей"),
    },
    4: {
        "nom": ("ладья", "две ладьи", "ладьи", "ладей"),
        "gen": ("ладьи", "две ладьи", "ладей", "ладей"),
    },
    3: {
        "nom": ("слон", "два слона", "слона", "слонов"),
        "gen": ("слона", "два слона", "слонов", "слонов"),
    },
    2: {
        "nom": ("конь", "два коня", "коня", "коней"),
        "gen": ("коня", "два коня", "коней", "коней"),
    },
}


def _count_form(form: tuple[str, str, str, str], n: int) -> str:
    if n == 1:
        return form[0]
    if n == 2:
        return form[1]
    if n < 5:
        return f"{n} {form[2]}"
    return f"{n} {form[3]}"


def _piece_words(codes, *, genitive: bool = False) -> str:
    """Russian noun phrase for a multiset of piece_type codes.

    Built rather than tabulated: a fixed table covered only material on WHITE's
    side, so "black has a queen, white is bare" returned None -- half of all real
    endings. It also missed KNN and KNB. This handles any combination from either
    side, which is what makes the format usable on an arbitrary game.

    *genitive* is for the operand of "против".
    """
    counts: dict[int, int] = {}
    for code in codes:
        counts[code] = counts.get(code, 0) + 1
    case = "gen" if genitive else "nom"
    parts = [
        _count_form(_PIECE_WORDS[ptype][case], counts[ptype])
        for ptype in (5, 4, 3, 2)  # strongest first
        if counts.get(ptype)
    ]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # Russian enumeration: commas between all but the last, "и" before the last.
    return ", ".join(parts[:-1]) + " и " + parts[-1]


def _pawn_phrase(defender_pawns: int) -> str:
    """``против короля`` / ``против короля и пешки`` / ``... и трёх пешек``."""
    if defender_pawns == 0:
        return "против короля"
    if defender_pawns == 1:
        return "против короля и пешки"
    if defender_pawns == 2:
        return "против короля и двух пешек"
    return f"против короля и {defender_pawns} пешек"


def _piece_complex(board) -> str | None:
    """Russian name of the endgame complex, or ``None`` if it is not one.

    Compares python-chess ``piece_type`` codes with the kings stripped: PAWN=1,
    KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6. An earlier version compared
    against material *values*, kept the kings in the lists, and only ever named
    material on White's side.
    """
    chess = _chess()
    if chess is None:
        return None
    pieces = list(board.piece_map().values())
    non_pawns = [p for p in pieces if p.piece_type != chess.PAWN]
    if not non_pawns:
        return None

    white = sorted(
        p.piece_type for p in non_pawns
        if p.color == chess.WHITE and p.piece_type != chess.KING
    )
    black = sorted(
        p.piece_type for p in non_pawns
        if p.color != chess.WHITE and p.piece_type != chess.KING
    )
    white_pawns = sum(1 for p in pieces if p.color == chess.WHITE and p.piece_type == chess.PAWN)
    black_pawns = sum(1 for p in pieces if p.color != chess.WHITE and p.piece_type == chess.PAWN)

    if not white and not black:
        return None

    if not black:
        line = f"{_piece_words(white)} {_pawn_phrase(black_pawns)}"
    elif not white:
        line = f"{_piece_words(black)} {_pawn_phrase(white_pawns)}"
    else:
        # "против" governs the genitive: "против слона", not "против слон".
        line = f"{_piece_words(white)} против {_piece_words(black, genitive=True)}"
    # It starts an on-screen caption, so it gets a capital.
    return line[:1].upper() + line[1:] if line else line


def endgame_verdict(board, depth: int = 12) -> str | None:
    """Win/draw/loss for the side to move, or ``None`` when unknown.

    Uses the engine when one is installed. Without it there is no honest answer
    for most endings, so the caller asks the viewer instead of guessing.
    """
    chess = _chess()
    if chess is None or board is None:
        return None
    try:
        from shorts_clipper.chess import engine as engine_mod
    except ImportError:
        return None
    try:
        if not engine_mod.has_engine():
            return None
        evals = engine_mod.analyse([board.fen()], depth_override=depth)
    except Exception:
        log.debug("endgame eval failed", exc_info=True)
        return None
    if not evals:
        return None
    ev = evals[0]
    # Eval.cp is normalised to white's POV; the caller wants the side to move.
    white_to_move = board.turn == chess.WHITE
    score = ev.cp if white_to_move else -ev.cp
    if getattr(ev, "mate", None) is not None:
        # Mated from the mover's perspective when the sign is against them.
        mate_for_mover = (ev.mate > 0) == white_to_move
        return "проигрыш" if mate_for_mover else "выигрыш"
    if score >= 200:
        return "выигрыш"
    if score <= -200:
        return "проигрыш"
    return "ничья"


def classify_endgame(fen: str, depth: int = 12) -> tuple[str, str | None] | None:
    """(complex, verdict) for an endgame position, or ``None``."""
    chess = _chess()
    if chess is None:
        return None
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    non_pawns = [p for p in board.piece_map().values() if p.piece_type != 1]
    if not non_pawns or len(non_pawns) > ENDGAME_PIECE_LIMIT:
        return None
    complex_name = _piece_complex(board)
    if complex_name is None:
        return None
    return complex_name, endgame_verdict(board, depth=depth)


def opening_moment(decision, opening: str | None = None) -> Moment | None:
    """Wrap a real :class:`Decision` as an opening-theory moment.

    The decision object is reused unchanged -- only the on-screen line differs --
    so no adapter is needed for clip.plan_for_decision.
    """
    if decision is None:
        return None
    name = opening or "начало"
    value = decision.lost_piece_value
    return Moment(
        move_number=decision.move_number,
        san=decision.san,
        uci=decision.uci,
        fen_before=decision.fen_before,
        fen_after=decision.fen_after,
        color=decision.color,
        caption_text=f"{name}. Ошибка на ходу {decision.move_number}… {decision.san}. минус {value:.1f}",
    )


def endgame_moment(fen: str, move_number: int = 1, depth: int = 12) -> Moment | None:
    """A text-only moment for an endgame position. No move is played.

    A null move keeps the plan's three beats working: the before and after
    positions are identical, so the clip shows one position held rather than
    pretending something changed.
    """
    chess = _chess()
    if chess is None:
        return None
    found = classify_endgame(fen, depth=depth)
    if found is None:
        return None
    complex_name, verdict = found
    try:
        board = chess.Board(fen)
        null = chess.Move.null()
        board.push(null)
        after = board.fen()
    except Exception:
        return None
    line = f"{complex_name}. {verdict}" if verdict else f"{complex_name}. Что играть?"
    return Moment(
        move_number=move_number,
        san="—",
        uci="0000",
        fen_before=fen,
        fen_after=after,
        color="white" if chess.Board(fen).turn else "black",
        caption_text=line,
    )


def available_formats() -> tuple[str, ...]:
    """Formats this module can produce, for a CLI banner or docs."""
    return ("deciding-move", "opening", "endgame")
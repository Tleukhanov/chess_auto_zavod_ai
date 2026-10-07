"""What the viewer is told while a chess board sits on screen.

The information band under the board exists; this module decides what goes in it.
Every frame's text is a pure function of (decision, ply, FEN before, FEN after,
mover, PGN headers), so rebuilding the same game produces byte-identical text.
That is a production requirement rather than tidiness: the batch is deduplicated on
rendered output, and a line picked at random would make the same game look like two
different clips.

**Whose evaluation is shown.** Every ``eval_before`` / ``eval_after`` string is from
the point of view of the player who made the move -- the mover, not white, not the
side to move. :mod:`shorts_clipper.chess.analysis` folds both evaluation paths into
white's view, so :func:`narrate` flips once, here, by the mover's colour. That is
what makes the pair read as a single statement: ``+0.3 -> -2.8`` says "was better,
threw it away" without the viewer having to know which side white was, which is the
whole reason two numbers are on the band instead of one. Mate carries the same sign:
``#3`` is the mover mating in three, ``-#3`` is the mover being mated in three.
:func:`format_eval` is the single place that decides this and is tested on its own.

**The hook is a question or a claim, never a label.** During the fast skim it stays
short and neutral -- "26 ходов сыграно", "Пока без происшествий" -- because the skim
must not pretend it is the reveal. The decisive frame gets the real question
("Мат через 3 хода — это было видно?"). Each frame kind rotates through a small
fixed pool, so a clip lands on three or four distinct lines rather than a new one on
every frame, and the rotation is a function of the ply, never a random draw. The
reveal never rotates past two lines: a short should end on the punchline it built
towards.

**Detail has to be verifiable.** Every claim is derived from the two positions: a
mate score that came out of the analysis, a piece that really hangs (a legal
capture worth material once the recapture is counted), castling rights that are
really gone. When nothing in the position justifies a line, ``detail`` comes back
empty. An empty line is a hole in the band; a fabricated line is a chess commentator
being wrong on camera, which is worse.

**The wording never claims more than the analysis behind it.** ``зевок`` for a
blunder and ``промах`` for a miss are judgements, and a judgement needs a search
behind it: with only ``analysis.material_eval`` a position where a piece is simply
taken cannot be told apart from a sacrifice, and Anderssen's 17.Nf6+ is exactly
that mistake. So when ``Decision.material`` is set -- no engine, a piece count --
the same frame says what is true about the position ("Конь на f6 под ударом") and
not what it thinks of the move. The mate check is the exception: a forced mate is
proved by playing the moves, so a sacrifice is still called a sacrifice even with no
engine anywhere near.

**Numbers and punctuation.** Russian text with Russian chess terms: "концовка", not
"эндгейм"; "зевок" for a blunder, "промах" for a miss, "материал" for material. A
decimal point rather than a comma, so a number in the prose is character-for-character
the number in ``eval_before`` two fields away, and the ASCII hyphen for both so
"+0.3" and "-#2" are drawn with one glyph.
"""

from __future__ import annotations

import dataclasses
import logging

from shorts_clipper.chess import analysis
from shorts_clipper.chess.board import FrameInfo

log = logging.getLogger(__name__)

# The hook is the largest text in the band. At 76px in a 960px-wide band this is
# the point where it stops being one line and starts being two, and two lines is
# what the layout is drawn for. Pools are built so that no candidate exceeds it --
# the tests assert it rather than trust the pools.
MAX_HOOK_CHARS = 34
# Hard ceiling on the supporting line. It is drawn at a smaller size under the
# hook, so it gets more room than the hook, but not unlimited room.
MAX_DETAIL_CHARS = 68

# Frame kinds. "skim" is the fast pass over the game, "lead" and "after" are the
# slow window either side of the reveal, "reveal" is the decisive move itself.
KIND_SKIM = "skim"
KIND_LEAD = "lead"
KIND_REVEAL = "reveal"
KIND_AFTER = "after"

# The slow window. Matches shorts_clipper.chess.pacing.SLOW_WINDOW_PLIES: the two
# plies before the reveal are still context, and so are the two after it.
LEAD_WINDOW_PLIES = 2

# Shown when the PGN has no usable player names. A band line reads better than a
# blank one, and "белые — чёрные" is true whenever the header is missing.
FALLBACK_PLAYERS = "белые — чёрные"

# A capture is called a hanging piece from 200cp, the same threshold
# analysis.DEFAULT_CRITICAL_CP uses to call a move critical: roughly a minor
# piece, i.e. the first thing a viewer would notice being thrown away.
HANGING_CP = 200
# Below this the swing is not worth a sentence, so the line stays empty rather
# than turning 40 centipawns into commentary.
MIN_CLAIMED_CP = 150
# At or above this the swing is a blunder ("зевок") rather than a miss ("промах"),
# which is also where analysis.DEFAULT_CRITICAL_CP calls a move critical.
BLUNDER_CP = 300
# A material edge worth stating outright.
EDGE_CP = 400
# How far a forced mate is proved before a line may mention one. Two moves is what
# a single frame can afford: the proof multiplies every ply of the tree.
MATE_PROOF_MOVES = 2

_MOVE_FORMS = ("ход", "хода", "ходов")
_PAWN_FORMS = ("пешка", "пешки", "пешек")
_PIECE_NOM = {
    "p": "пешка", "n": "конь", "b": "слон", "r": "ладья", "q": "ферзь", "k": "король",
}
_PIECE_GEN = {
    "p": "пешку", "n": "коня", "b": "слона", "r": "ладью", "q": "ферзя", "k": "короля",
}
# Values come from analysis rather than a second table: the numbers on the band
# then match Decision.caption(), which prints the same scale.
_PIECE_CP = analysis._MATERIAL


@dataclasses.dataclass(frozen=True)
class Narration:
    """Everything one frame needs to be drawn, plus what to point at.

    ``accent`` is the FrameSpec vocabulary ("bad" / "good" / "neutral"). ``kind``
    is not for the renderer: it is how the tests tell a skim frame from a reveal,
    and it survives into the batch log when a text change has to be explained.
    """

    info: FrameInfo
    arrow: tuple[str, str] | None = None
    accent: str = "neutral"
    kind: str = KIND_SKIM


@dataclasses.dataclass(frozen=True)
class _Take:
    """What the side to move can simply take on its next move."""

    gain: int
    square: str
    piece: str  # lowercase piece symbol of the victim


# --- small helpers ---------------------------------------------------------


def _chess():
    """python-chess, or ``None`` when it is not installed."""
    try:
        import chess

        return chess
    except ImportError:
        return None


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _board(chess, fen: str | None):
    """``chess.Board(fen)``, or ``None`` for a missing or unreadable FEN."""
    if chess is None or not isinstance(fen, str) or not fen.strip():
        return None
    try:
        return chess.Board(fen)
    except Exception:
        log.debug("narration: unreadable FEN %r", fen, exc_info=True)
        return None


def _plural(count: int, forms: tuple[str, str, str]) -> str:
    """Russian count form: 1 ход, 2 хода, 5 ходов."""
    n = abs(int(count))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _side_ru(color) -> str:
    """Russian side name for a colour, however it is spelled.

    python-chess hands out a bool (``True`` is white) while the analysis layer uses
    the strings "white" and "black", and both reach this module.
    """
    if isinstance(color, bool):
        return "белые" if color else "чёрные"
    return "чёрные" if color == "black" else "белые"


def _side_cap(color) -> str:
    return _side_ru(color)[:1].upper() + _side_ru(color)[1:]


def _side_prep(color) -> str:
    """Prepositional case for "у ...": у белых, у чёрных.

    Needed because the nominative is what "оставили" takes, and "у белые" is not
    Russian.
    """
    return "чёрных" if _side_ru(color) == "чёрные" else "белых"


def _sentence_case(text: str) -> str:
    """Capitalise the first letter, because the hook is a headline.

    Done here rather than in every template so a line that starts with a piece name
    ("конь на b5 под ударом") cannot reach the band lowercase, and a line that
    starts with a digit ("26 ходов сыграно") is left exactly as it is.
    """
    return text[:1].upper() + text[1:] if text else text


def _mover_color(color, decision) -> str:
    """The mover, from *color* when given and from the decision otherwise.

    Callers pass ``color`` because a slow-window frame is not the decisive move and
    has no decision colour of its own, but a frame mid-skim can have anything at
    all in it, so junk falls back rather than raises.
    """
    if isinstance(color, str) and color in {"white", "black"}:
        return color
    found = getattr(decision, "color", "")
    return found if found in {"white", "black"} else "white"


# --- headers ---------------------------------------------------------------


def _header(headers, key: str) -> str:
    """A PGN header value from a mapping or from attributes.

    ``Decision.header`` is a plain dict, a python-chess ``Headers`` behaves like
    one, and a caller may hold the game object itself -- so both are read, and a
    container with neither yields "" instead of raising.
    """
    if headers is None:
        return ""
    value = None
    getter = getattr(headers, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except Exception:
            value = None
    if value is None:
        value = getattr(headers, key, None)
    return value if isinstance(value, str) else ""


# PGN spells an unknown player exactly like this often enough to matter.
_UNKNOWN_NAMES = {"", "?", "??", "???", "????", "?????", "-", "--", "unknown", "n/a"}


def _clean_name(raw: str) -> str:
    """A usable surname from a PGN name, or "".

    ``Carlsen, Magnus`` is the usual PGN spelling and the given name is dead
    weight in a 960px band, so only the part before the comma is kept. Everything
    the PGN uses to say "nobody" is rejected outright rather than printed.
    """
    name = (raw or "").strip()
    if name.lower() in _UNKNOWN_NAMES:
        return ""
    surname = name.split(",", 1)[0].strip()
    if surname.lower() in _UNKNOWN_NAMES:
        return ""
    return surname


def players_line(headers) -> str:
    """``Фамилия — Фамилия``, falling back to ``белые — чёрные``."""
    white = _clean_name(_header(headers, "White"))
    black = _clean_name(_header(headers, "Black"))
    if not white or not black:
        return FALLBACK_PLAYERS
    return f"{white} — {black}"


def _headers(headers, decision):
    if headers is not None:
        return headers
    return getattr(decision, "header", None)


def progress_line(ply: int, total_plies: int) -> str:
    """``24 / 134``. Empty when the game length is unknown, since a ratio with a
    made-up denominator is worse than no progress at all."""
    total = max(0, _as_int(total_plies))
    if total <= 0:
        return ""
    done = min(max(0, _as_int(ply)), total)
    return f"{done} / {total}"


# --- evaluation ------------------------------------------------------------


def format_eval(cp: int | None, mate: int | None = None) -> str:
    """One evaluation string, from the mover's point of view.

    ``cp`` is centipawns (hundredths of a pawn), ``mate`` a distance in moves with
    the same sign convention: positive is the mover mating. Centipawns print with
    an explicit sign, so a falling evaluation reads as a falling one; mate prints
    as ``#3`` / ``-#3``.

    Anything that rounds to nothing prints as ``0.0``, and a negative score that
    rounds to nothing must not print as ``-0.0``.
    """
    if mate is not None:
        return "#0" if mate == 0 else (f"#{abs(int(mate))}" if mate > 0 else f"-#{abs(int(mate))}")
    if cp is None:
        return ""
    value = int(cp) / 100.0
    if round(value, 1) == 0:
        return "0.0"
    return f"{value:+.1f}"


def _evals(decision, mover: str) -> tuple[str, str]:
    """The before/after pair, flipped from white's view into the mover's."""
    if decision is None:
        return "", ""
    sign = 1 if mover == "white" else -1
    before = getattr(decision, "eval_before_cp", None)
    after = getattr(decision, "eval_after_cp", None)
    mate_before = getattr(decision, "mate_before", None)
    mate_after = getattr(decision, "mate_after", None)
    return (
        format_eval(None if before is None else sign * _as_int(before),
                    None if mate_before is None else sign * _as_int(mate_before)),
        format_eval(None if after is None else sign * _as_int(after),
                    None if mate_after is None else sign * _as_int(mate_after)),
    )


# --- position facts --------------------------------------------------------


def _piece_at(fen: str, square: str) -> str:
    """The piece symbol on *square* in a FEN, or "" (empty squares included).

    Read straight out of the placement field so the castling check works even when
    python-chess is missing -- that check is pure bookkeeping and should not need
    the optional dependency.
    """
    parts = fen.split()
    if not parts:
        return ""
    rank = int(square[1])
    row = parts[0].split("/")[8 - rank] if 1 <= rank <= 8 else ""
    file_index = ord(square[0]) - ord("a")
    index = 0
    for char in row:
        if char.isdigit():
            index += int(char)
        elif index == file_index:
            return char
        else:
            index += 1
    return ""


def _castling_note(before_fen: str, after_fen: str, mover: str) -> str:
    """Say the rook can no longer castle, but only when that is provable.

    Rights only ever disappear because the king or the matching rook moved. So if
    the king is still home, the rook is still home, and the rights are gone, then
    it was the rook -- and "it was the rook" is exactly what the viewer cannot see,
    because the rook is back where it started.
    """
    if not before_fen or not after_fen:
        return ""
    before_parts = before_fen.split()
    after_parts = after_fen.split()
    if len(before_parts) < 3 or len(after_parts) < 3:
        return ""
    # Field 3 is the castling field: placement, side to move, rights, en passant.
    right_before, right_after = before_parts[2], after_parts[2]
    if mover == "white":
        if "K" not in right_before or "K" in right_after:
            return ""
        king_home, rook_home = "e1", "h1"
    else:
        if "k" not in right_before or "k" in right_after:
            return ""
        king_home, rook_home = "e8", "h8"
    if _piece_at(before_fen, king_home).lower() != "k":
        return ""
    if _piece_at(before_fen, rook_home).lower() != "r":
        return ""
    return "Ладья уже ходила — рокировки не будет"


def _material_edge(board) -> int:
    """Pure material, white minus black, in centipawns. No piece-square tables.

    A claim like "перевес у белых" has to survive a strong player looking at the
    position, so it counts pieces rather than opinions about where they stand.
    """
    edge = 0
    for piece in board.piece_map().values():
        value = _PIECE_CP.get(piece.symbol().lower(), 0)
        edge += value if piece.color else -value
    return edge


def _best_take(board) -> _Take | None:
    """The most material the side to move wins by simply capturing, net of recapture.

    "Hanging" is a net claim: a piece defended by more than the attacker takes is
    not really hanging, and a knight exchange is not a loss of material. Counting
    the recapture is what keeps the line honest.
    """
    if board is None:
        return None
    best: _Take | None = None
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        victim = board.piece_at(mv.to_square)
        if victim is None:
            # En passant takes the pawn beside the destination square, not on it.
            if not board.is_en_passant(mv):
                continue
            behind = mv.to_square - 8 if board.turn else mv.to_square + 8
            victim = board.piece_at(behind)
            if victim is None:
                continue
        won = _PIECE_CP.get(victim.symbol().lower(), 0)
        after = board.copy(stack=False)
        after.push(mv)
        recaptured = 0
        for reply in after.legal_moves:
            if reply.to_square != mv.to_square or not after.is_capture(reply):
                continue
            taker = after.piece_at(reply.to_square)
            if taker is None:
                continue
            recaptured = max(recaptured, _PIECE_CP.get(taker.symbol().lower(), 0))
        gain = won - recaptured
        if gain >= HANGING_CP and (best is None or gain > best.gain):
            best = _Take(
                gain=gain,
                square=_square_name(mv.to_square),
                piece=victim.symbol().lower(),
            )
    return best


def _square_name(square: int) -> str:
    """``"e4"`` for a python-chess square index, without importing the module."""
    return f"{chr(ord('a') + square % 8)}{square // 8 + 1}"


def _hanging_side(board) -> tuple[bool, _Take] | None:
    """Whichever side has a piece that can be taken for material, and the piece.

    Both sides are checked, because "hanging" means attacked and not defended
    enough, which is a statement about the position rather than about whose turn it
    is. The direct probe asks whether the side to move can win material now; the
    flipped one asks whether the opponent could take if it were their move, which
    is the same fact seen from the other side. What is returned is always the side
    that *loses* the piece, never the side that takes it.

    The second probe is skipped while the side to move is in check: flipping the
    turn then describes a position the opponent is not allowed to reach.
    """
    if board is None:
        return None
    taken = _best_take(board)
    if taken is not None:
        return (not board.turn, taken)
    if not board.is_check():
        flipped = board.copy(stack=False)
        flipped.turn = not flipped.turn
        taken = _best_take(flipped)
        if taken is not None:
            return (board.turn, taken)
    return None


def _recover_move(chess, before, after):
    """The single legal move that turns *before* into *after*, as (move, san).

    A frame that covers four plies has no single move, and the loop simply finds
    none: the line stays empty rather than naming the last of four moves as if it
    were the frame's move. Comparing full FENs rather than guessing from piece
    placement means captures, castling, en passant and promotion all come out
    right for free.
    """
    if chess is None or before is None or after is None:
        return None, ""
    if before.fen() == after.fen():
        return None, ""
    for mv in before.legal_moves:
        trial = before.copy(stack=False)
        trial.push(mv)
        if trial.fen() == after.fen():
            return mv, before.san(mv)
    return None, ""


def _move_from_uci(chess, uci):
    """``chess.Move`` for a UCI string, or ``None``."""
    if chess is None or not isinstance(uci, str) or len(uci) < 4:
        return None
    try:
        return chess.Move.from_uci(uci)
    except Exception:
        log.debug("narration: bad uci %r", uci, exc_info=True)
        return None


# --- hooks -----------------------------------------------------------------


def _hook_index(ply: int, size: int) -> int:
    """Which line of a pool this ply gets.

    Deliberately not ``ply % size``: the skim advances by a fixed step, and when
    that step is a multiple of the pool size a plain modulo picks the same line on
    every frame -- which is how a rotation turns back into a constant. Mixing in
    the floor of a slower division breaks the lock-step while staying a pure
    function of the ply, so a rebuild repeats itself exactly.
    """
    if size <= 0:
        return 0
    ply = max(0, int(ply))
    return (ply + ply // 5) % size


def _played_moves(ply: int) -> int:
    """Full moves finished *ply* plies in -- Russian counts a move by both sides."""
    return max(0, _as_int(ply)) // 2


def _balance_hook(board) -> str:
    """A skim line that states the balance, and only when the board agrees.

    Without python-chess there is no way to know, so this returns nothing and the
    pool simply has one line fewer rather than claiming a game is level when
    nobody has counted a single piece.
    """
    if board is None:
        return ""
    edge = _material_edge(board)
    if abs(edge) < 120:
        return "Пока всё примерно равно"
    if abs(edge) < EDGE_CP:
        return f"{_side_cap('white' if edge > 0 else 'black')} чуть впереди"
    return f"Перевес у {_side_ru('white' if edge > 0 else 'black')}"


def _skim_hook(board, ply: int) -> str:
    """Neutral, never a reveal: the fast pass exists to be fast."""
    moves = _played_moves(ply)
    pool = [
        f"{moves} {_plural(moves, _MOVE_FORMS)} сыграно",
        "Пока без происшествий",
        _balance_hook(board),
        "Смотрим партию целиком",
    ]
    lines = [line for line in pool if line]
    return lines[_hook_index(ply, len(lines))]


def _lead_hook(offset: int) -> str:
    """A real question, one or two plies before the position breaks.

    Keyed on the offset inside the slow window rather than on the ply, because the
    two frames of the window sit on consecutive plies and any rotation that steps
    by one would land both of them on the same line.
    """
    pool = ["Что здесь вообще играть?", "Кто ошибётся первым?"]
    return pool[abs(int(offset)) % len(pool)]


def _after_hook(offset: int) -> str:
    pool = ["Вот цена этого хода", "Теперь всё по-другому"]
    return pool[abs(int(offset)) % len(pool)]


def _engine_backed(decision) -> bool:
    """Whether a search, rather than a piece count, produced these numbers.

    ``analysis.Decision.material`` is the flag for it, and it defaults to True here:
    when it is missing, the narration assumes the weaker evidence and drops its
    quality words. That asymmetry is the whole point -- "зевок" is a judgement about
    the move, and without a search a position where a piece is simply taken (a
    sacrifice!) cannot be told apart from a mistake. Anderssen's 17.Nf6+ is the
    case that decides it: the piece really does hang, and the move is brilliant.
    """
    return not bool(getattr(decision, "material", True))


def _reveal_wording(decision, mover: str, after, fen_before: str) -> tuple[str, str]:
    """The hook and the detail for the decisive frame, as one decision.

    They are computed together on purpose: the two lines have to agree, and the
    fastest way to guarantee that is to establish the facts once and let both lines
    follow from them. The order is also the order of how damning the move is -- a
    mate that was on the board, a mate that this move creates, a piece left to be
    taken, castling that is gone, and only then a bare number.

    The wording never claims more than the analysis behind it. With an engine, a
    verified hanging piece is a blunder ("зевок") and a bare swing is a miss
    ("промах"). Without one, the same position gets the factual phrasing instead,
    because the material estimate cannot tell a blunder from a sacrifice and
    claiming the difference is how a channel ends up calling a masterpiece wrong.
    """
    mate_in = _mover_mate_in(decision, mover)
    if mate_in:
        # The engine said so. This is the question the whole clip has been
        # walking towards, so it gets to be the headline.
        moves = f"{mate_in} {_plural(mate_in, _MOVE_FORMS)}"
        short = f"Мат через {moves} — это было видно?"
        hook = short if len(short) <= MAX_HOOK_CHARS else f"Мат через {moves}. Видно?"
        return hook, f"После этого хода мат через {moves}"

    # A piece given away for a forced mate is a sacrifice, not a blunder, and the
    # material path cannot tell the two apart. Before calling anything a mistake,
    # check whether the move wins on the spot -- Anderssen's Qf6+ is the case that
    # matters, and without this line the channel calls a masterpiece a blunder.
    forced = _forced_mate_in(after, after.turn if after is not None else None)
    if forced:
        given = _best_take(after) if after is not None else None
        if given is not None:
            hook = f"{_side_cap(mover)} отдали {_PIECE_GEN[given.piece]} — и мат"
            return hook, f"Жертва: мат через {forced}, даже если заберут"
        return "И это приводит к мату", f"Мат через {forced} — ответа нет"

    if after is not None:
        # The side to move here is the opponent, so a capture worth real material
        # is a piece the mover left hanging. That is the shape of most blunders and
        # it is checkable from the position alone.
        taken = _best_take(after)
        if taken is not None:
            caught = f"Заберут без вопросов: минус {taken.gain / 100:.1f}"
            if _engine_backed(decision):
                return f"Зевок: {_PIECE_NOM[taken.piece]} на {taken.square} висит", caught
            return f"{_PIECE_NOM[taken.piece]} на {taken.square} под ударом", caught

    note = _castling_note(fen_before, after.fen() if after is not None else "", mover)
    if note:
        return "Вот этот ход — и он проигрывает", note

    loss = _mover_loss(decision)
    if loss is None or loss < MIN_CLAIMED_CP:
        return "Вот этот ход — и он проигрывает", ""
    swing = f"{loss / 100:.1f}"
    if not _engine_backed(decision):
        # The number came from a piece count, not from a search, and must not be
        # dressed up as analysis.
        return f"Минус {swing} за один ход", f"По счёту фигур: минус {swing}"
    if loss >= BLUNDER_CP:
        return f"Зевок: минус {swing} за один ход", f"Этот ход стоил {swing}"
    return f"Промах: минус {swing} за один ход", f"Этот ход стоил {swing}"


def _mover_mate_in(decision, mover: str) -> int | None:
    """Moves until mate after this move, when the mover walked into one.

    ``Decision.mate_after`` is white's view in moves, so a white mover walking into
    mate is negative and a black mover walking into mate is positive -- the sign
    only means anything once it is read against the mover, which is the same
    reading :attr:`analysis.Decision.is_mated` does. A score that says the mover is
    the one delivering mate returns nothing here: that is a good move, and putting
    "мат" on the band for it would be the opposite of the truth.
    """
    mate_after = getattr(decision, "mate_after", None)
    if mate_after is None:
        return None
    signed = _as_int(mate_after) * (1 if mover == "white" else -1)
    if signed >= 0:
        return None
    return abs(signed)


def _mover_loss(decision) -> int | None:
    """Centipawns the mover threw away, or ``None`` when it was not measured."""
    loss = getattr(decision, "mover_loss_cp", None)
    return None if loss is None else _as_int(loss)


# --- detail ----------------------------------------------------------------


def _mated_within(board, defender: bool, moves_left: int) -> bool:
    """Whether *defender* is mated within *moves_left* of its own moves, forced.

    Forced, not merely available: every reply the defender has must still lose, so
    this can say "мат через два хода" without lying about a defence that was
    simply missed.
    """
    if board.is_checkmate() and board.turn == defender:
        return True
    if moves_left <= 0:
        return False
    if board.turn == defender:
        replies = list(board.legal_moves)
        if not replies:
            return False
        for reply in replies:
            if not _push(board, reply, lambda after: _mated_within(after, defender, moves_left - 1)):
                return False
        return True
    for mv in list(board.legal_moves):
        if _push(board, mv, lambda after: after.is_checkmate()):
            return True
    if moves_left == 1:
        return False
    for mv in list(board.legal_moves):
        if not _push(board, mv, lambda after: _mated_within(after, defender, moves_left - 1)):
            return False
    return True


def _push(board, move, probe):
    """Push *move*, hand the board to *probe*, and always pop it again.

    The pop is in a ``finally`` because the board is borrowed from the frame's own
    position: a probe that raised would otherwise leave a half-played board behind
    for whatever read it next.
    """
    board.push(move)
    try:
        return bool(probe(board))
    finally:
        board.pop()


def _forced_mate_in(board, defender, moves_left: int = MATE_PROOF_MOVES) -> int | None:
    """Mate distance for the side *not* to move, or ``None``.

    This is the one place narration looks ahead instead of at the position, and it
    is what keeps a sacrifice from being written up as a blunder. Depth is capped at
    two moves because the proof multiplies every ply of the tree and this runs once
    per frame inside a batch: a claim it cannot prove is not a claim it makes.
    """
    if board is None or defender is None:
        return None
    for distance in range(1, max(1, int(moves_left)) + 1):
        if _mated_within(board, defender, distance):
            return distance
    return None


def _scene_detail(board) -> str:
    """What is true of the position on screen, or "" when nothing is.

    Every line here is read off the board. There is no engine in this path on
    purpose: a skim frame has no search behind it, and the alternative is a
    confident sentence about a position nobody analysed.
    """
    if board is None:
        return ""
    if board.is_checkmate():
        return f"{_side_cap(board.turn)} получили мат"
    if board.is_insufficient_material():
        return "Материала на победу не хватает"
    # Mate in one only, on the frames that are just context: the exact test is one
    # scan of the legal moves, while proving two moves multiplies the tree and
    # there is no reason to pay that for a frame nobody will study.
    mate = _forced_mate_in(board, not board.turn, moves_left=1)
    if mate:
        return f"У {_side_prep(board.turn)} мат одним ходом"
    hanging = _hanging_side(board)
    if hanging is not None:
        side, taken = hanging
        return (
            f"У {_side_prep(side)} {_PIECE_GEN[taken.piece]} "
            f"на {taken.square} висит — минус {taken.gain / 100:.1f}"
        )
    edge = _material_edge(board)
    if abs(edge) >= EDGE_CP:
        pawns = round(abs(edge) / 100)
        return f"{_side_cap('white' if edge > 0 else 'black')} впереди на {pawns} {_plural(pawns, _PAWN_FORMS)}"
    return ""


# --- the entry point -------------------------------------------------------


def _frame_kind(decision, ply: int) -> str:
    """Which part of the edit this frame belongs to.

    A paced frame shows the position *after* the last ply it covers, so the frame
    that reveals the decisive move is the one whose ply equals the decision's --
    not the one before it.
    """
    if decision is None:
        return KIND_SKIM
    target = getattr(decision, "ply", None)
    if target is None:
        return KIND_SKIM
    target = _as_int(target)
    if ply == target:
        return KIND_REVEAL
    if ply < target and target - ply <= LEAD_WINDOW_PLIES:
        return KIND_LEAD
    if ply > target and ply - target <= LEAD_WINDOW_PLIES:
        return KIND_AFTER
    return KIND_SKIM


def _window_offset(decision, ply: int) -> int:
    """How far this frame sits from the reveal, in plies. 0 outside the window."""
    target = getattr(decision, "ply", None)
    if target is None:
        return 0
    return int(ply) - _as_int(target)


def narrate(
    *,
    decision,
    ply: int,
    total_plies: int,
    fen_before: str,
    fen_after: str,
    color: str,
    headers=None,
) -> Narration:
    """The text and the arrow for one frame. Pure, and never raises.

    *decision* is the :class:`~shorts_clipper.chess.analysis.Decision` this clip is
    about, or ``None`` for a clip with no decisive move to point at. *ply* is the
    last ply the frame covers and *total_plies* the length of the game, both
    counted in plies, which is what the existing progress caption counts in.
    *color* is the side that made the move in this frame; *headers* is a PGN header
    mapping (or anything with the header names as attributes) and falls back to
    the decision's own header.

    ``eval_before`` / ``eval_after`` are filled only on the reveal frame. A skim
    frame has no search behind it, and a piece count wearing an evaluation's sign
    would be a number the clip never checked -- an honest gap in the band.
    """
    ply_no = max(0, _as_int(ply))
    total = max(0, _as_int(total_plies))
    kind = _frame_kind(decision, ply_no)
    offset = _window_offset(decision, ply_no)
    mover = _mover_color(color, decision)
    chess = _chess()
    before = _board(chess, fen_before)
    after = _board(chess, fen_after)

    move, san = _recover_move(chess, before, after)
    if kind == KIND_REVEAL:
        # The decision carries the move as it was actually played, with its check
        # and mate suffixes; the recovered one is only a fallback.
        san = str(getattr(decision, "san", "") or "") or san
        move = _move_from_uci(chess, getattr(decision, "uci", "")) or move

    try:
        if kind == KIND_REVEAL:
            hook, detail = _reveal_wording(decision, mover, after, fen_before)
        elif kind == KIND_LEAD:
            hook = _lead_hook(offset)
            detail = _scene_detail(after)
        elif kind == KIND_AFTER:
            hook = _after_hook(offset)
            detail = _scene_detail(after)
        else:
            hook = _skim_hook(after, ply_no)
            detail = _scene_detail(after)
    except Exception:
        log.debug("narration: wording failed at ply %s", ply_no, exc_info=True)
        hook, detail = "", ""
    hook = _sentence_case(hook)

    if len(hook) > MAX_HOOK_CHARS:
        log.warning("narration: hook is %d chars at ply %d: %r", len(hook), ply_no, hook)
    if len(detail) > MAX_DETAIL_CHARS:
        log.warning("narration: detail is %d chars at ply %d: %r", len(detail), ply_no, detail)

    eval_before, eval_after = _evals(decision, mover) if kind == KIND_REVEAL else ("", "")

    arrow: tuple[str, str] | None = None
    if kind == KIND_REVEAL and move is not None:
        arrow = (chess.square_name(move.from_square), chess.square_name(move.to_square))

    return Narration(
        info=FrameInfo(
            hook=hook,
            detail=detail,
            progress=progress_line(ply_no, total),
            eval_before=eval_before,
            eval_after=eval_after,
            players=players_line(_headers(headers, decision)),
            moved_san=san,
        ),
        arrow=arrow,
        accent="bad" if kind == KIND_REVEAL else "neutral",
        kind=kind,
    )
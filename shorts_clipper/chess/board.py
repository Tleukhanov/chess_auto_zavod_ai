"""Programmatic chess board rendering for vertical video.

Chess-specific formats do not need footage: the board *is* the visual. So it is
drawn here rather than sourced, which keeps the pipeline licence-free,
deterministic, and free of anyone else's screen recording.

Rendering is PIL rather than SVG because nothing in the dependency set can
rasterise SVG (no cairosvg/svglib) and PIL gives direct control over the 1080x1920
vertical frame: the board sits high, a header sits above it and an information
band below, so a 9:16 frame carries the position and the words about it at the
same time instead of only the board.

Only the unicode chess glyphs are used, so no image assets ship with the repo.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from pathlib import Path

log = logging.getLogger(__name__)

FRAME_W = 1080
FRAME_H = 1920
BOARD = 1000
# Vertical layout. The board used to be centred, which put 460px of empty black
# above and below it -- measured on a rendered highlight clip, the board
# occupied 52% of a 9:16 frame and a third of the picture did nothing. The board
# now sits high and the freed space below becomes the information band.
HEADER_TOP = 0
HEADER_HEIGHT = 300
BOARD_TOP = HEADER_HEIGHT
INFO_TOP = BOARD_TOP + BOARD + 40
INFO_HEIGHT = FRAME_H - INFO_TOP

# Information band. The sizes are absolute pixels for a 1080x1920 frame: the
# caption sizes above (64/44) were chosen when text was a caption and are
# unreadable once the text is the headline of the frame. Hook is tried largest
# first and steps down until the wrapped block fits the band, so a long hook
# shrinks instead of running off the frame.
INFO_PAD = 60
HOOK_SIZES = (76, 72, 68, 64, 60, 56, 52, 48, 44)
HOOK_MIN_SIZE = HOOK_SIZES[-1]
DETAIL_SIZES = (44, 40, 36, 32)
MOVE_SIZES = (52, 48, 44, 40, 36, 32)
PROGRESS_SIZE = 34
PLAYERS_SIZE = 34
BOTTOM_TEXT_SIZE = 44
# Floor for the gap between two band rows. Whatever the rows leave over is shared
# out above this floor, so the band fills its region instead of leaving behind the
# black hole the centred board used to have under itself.
BAND_GAP = 26
_ARROW = "→"
_ARROW_FALLBACK = "->"
# A notation ending in '+' prints against a signed number as "Qd6++0.4" without
# a separator, so the move and the swing it caused are joined by a dot.
_MOVE_SEPARATOR = "   ·   "
_MOVE_STROKE = 2

# Muted palette: the board must not fight the caption text for attention.
LIGHT_SQUARE = (222, 214, 200)
DARK_SQUARE = (124, 116, 108)
BOARD_BG = (18, 18, 20)
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (150, 150, 155)
ACCENT_BAD = (214, 74, 62)
ACCENT_GOOD = (96, 178, 116)
HIGHLIGHT = (226, 190, 92)

PIECE_GLYPHS = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

# No single system font is guaranteed to carry both Cyrillic and the chess block:
# on Windows, Segoe UI has Cyrillic and no chess pieces, while Segoe UI Symbol
# has the pieces and no Cyrillic. So the two are resolved separately and a font
# is only accepted for a role once its cmap has actually been verified.
_TEXT_FONTS = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
)
_PIECE_FONTS = (
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_PROBE_CHAR = "\ue123"  # private-use, guaranteed absent from these fonts


def _load(path: str | Path, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(path), size)


def _covers(font, char: str) -> bool:
    """Whether *font* really has *char*, by comparing against an absent codepoint.

    Comparing rendered bitmaps is the only portable check here -- PIL exposes no
    cmap API. A missing glyph renders as .notdef, identical to any other missing
    codepoint, so byte-comparing against a private-use probe is reliable.
    """
    try:
        mask = font.getmask(char)
        ref = font.getmask(_PROBE_CHAR)
        if mask.size != ref.size:
            return True
        return bytes(mask) != bytes(ref)
    except Exception:
        return False


@functools.lru_cache(maxsize=256)
def _first_font(paths: tuple[str, ...], size: int, probe: str):
    # Cached because the band tries a hook at up to nine sizes per frame and
    # loading a TTF is milliseconds each; PIL font objects are immutable once
    # loaded, so the same object can be handed to every draw call.
    for path in paths:
        if not Path(path).is_file():
            continue
        try:
            font = _load(path, size)
        except Exception:
            continue
        if _covers(font, probe):
            return font
    try:
        import matplotlib

        bundled = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            font = _load(bundled, size)
            if _covers(font, probe):
                return font
    except Exception:
        log.debug("matplotlib font fallback unavailable", exc_info=True)
    log.warning("No font covering %r found; falling back to PIL default", probe)
    from PIL import ImageFont

    return ImageFont.load_default()


@dataclasses.dataclass(frozen=True)
class FrameInfo:
    """What the viewer is told while the board is on screen.

    Rendered into the band below the board. Every field is optional so a format
    that only wants one line does not have to blank the rest, and so a caller
    that passes nothing keeps the old behaviour exactly.
    """

    hook: str = ""
    """The headline. What is happening, in a viewer question shape."""

    detail: str = ""
    """Supporting line under the hook: why the move matters."""

    progress: str = ""
    """Where we are in the game, e.g. ``24 / 134``."""

    eval_before: str = ""
    """Evaluation before the move, sign from the mover's side."""

    eval_after: str = ""
    """Evaluation after it. Together with *eval_before* this is the proof."""

    players: str = ""
    """Who is playing, e.g. ``Карлусен — Непомнящий``."""

    moved_san: str = ""
    """The move in algebraic notation, e.g. ``Qd6+``."""


@dataclasses.dataclass(frozen=True)
class FrameSpec:
    """One rendered frame of a board state."""

    fen: str
    top_text: str
    bottom_text: str
    accent: str = "neutral"  # "bad" | "good" | "neutral"
    highlight_squares: tuple[str, ...] = ()
    arrow: tuple[str, str] | None = None
    # Engine evaluation, shown so the claim on screen is checkable rather than
    # asserted. A chess audience verifies, and a number is what they can check.
    eval_text: str | None = None
    # The information band below the board. None keeps the frame exactly as it
    # was before this field existed, so the deciding, opening, endgame and
    # challenge formats are unaffected.
    info: FrameInfo | None = None


def _accent_color(accent: str):
    return {"bad": ACCENT_BAD, "good": ACCENT_GOOD}.get(accent, TEXT_MAIN)


def _square_centres(size: int):
    """(file, rank) -> pixel centre, with rank 8 at the top."""
    out = {}
    step = size / 8.0
    for row in range(8):
        rank = 8 - row
        for col in range(8):
            file = chr(ord("a") + col)
            out[(file, rank)] = (col * step + step / 2, row * step + step / 2)
    return out


def _text_extent(draw, text: str, font) -> tuple[int, int]:
    """Ink width and height of *text* in *font*, measured not assumed.

    Width is the right edge rather than right-minus-left so a glyph with a
    negative left bearing is not mistaken for something that fits. Every size
    decision in the band is taken from this, because a nominal pixel size says
    nothing about how wide Cyrillic or a wrapped hook actually comes out.
    """
    if not text:
        return (0, 0)
    _, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return (x1, y1 - y0)


def _text_anchor_box(draw, text: str, font) -> tuple[int, int]:
    """(lead, extent) of *text* drawn from an ascender-anchored origin.

    PIL reports an ascender anchor's ink as (y0, y1) measured down from the anchor
    line, so a layout can advance by the real distance from one anchor line to the
    bottom of the next row's ink. Advancing by ink *height* instead would drop the
    leading in front of every row, and the band then spaces unevenly and walks off
    the bottom of the frame: that was measured, not assumed.
    """
    if not text:
        return (0, 0)
    _, y0, _, y1 = draw.textbbox((0, 0), text, font=font, anchor="la")
    return (y0, y1)



def _line_step(draw, font) -> int:
    """Baseline-to-baseline distance for *font*: one line of a block."""
    try:
        ascent, descent = font.getmetrics()
        return max(1, ascent + descent)
    except Exception:
        return max(1, _text_extent(draw, "ХAy", font)[1])


def _block_extent(draw, font, lines: list[str], step: int) -> int:
    """Distance from a block's anchor line to the bottom of its last line."""
    if not lines:
        return 0
    return (len(lines) - 1) * step + _text_anchor_box(draw, lines[-1], font)[1]


def _fit_font(draw, text: str, width: int, sizes: tuple[int, ...]):
    """The largest font in *sizes* whose *text* fits *width*, else the smallest.

    Shrinking rather than wrapping, because these fields are meant to be one
    line: a split evaluation is harder to read on a phone than a smaller one.
    """
    font = _first_font(_TEXT_FONTS, sizes[-1], "Х")
    for size in sizes:
        font = _first_font(_TEXT_FONTS, size, "Х")
        if _text_extent(draw, text, font)[0] <= width:
            break
    return font


def _wrap(draw, text: str, font, width: int) -> list[str]:
    """Greedy word wrap of *text* to *width* px, never returning an over-wide line.

    A single token wider than the band (a long URL, a run with no spaces in it)
    is broken by character instead of running past the padding, so no caller ever
    has to clip the result.
    """
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        current = ""
        for word in para.split(" "):
            trial = f"{current} {word}".strip()
            if not current or _text_extent(draw, trial, font)[0] <= width:
                current = trial
                continue
            lines.append(current)
            current = word
        lines.append(current)

    out: list[str] = []
    for line in lines:
        if not line or _text_extent(draw, line, font)[0] <= width:
            out.append(line)
            continue
        chunk = ""
        for char in line:
            if chunk and _text_extent(draw, chunk + char, font)[0] > width:
                out.append(chunk)
                chunk = char
            else:
                chunk += char
        out.append(chunk)
    return [line for line in out if line] or [""]


def _fit_hook(draw, text: str, width: int, room: int):
    """The hook at the largest size in :data:`HOOK_SIZES` whose ink fits *room*.

    Returns ``(font, lines, lead, ink)``. A hook too long even at the smallest
    size keeps the leading lines that do fit: the opening of a hook carries the
    question and the tail is what a phone viewer can afford to lose. This is the
    only place the band may lose content, and losing it beats a frame that
    overflows or a render that raises on a long string.
    """
    for size in HOOK_SIZES:
        font = _first_font(_TEXT_FONTS, size, "Х")
        lines = _wrap(draw, text, font, width)
        step = _line_step(draw, font)
        lead = _text_anchor_box(draw, lines[0], font)[0]
        extent = _block_extent(draw, font, lines, step)
        if extent - lead <= room:
            return font, lines, lead, extent - lead

    font = _first_font(_TEXT_FONTS, HOOK_MIN_SIZE, "Х")
    lines = _wrap(draw, text, font, width)
    step = _line_step(draw, font)
    lead = _text_anchor_box(draw, lines[0], font)[0]
    keep = max(1, (room + lead) // step)
    if len(lines) > keep:
        log.debug("Hook clipped to %d of %d lines to fit the band", keep, len(lines))
        lines = lines[:keep]
    extent = _block_extent(draw, font, lines, step)
    return font, lines, lead, max(1, extent - lead)


def _draw_run(draw, x: int, y: int, text: str, font, fill) -> int:
    """Draw one coloured run at *x* and return where the next one starts."""
    if not text:
        return x
    draw.text((x, y), text, font=font, fill=fill, anchor="la")
    return x + _text_extent(draw, text, font)[0]


def _eval_runs(info: FrameInfo, arrow: str) -> list[tuple[str, str]]:
    """The evaluation line as (text, colour) pairs.

    Both sides present reads as a swing -- before, arrow, after in the accent --
    because the drop is the story and a bare pair of numbers does not show it.
    """
    before, after = info.eval_before.strip(), info.eval_after.strip()
    if before and after:
        return [(before, "dim"), (f"  {arrow}  ", "dim"), (after, "accent")]
    if after:
        return [(after, "accent")]
    if before:
        return [(before, "dim")]
    return []


def _move_runs(info: FrameInfo, san: str, arrow: str) -> list[tuple[str, str]]:
    """The move line as (text, colour) pairs: the notation, then the swing.

    One line, because a move and the evaluation it caused belong together: read
    across, ``Qd6+  ·  +0.4 → -2.1``. The dot matters -- without a separator a
    notation ending in '+' prints against a signed number as ``Qd6++0.4``.
    """
    runs = _eval_runs(info, arrow)
    parts: list[tuple[str, str]] = []
    if san:
        parts.append((san, "move"))
    if san and runs:
        parts.append((_MOVE_SEPARATOR, "dim"))
    parts.extend(runs)
    return parts


def _band_row(draw, kind: str, font, payload):
    """One measured band row: ``(kind, font, lead, ink height, payload)``.

    *lead* is how far below an ascender anchor the ink starts and *ink* is how
    tall it is. Measuring both is what lets the band space rows by the black a
    viewer actually sees between them instead of by nominal font size.
    """
    if kind == "hook":
        step = _line_step(draw, font)
        lead = _text_anchor_box(draw, payload[0], font)[0]
        ink = _block_extent(draw, font, payload, step) - lead
    elif kind == "move":
        parts = [text for text, _ in payload if text]
        lead = min(_text_anchor_box(draw, part, font)[0] for part in parts)
        ink = max(_text_anchor_box(draw, part, font)[1] for part in parts) - lead
    else:
        lead, extent = _text_anchor_box(draw, payload, font)
        ink = extent - lead

    return (kind, font, lead, max(1, ink), payload)


def _draw_band_row(draw, row, y: int, accent) -> None:
    """Draw one band row whose ink starts at *y*."""
    kind, font, _, _, payload = row
    if kind == "progress":
        draw.text((FRAME_W - INFO_PAD, y), payload, font=font, fill=TEXT_DIM, anchor="ra")
    elif kind == "hook":
        step = _line_step(draw, font)
        for i, line in enumerate(payload):
            draw.text((INFO_PAD, y + i * step), line, font=font, fill=accent, anchor="la")
    elif kind == "move":
        x = INFO_PAD
        for text, colour in payload:
            if colour == "move":
                # A stroke in the fill colour is what makes the notation read as
                # bold: only one text font family is loaded, and a heavy headline
                # beat deserves the weight more than a second font file does.
                draw.text((x, y), text, font=font, fill=TEXT_MAIN, anchor="la",
                          stroke_width=2, stroke_fill=TEXT_MAIN)
                x += _text_extent(draw, text, font)[0] + _MOVE_STROKE
            else:
                x = _draw_run(draw, x, y, text, font, accent if colour == "accent" else TEXT_DIM)
    else:
        draw.text((INFO_PAD, y), payload, font=font, fill=TEXT_DIM, anchor="la")



def _render_info_band(draw, spec: FrameSpec, bottom_limit: int) -> None:
    """Draw the information band between :data:`INFO_TOP` and *bottom_limit*.

    Every row is measured before anything is placed, and what the rows do not use
    is shared out as equal gaps -- including the tail below the last row. Both
    steps are the point of moving the board up: the band has to occupy its
    region, or the frame still carries the dead black that the centred board
    measured at 460px above itself and 459px below.
    """
    info = spec.info
    if info is None:
        return

    width = FRAME_W - 2 * INFO_PAD
    accent = _accent_color(spec.accent)
    region = bottom_limit - INFO_TOP

    hook = info.hook.strip()
    detail = info.detail.strip()
    progress = info.progress.strip()
    players = info.players.strip()
    san = info.moved_san.strip()
    # Coverage is a property of the font file rather than of the size, so it is
    # settled once: an arrow that rendered as tofu would cost more than it said.
    probe_font = _first_font(_TEXT_FONTS, MOVE_SIZES[0], "Х")
    arrow = _ARROW if _covers(probe_font, _ARROW) else _ARROW_FALLBACK
    runs = _move_runs(info, san, arrow)

    # Reading order is progress, hook, detail, move and evaluation, players. The
    # hook is sized after the rest are measured because its size decides how much
    # height is left for them.
    above: list[tuple] = []
    below: list[tuple] = []
    if progress:
        font = _first_font(_TEXT_FONTS, PROGRESS_SIZE, "Х")
        above.append(_band_row(draw, "progress", font, progress))
    if detail:
        font = _fit_font(draw, detail, width, DETAIL_SIZES)
        below.append(_band_row(draw, "detail", font, detail))
    if runs:
        font = _fit_font(draw, "".join(text for text, _ in runs), width, MOVE_SIZES)
        below.append(_band_row(draw, "move", font, runs))
    if players:
        font = _first_font(_TEXT_FONTS, PLAYERS_SIZE, "Х")
        below.append(_band_row(draw, "players", font, players))

    rows = above
    if hook:
        # The hook may take the region minus the fixed ink and one minimum gap
        # per slot, the tail included. A gap in front of a row also has to cover
        # that row's leading, so the largest lead below the hook is charged here.
        leads = [row[2] for row in below] or [0]
        room = region - sum(row[3] for row in below) - (BAND_GAP + max(leads)) * (len(below) + 2)
        font, lines, lead, ink = _fit_hook(draw, hook, width, room)
        rows = above + [_band_row(draw, "hook", font, lines)]
    rows = rows + below

    if not rows:
        return

    # Every gap in the band gets the same share of the height left over: the ones
    # between rows and the tail under the last one. Black at the bottom of the
    # band is the same dead space as black between two lines of it.
    share = (region - sum(row[3] for row in rows)) // len(rows)

    y = INFO_TOP - rows[0][2]
    for i, row in enumerate(rows):
        y = min(y, max(INFO_TOP, bottom_limit - row[3] - row[2]))
        _draw_band_row(draw, row, y, accent)
        # The next row's leading is subtracted so the black between two rows is
        # the share itself and not the share plus one font's worth of leading.
        nxt = rows[i + 1][2] if i + 1 < len(rows) else 0
        y += row[2] + row[3] + max(1, share - nxt)


def render_frame(spec: FrameSpec, out_path: str | Path) -> Path:
    """Render *spec* to a PNG and return its path."""
    from PIL import Image, ImageDraw

    board_top = BOARD_TOP
    board_left = (FRAME_W - BOARD) // 2

    img = Image.new("RGB", (FRAME_W, FRAME_H), BOARD_BG)
    draw = ImageDraw.Draw(img)

    draw.text(
        (FRAME_W // 2, 120),
        spec.top_text,
        fill=_accent_color(spec.accent),
        font=_first_font(_TEXT_FONTS, 64, "Х"),
        anchor="mm",
    )
    if spec.bottom_text:
        draw.text(
            (FRAME_W // 2, FRAME_H - 180),
            spec.bottom_text,
            fill=TEXT_DIM,
            font=_first_font(_TEXT_FONTS, BOTTOM_TEXT_SIZE, "Х"),
            anchor="mm",
        )
    if spec.eval_text:
        # Sits under the hook, above the board: present but never competing with
        # the question.
        draw.text(
            (FRAME_W // 2, 232),
            spec.eval_text,
            fill=TEXT_DIM,
            font=_first_font(_TEXT_FONTS, 40, "Х"),
            anchor="mm",
        )

    step = BOARD / 8.0
    highlight = {s.lower() for s in spec.highlight_squares}

    for row in range(8):
        for col in range(8):
            file = chr(ord("a") + col)
            rank = 8 - row
            name = f"{file}{rank}"
            x0 = board_left + col * step
            y0 = board_top + row * step
            light = (row + col) % 2 == 1
            fill = LIGHT_SQUARE if light else DARK_SQUARE
            if name in highlight:
                fill = HIGHLIGHT
            draw.rectangle([x0, y0, x0 + step, y0 + step], fill=fill)

    try:
        import chess

        board = chess.Board(spec.fen)
        pieces = board.piece_map()
    except Exception:
        log.debug("FEN parse failed while drawing", exc_info=True)
        pieces = {}

    glyph_font = _first_font(_PIECE_FONTS, int(step * 0.82), "\u2654")
    moved = {s.lower() for s in spec.highlight_squares}
    for (file, rank), (cx, cy) in _square_centres(BOARD).items():
        piece = pieces.get(chess_square(file, rank)) if _chess_ok() else None
        if piece is None:
            continue
        glyph = PIECE_GLYPHS[piece.symbol()]
        px = board_left + cx
        py = board_top + cy
        # Standard colours: white pieces are outline glyphs and need a dark edge
        # to read on a light square, black pieces are filled and need none.
        # Colouring whole sides instead would make "which side is which" a guess.
        if piece.color:
            draw.text((px, py + step * 0.02), glyph, fill=(252, 250, 246),
                      font=glyph_font, anchor="mm", stroke_width=5, stroke_fill=(40, 38, 36))
        else:
            draw.text((px, py + step * 0.02), glyph, fill=(38, 36, 34),
                      font=glyph_font, anchor="mm")
        # The square the blundered piece landed on keeps the accent, so the eye
        # goes to the move rather than to a whole army.
        if f"{file}{rank}" in moved and piece.color:
            draw.text((px, py + step * 0.02), glyph, fill=ACCENT_BAD,
                      font=glyph_font, anchor="mm")

    if spec.arrow:
        centres = _square_centres(BOARD)
        try:
            a = centres[(spec.arrow[0][0], int(spec.arrow[0][1]))]
            b = centres[(spec.arrow[1][0], int(spec.arrow[1][1]))]
            p1 = (board_left + a[0], board_top + a[1])
            p2 = (board_left + b[0], board_top + b[1])
            draw.line([p1, p2], fill=ACCENT_BAD, width=14)
            draw.ellipse([p2[0] - 16, p2[1] - 16, p2[0] + 16, p2[1] + 16], fill=ACCENT_BAD)
        except Exception:
            log.debug("Arrow skipped", exc_info=True)

    draw.rectangle(
        [board_left, board_top, board_left + BOARD, board_top + BOARD],
        outline=(60, 60, 64), width=3,
    )

    if spec.info is not None:
        # bottom_text keeps its own anchor; the band stops above it so players
        # never print through it, and with no bottom_text the band owns the
        # bottom of the frame.
        bottom_limit = FRAME_H
        if spec.bottom_text:
            _, y0, _, _ = draw.textbbox(
                (FRAME_W // 2, FRAME_H - 180),
                spec.bottom_text,
                font=_first_font(_TEXT_FONTS, BOTTOM_TEXT_SIZE, "Х"),
                anchor="mm",
            )
            bottom_limit = min(bottom_limit, y0 - BAND_GAP)
        _render_info_band(draw, spec, bottom_limit)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _chess_ok() -> bool:
    try:
        import chess  # noqa: F401

        return True
    except ImportError:
        return False


def chess_square(file: str, rank: int):
    import chess

    return chess.square(chess.FILE_NAMES.index(file), rank - 1)

"""Programmatic chess board rendering for vertical video.

Chess-specific formats do not need footage: the board *is* the visual. So it is
drawn here rather than sourced, which keeps the pipeline licence-free,
deterministic, and free of anyone else's screen recording.

Rendering is PIL rather than SVG because nothing in the dependency set can
rasterise SVG (no cairosvg/svglib) and PIL gives direct control over the 1080x1920
vertical frame: the board sits in the middle with text bands above and below.

Only the unicode chess glyphs are used, so no image assets ship with the repo.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

log = logging.getLogger(__name__)

FRAME_W = 1080
FRAME_H = 1920
BOARD = 1000

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


def _first_font(paths: tuple[str, ...], size: int, probe: str):
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


def render_frame(spec: FrameSpec, out_path: str | Path) -> Path:
    """Render *spec* to a PNG and return its path."""
    from PIL import Image, ImageDraw

    board_top = (FRAME_H - BOARD) // 2
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
            font=_first_font(_TEXT_FONTS, 44, "Х"),
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

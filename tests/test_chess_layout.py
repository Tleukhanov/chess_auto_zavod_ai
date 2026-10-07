"""Tests for the vertical layout of one rendered chess frame.

The defect these pin is a measurement, not a crash. ``render_frame`` centred the
board with ``board_top = (FRAME_H - BOARD) // 2`` on a 1080x1920 frame, which on a
real rendered clip put the board at 52% of the frame height with 460px of empty
black above it and 459px below: 24% dead at the top, 24% at the bottom, a third
of the picture doing nothing. The board now starts at ``BOARD_TOP`` and the freed
height below it carries the information band.

So these tests assert geometry and pixels, not exceptions: the constants have to
stay self-consistent, the board has to sit where the contract says, and a frame
with the band populated must not open a hole bigger than 60px of pure black below
the board. They need PIL and a position, not a video, because the whole point is
that the layout is knowable from one frame.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image, ImageDraw

    HAVE_PIL = True
except ImportError:  # pragma: no cover - environment without the optional dep
    HAVE_PIL = False

from shorts_clipper.chess.board import (
    ACCENT_BAD,
    ACCENT_GOOD,
    BOARD,
    BOARD_BG,
    BOARD_TOP,
    FRAME_H,
    FRAME_W,
    HEADER_HEIGHT,
    HEADER_TOP,
    INFO_HEIGHT,
    INFO_TOP,
    FrameInfo,
    FrameSpec,
    _fit_hook,
    render_frame,
)

FEN = "r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2P5/PP1Q1PPP/RNB1K1NR w KQ - 0 1"
# A band that left a hole this big read as an unfinished layout on a phone.
MAX_BLACK_BAND = 60

FULL_INFO = FrameInfo(
    hook="Белые ходят. Мат в 2. Находишь?",
    detail="Qe8+ — мат в два хода, а не размен",
    progress="24 / 134",
    eval_before="+0.4",
    eval_after="-2.1",
    players="Карлусен — Непомнящий",
    moved_san="Qd6+",
)


def _spec(info=None, **kwargs) -> FrameSpec:
    fields = {
        "fen": FEN,
        "top_text": "",
        "bottom_text": "",
        "accent": "bad",
        "highlight_squares": ("d6", "h4"),
        "arrow": ("d6", "h4"),
        "info": info,
    }
    fields.update(kwargs)
    return FrameSpec(**fields)


def _render(tmp: str, spec: FrameSpec):
    return Image.open(render_frame(spec, Path(tmp) / "frame.png")).convert("RGB")


def _ink_mask(img):
    """Rows/columns carrying any pixel brighter than 60, as a 0/255 mask."""
    return img.convert("L").point(lambda v: 255 if v > 60 else 0)


def _largest_black_band(img, y0: int, y1: int) -> int:
    """Longest run of rows in ``[y0, y1)`` with no pixel brighter than 60.

    This is the measurement the layout is judged on: a run of fully black rows is
    the dead space a viewer sees as an unfinished frame.
    """
    mask = _ink_mask(img)
    biggest = run = 0
    for y in range(y0, y1):
        if mask.crop((0, y, mask.width, y + 1)).getbbox():
            biggest, run = max(biggest, run), 0
        else:
            run += 1
    return max(biggest, run)


def _board_span(img) -> tuple[int, int, int, int] | None:
    """``(top, bottom, left, right)`` of the drawn board, measured from pixels.

    Anything that is not the frame background counts, which picks up the outline
    as well as the squares. The scan is clipped to the board's own band so a
    caption or a line of the information band cannot be mistaken for it.
    """
    mid = BOARD_TOP + BOARD // 2
    columns = [x for x in range(FRAME_W) if img.getpixel((x, mid)) != BOARD_BG]
    rows = [
        y for y in range(BOARD_TOP - 20, BOARD_TOP + BOARD + 20)
        if img.getpixel((FRAME_W // 2, y)) != BOARD_BG
    ]
    if not columns or not rows:
        return None
    return rows[0], rows[-1], columns[0], columns[-1]


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class GeometryContractTests(unittest.TestCase):
    def test_info_band_reaches_the_bottom_of_the_frame(self):
        self.assertEqual(INFO_TOP + INFO_HEIGHT, FRAME_H)
        self.assertEqual(INFO_TOP, BOARD_TOP + BOARD + 40)
        self.assertGreater(INFO_TOP, BOARD_TOP + BOARD, "band overlaps the board")
        self.assertGreater(INFO_HEIGHT, 0)

    def test_board_starts_where_the_header_ends(self):
        self.assertEqual(BOARD_TOP, HEADER_HEIGHT)
        self.assertEqual(HEADER_TOP + HEADER_HEIGHT, BOARD_TOP)

    def test_board_is_centred_horizontally(self):
        self.assertEqual((FRAME_W - BOARD) % 2, 0)
        self.assertLess(BOARD, FRAME_W)

    def test_frame_is_a_nine_by_sixteen_vertical(self):
        self.assertEqual((FRAME_W, FRAME_H), (1080, 1920))


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class RenderTests(unittest.TestCase):
    def test_frame_is_a_1080x1920_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = render_frame(_spec(), Path(tmp) / "frame.png")
            self.assertEqual(out.suffix, ".png")
            with Image.open(out) as img:
                self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_without_info_the_frame_still_renders(self):
        # info=None is what the deciding, opening, endgame and challenge formats
        # pass, so this path has to keep working untouched.
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(None))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))
            self.assertIsNotNone(_board_span(img))

    def test_without_info_the_captions_still_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(
                None,
                top_text="Ход 34: Qd6+",
                bottom_text="Магнус Карлсен — Игнитий Непомнящий",
                eval_text="+1.2",
                accent="good",
            ))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))
            self.assertIsNotNone(_board_span(img))

    def test_every_info_field_populated_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FULL_INFO))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))
            band = img.crop((0, INFO_TOP, FRAME_W, FRAME_H)).convert("L")
            self.assertIsNotNone(
                band.point(lambda v: 255 if v > 60 else 0).getbbox(),
                "nothing was drawn in the information band",
            )

    def test_single_field_info_renders(self):
        for info in (
            FrameInfo(hook="Один только хук"),
            FrameInfo(players="Карлусен — Непомнящий"),
            FrameInfo(moved_san="Qd6+"),
            FrameInfo(eval_after="-9.9"),
            FrameInfo(eval_before="+0.2"),
            FrameInfo(progress="7 / 90"),
            FrameInfo(detail="Только пояснение"),
            FrameInfo(),
        ):
            with self.subTest(fields=info):
                with tempfile.TemporaryDirectory() as tmp:
                    img = _render(tmp, _spec(info))
                    self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_info_with_bottom_text_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FULL_INFO, bottom_text="Магнус Карлсен"))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_board_sits_high_and_is_a_thousand_pixels_wide(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FULL_INFO))
            span = _board_span(img)
            self.assertIsNotNone(span, "no board drawn")
            top, bottom, left, right = span
            self.assertEqual(top, BOARD_TOP)
            self.assertEqual(bottom, BOARD_TOP + BOARD)
            self.assertEqual(left, (FRAME_W - BOARD) // 2)
            # PIL strokes the outline on both edges, so the box is one pixel
            # wider than the squares inside it.
            self.assertEqual(right - left, BOARD)

    def test_populated_band_leaves_no_hole_bigger_than_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FULL_INFO))
            biggest = _largest_black_band(img, BOARD_TOP + BOARD, FRAME_H)
            self.assertLess(
                biggest, MAX_BLACK_BAND,
                f"{biggest}px of pure black below the board, was 459px before",
            )

    def test_hook_colour_follows_the_accent(self):
        with tempfile.TemporaryDirectory() as tmp:
            band = _render(tmp, _spec(FULL_INFO, accent="bad")).crop((0, INFO_TOP, FRAME_W, FRAME_H))
            self.assertIn(ACCENT_BAD, {colour for _, colour in band.getcolors(maxcolors=1 << 20)})
            band = _render(tmp, _spec(FULL_INFO, accent="good")).crop((0, INFO_TOP, FRAME_W, FRAME_H))
            self.assertIn(ACCENT_GOOD, {colour for _, colour in band.getcolors(maxcolors=1 << 20)})


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class LongTextTests(unittest.TestCase):
    def test_very_long_hook_wraps_or_shrinks_and_still_fits_the_frame(self):
        hook = (
            "This is the move that ended the game and nobody saw it coming "
            "because " * 12
        )
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FrameInfo(hook=hook, progress="3 / 4",
                                               players="a - b", moved_san="Qh5#")))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_hook_with_no_spaces_at_all_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FrameInfo(hook="W" * 900)))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_overlong_detail_and_players_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FrameInfo(hook="Крюк", detail="d" * 900,
                                               players="p" * 400, moved_san="m" * 200)))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))

    def test_band_never_overflows_past_the_frame(self):
        # A hook far too long for the band is truncated to the lines that fit, so
        # the frame still ends in empty space instead of in a half-cut line of
        # type. Ink on the last row would mean the band ran off the frame.
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(tmp, _spec(FrameInfo(hook="Хук " * 200, detail="деталь " * 100,
                                               progress="1 / 2", eval_before="+1.0",
                                               eval_after="-4.0", players="а - б",
                                               moved_san="Qd6+")))
            self.assertEqual(img.size, (FRAME_W, FRAME_H))
            last = img.crop((0, FRAME_H - 1, FRAME_W, FRAME_H)).convert("L")
            self.assertLessEqual(last.getextrema()[1], 60, "band ink reached the last row")

    def test_hook_shrinks_when_it_cannot_fit(self):
        draw = ImageDraw.Draw(Image.new("RGB", (FRAME_W, 400)))
        width = FRAME_W - 120
        big_font, _, _, _ = _fit_hook(draw, "Короткий хук", width, 200)
        small_font, _, _, _ = _fit_hook(draw, "Длинный хук " * 60, width, 200)
        self.assertGreater(big_font.size, small_font.size)
        self.assertGreaterEqual(small_font.size, 44, "hook shrank below the legible floor")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

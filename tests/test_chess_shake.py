"""Tests for the camera shake on the decisive beats.

A shake that runs for a whole clip reads as a broken render; a shake on the
climax reads as impact. These cover the three ways that goes wrong, all of
which fail silently rather than loudly: the filter expression is rejected and
the clip renders as None, the expression is accepted but clamped into a no-op
and the bytes come out identical to no shake, and the amplitude is odd, which
yuv420p rounds into a lurch on one side only.

The render tests need ffmpeg; the rest do not.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from shorts_clipper.chess import clip
from shorts_clipper.chess.board import FrameSpec

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/6P1/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/6P1/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2",
]


def _have_ffmpeg() -> bool:
    try:
        from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

        return Path(ffmpeg_path()).is_file()
    except Exception:
        return False


def _seconds(stamp: str | None) -> float:
    if not stamp:
        return 0.0
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


HAVE_FFMPEG = _have_ffmpeg()


def _plan(shake=(), after: float = 0.6) -> clip.ChessClipPlan:
    return clip.ChessClipPlan(
        frames=[
            FrameSpec(fen=FENS[0], top_text="Ход 7… Bg4", bottom_text="смотри до хода"),
            FrameSpec(fen=FENS[1], top_text="Ход 7… Bg4", bottom_text="слон встал под бой",
                      accent="bad", arrow=("c8", "g4")),
            FrameSpec(fen=FENS[2], top_text="Ход 7… Bg4", bottom_text="теперь кон",
                      accent="bad"),
        ],
        before_seconds=0.6,
        move_seconds=0.6,
        after_seconds=after,
        shake_frames=shake,
    )


class ShakePlanTests(unittest.TestCase):
    """Everything here runs without ffmpeg."""

    def test_default_plan_shakes_nothing(self):
        plan = _plan()
        self.assertEqual(plan.shake_frames, ())
        self.assertIsNone(plan.holds, "shake must not have disturbed the explicit holds")

    def test_default_duration_is_unchanged(self):
        self.assertAlmostEqual(
            clip.ChessClipPlan(frames=[FrameSpec(fen=FENS[0], top_text="", bottom_text="")]).duration,
            clip.HOLD_BEFORE + clip.HOLD_MOVE + clip.HOLD_AFTER,
        )

    def test_duration_is_affected_by_neither_shake_nor_holds(self):
        self.assertAlmostEqual(_plan().duration, _plan(shake=(2,)).duration)
        paced = clip.ChessClipPlan(frames=[], holds=[1.5, 0.4], shake_frames=(0,))
        self.assertAlmostEqual(paced.duration, 1.9)

    def test_out_of_range_frame_index_is_accepted(self):
        plan = _plan(shake=(99, -1))
        self.assertEqual(plan.shake_frames, (99, -1))
        self.assertAlmostEqual(plan.duration, 1.8)

    def test_amplitude_is_in_the_visible_band_and_even(self):
        self.assertGreaterEqual(clip.SHAKE_PX, 3, "under 3px a shake is invisible on a phone")
        self.assertLessEqual(clip.SHAKE_PX, 6, "over 6px the squares stop being countable")
        # yuv420p rounds crop and pad offsets down to a whole chroma sample, so an
        # odd amplitude shakes by amplitude+1 on one side and amplitude-1 on the
        # other -- a lurch, not a jolt.
        self.assertEqual(clip.SHAKE_PX % 2, 0)

    def test_shake_decays_inside_the_beat(self):
        """A shake that does not settle reads as a broken render, not an impact."""
        settled = None
        for n in range(0, int(clip.FPS)):
            t = n / clip.FPS
            wave = clip.SHAKE_PX * math.exp(-clip.SHAKE_DECAY * t)
            if wave < 1.0:
                settled = t
                break
        self.assertIsNotNone(settled, "the shake never decays below a pixel")
        self.assertLess(
            settled, 0.5,
            f"the shake still moves at {settled:.2f}s, which is a vibration, not an impact",
        )

    def test_crop_window_stays_inside_the_padded_frame(self):
        """A crop clamped back to x=0 renders happily and shakes nothing."""
        filters = clip._shake_filters("0x000000")
        self.assertIn(f"pad=w={clip.CLIP_WIDTH + 2 * clip.SHAKE_PX}", filters)
        self.assertIn(f"crop=w={clip.CLIP_WIDTH}:h={clip.CLIP_HEIGHT}", filters)
        # The window is centred on the padded frame plus the wave, which is bounded
        # by SHAKE_PX, so it can never leave [0, 2 * SHAKE_PX].
        self.assertIn(f"x='{clip.SHAKE_PX}+round(", filters)
        self.assertIn(f"y='{clip.SHAKE_PX}+round(", filters)

    def test_pad_colour_is_the_frame_background(self):
        from shorts_clipper.chess.board import render_frame

        with tempfile.TemporaryDirectory() as tmp:
            png = render_frame(FrameSpec(fen=FENS[0], top_text="x", bottom_text="y"),
                               Path(tmp) / "f.png")
            colour = clip._shake_pad_colour(png)
        self.assertRegex(colour, r"^0x[0-9a-f]{6}$")
        self.assertEqual(len(colour), 8)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class ShakeRenderTests(unittest.TestCase):
    def test_shaken_clip_differs_from_an_unshaken_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = clip.render_clip(_plan(), root / "plain.mp4")
            shaken = clip.render_clip(_plan(shake=(2,)), root / "shaken.mp4")
            self.assertIsNotNone(plain, "unshaken render came back None")
            self.assertIsNotNone(shaken, "shaken render came back None -- the filter was rejected")
            self.assertNotEqual(
                plain.read_bytes(), shaken.read_bytes(),
                "the shake produced identical bytes, so the expression is a no-op",
            )

    def test_out_of_range_shake_index_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = clip.render_clip(_plan(shake=(99, -1)), Path(tmp) / "x.mp4")
            self.assertIsNotNone(out, "an index naming no frame must be ignored, not raised")

    def test_shake_keeps_the_audio_stream_and_the_length(self):
        from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

        def probe(path):
            info = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-i", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            duration = next(
                (line.split("Duration:")[1].split(",")[0].strip()
                 for line in info.stderr.splitlines() if "Duration:" in line),
                None,
            )
            decoded = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-i", str(path), "-map", "0:v",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=300,
            )
            counts = [
                int(line.split("frame=")[1].split()[0])
                for line in decoded.stderr.splitlines() if line.startswith("frame=")
            ]
            return duration, counts[-1] if counts else 0, info.stderr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = clip.render_clip(_plan(), root / "plain.mp4")
            shaken = clip.render_clip(_plan(shake=(2,)), root / "shaken.mp4")
            self.assertIsNotNone(plain)
            self.assertIsNotNone(shaken)

            plain_dur, plain_frames, plain_err = probe(plain)
            shaken_dur, shaken_frames, shaken_err = probe(shaken)

            # At a fixed frame rate the frame count is the length of the clip. The
            # container duration is not: it is the tail of the silent AAC track,
            # which x264's frame-level threading pads a frame or two differently
            # from run to run, so it is compared with slack.
            self.assertGreater(plain_frames, 0)
            self.assertEqual(plain_frames, shaken_frames, "the shake changed the clip length")
            self.assertLessEqual(
                abs(_seconds(plain_dur) - _seconds(shaken_dur)), 0.5,
                f"container duration moved: {plain_dur} -> {shaken_dur}",
            )
            for label, err in (("plain", plain_err), ("shaken", shaken_err)):
                self.assertIn("Stream #0:0", err, f"{label}: no video stream")
                self.assertIn("Stream #0:1", err, f"{label}: no audio stream")

    def test_shaken_beat_moves_and_the_others_do_not(self):
        """Measured on the pixels: only the named beat shifts, and it settles."""
        import numpy as np
        import PIL.Image

        from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

        def frames(path, out):
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [ffmpeg_path(), "-y", "-v", "error", "-i", str(path), "-vsync", "0",
                 str(out / "%05d.png")],
                check=True, timeout=300, capture_output=True,
            )
            return [np.asarray(PIL.Image.open(p).convert("L"), dtype=np.float32)
                    for p in sorted(out.glob("*.png"))]

        def shift(a, b, reach):
            best = None
            for dy in range(-reach, reach + 1):
                for dx in range(-reach, reach + 1):
                    err = float(np.abs(b[400:1500, 240:840] - a[400 - dy:1500 - dy, 240 - dx:840 - dx]).mean())
                    if best is None or err < best[0]:
                        best = (err, dx, dy)
            return best

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = clip.render_clip(_plan(), root / "plain.mp4")
            shaken = clip.render_clip(_plan(shake=(2,)), root / "shaken.mp4")
            self.assertIsNotNone(plain)
            self.assertIsNotNone(shaken)
            a = frames(plain, root / "a")
            b = frames(shaken, root / "b")
            per_beat = int(round(0.6 * clip.FPS))
            self.assertGreaterEqual(len(a), 3 * per_beat)
            self.assertGreaterEqual(len(b), 3 * per_beat)

            # A shake is a translation of the whole board, so the best match for a
            # moved frame is the still frame at the matching integer offset.
            first = shift(a[2 * per_beat], b[2 * per_beat], clip.SHAKE_PX + 1)
            self.assertGreater(
                max(abs(first[1]), abs(first[2])), 0,
                "the first frame of the shaken beat did not move at all",
            )
            self.assertLessEqual(
                max(abs(first[1]), abs(first[2])), clip.SHAKE_PX,
                "the shake exceeded the amplitude it is supposed to have",
            )

            # ... and it has to be back to zero well before the hold is over.
            last = 2 * per_beat + per_beat - 1
            self.assertEqual(
                shift(a[last], b[last], clip.SHAKE_PX + 1)[1:], (0, 0),
                "the shake was still running at the end of the hold",
            )

            # The beats that were not named must be untouched.
            for beat in (0, 1):
                lo, hi = beat * per_beat, (beat + 1) * per_beat
                for n in range(lo, hi):
                    self.assertEqual(
                        float(np.abs(a[n] - b[n]).max()), 0.0,
                        f"beat {beat} frame {n} moved although it was not named for a shake",
                    )


if __name__ == "__main__":
    unittest.main()
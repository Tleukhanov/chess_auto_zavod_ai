"""Tests for the chess clip assembler.

The clip is the first thing that runs ffmpeg for the chess channel, so these
cover the two failures that actually happened: a plan with no frames, and a
video-only mp4 that the later bed mix could not attach audio to.
"""

from __future__ import annotations

import random
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper.chess import analysis, clip
from shorts_clipper.chess.board import FrameSpec


def _blunder_game(seed: int = 20261005, want_cp: int = 400):
    rng = random.Random(seed)
    board = chess.Board()
    history: list[str] = []
    while len(history) < 120:
        if board.is_game_over(claim_draw=False):
            break
        white_to_move = board.turn == chess.WHITE
        before = analysis.material_eval(board.fen())
        worst = None
        for mv in board.legal_moves:
            san = board.san(mv)
            after = board.copy()
            after.push(mv)
            delta = analysis.material_eval(after.fen()) - before
            delta -= analysis._best_capture_gain(chess, after)
            if not white_to_move:
                delta = -delta
            if worst is None or delta < worst[0]:
                worst = (delta, san)
        if worst[0] <= -want_cp:
            game = chess.pgn.Game()
            node = game
            for s in [*history, worst[1]]:
                node = node.add_variation(node.board().parse_san(s))
            return game
        san = board.san(rng.choice(list(board.legal_moves)))
        history.append(san)
        board.push_san(san)
    return None


class ClipPlanTests(unittest.TestCase):
    def test_empty_plan_renders_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = clip.render_clip(clip.ChessClipPlan(frames=[]), Path(tmp) / "x.mp4")
            self.assertIsNone(out)

    def test_duration_is_sum_of_holds(self):
        plan = clip.ChessClipPlan(
            frames=[FrameSpec(fen=analysis._START_FEN, top_text="", bottom_text="")],
            before_seconds=1.0,
            move_seconds=2.0,
            after_seconds=3.0,
        )
        self.assertAlmostEqual(plan.duration, 6.0)

    def test_add_bed_without_music_returns_a_real_file(self):
        """Returning a non-existent path leaves the caller unable to detect failure."""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            src.write_bytes(b"not really a video")
            out = clip.add_bed(src, None, Path(tmp) / "out.mp4")
            self.assertTrue(out.exists(), f"returned a path that does not exist: {out}")
            self.assertEqual(out, src, "unwritable copy must fall back to the input")

    def test_add_bed_same_path_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            src.write_bytes(b"not really a video")
            out = clip.add_bed(src, None, src)
            self.assertEqual(out, src)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class ClipRenderTests(unittest.TestCase):
    def test_plan_has_three_beats_and_arrow_on_the_move(self):
        game = _blunder_game()
        self.assertIsNotNone(game, "no blunder >=400cp found")
        dec = analysis.find_decisions(game, limit=1, min_ply=0)[0]
        plan = clip.plan_for_decision(dec)
        self.assertEqual(len(plan.frames), 3)
        arrows = [f.arrow for f in plan.frames]
        self.assertIsNone(arrows[0])
        self.assertIsNotNone(arrows[1], "the move beat must carry the arrow")
        self.assertIsNone(arrows[2])
        # The last beat shows the position after the move, the first two before.
        self.assertEqual(plan.frames[0].fen, dec.fen_before)
        self.assertEqual(plan.frames[1].fen, dec.fen_before)
        self.assertEqual(plan.frames[2].fen, dec.fen_after)

    def test_render_produces_video_and_audio_streams(self):
        """Regression: the clip was written video-only, so the bed mix failed.

        add_bed maps [0:a]; on a stream-less input ffmpeg refuses with
        "matches no streams", and the mix silently degraded to no audio at all.
        """
        from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

        game = _blunder_game()
        self.assertIsNotNone(game)
        dec = analysis.find_decisions(game, limit=1, min_ply=0)[0]
        plan = clip.plan_for_decision(dec)
        plan.before_seconds = 0.6
        plan.move_seconds = 0.6
        plan.after_seconds = 0.6

        with tempfile.TemporaryDirectory() as tmp:
            out = clip.render_clip(plan, Path(tmp) / "c.mp4")
            self.assertIsNotNone(out, "render_clip returned None")
            probe = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-i", str(out)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertIn("Stream #0:0", probe.stderr, "no video stream")
            self.assertIn("Stream #0:1", probe.stderr, "no audio stream")

    def test_bed_mix_succeeds_on_a_rendered_clip(self):
        import wave

        from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

        game = _blunder_game()
        self.assertIsNotNone(game)
        dec = analysis.find_decisions(game, limit=1, min_ply=0)[0]
        plan = clip.plan_for_decision(dec)
        plan.before_seconds = 0.5
        plan.move_seconds = 0.5
        plan.after_seconds = 0.5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            silent = clip.render_clip(plan, root / "s.mp4")
            self.assertIsNotNone(silent)

            # A 6s sine WAV as the stand-in bed.
            music = root / "bed.wav"
            import math
            import struct

            sr = 44100
            frames = bytearray()
            for i in range(sr * 6):
                v = int(12000 * math.sin(2 * math.pi * 220 * i / sr))
                frames += struct.pack("<h", v)
            with wave.open(str(music), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(bytes(frames))

            out = clip.add_bed(silent, music, root / "m.mp4")
            self.assertEqual(out.name, "m.mp4", "bed mix fell back to the input")
            probe = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-i", str(out)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertIn("Stream #0:1", probe.stderr)


if __name__ == "__main__":
    unittest.main()
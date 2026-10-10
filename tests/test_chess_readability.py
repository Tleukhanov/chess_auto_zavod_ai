"""Regression tests for reading speed and for whose move it is.

Both came from watching a delivered clip rather than from reading the plan:

- The skim ran at 0.125s a frame, which put a whole 52-ply game through in 1.1s.
  The viewer could not tell which side was moving, which is the one thing a chess
  position has to convey. The floor is now a readable hold, and the *frame count*
  is what falls as the game grows, not the frame duration.
- Nothing in the frame said whose turn it was. ``FrameInfo`` had no field for it at
  all, so a normal move and the opponent walking into something looked alike.
"""

from __future__ import annotations

import unittest

from shorts_clipper.chess import analysis, clip, pacing
from tests.test_chess_pacing import HAVE_CHESS, _random_game


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class SkimReadabilityTests(unittest.TestCase):
    def _plan(self, plies: int, seed: int):
        game = _random_game(plies=plies, seed=seed)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            return None, None
        return pacing.plan_for_highlight(game, decisions[0]), game

    def test_no_skim_frame_is_unreadably_short(self):
        plan, _ = self._plan(52, 3)
        if plan is None:
            self.skipTest("no decisive move in the fixture game")

        skim = [h for h, f in zip(plan.holds, plan.frames, strict=True) if f.accent != "bad"]
        self.assertTrue(skim)
        self.assertGreaterEqual(
            min(skim),
            0.30,
            "a skim frame under 0.3s cannot be read, so the clip flickers",
        )

    def test_the_whole_game_is_not_still_a_flicker(self):
        """The defect in absolute terms: the whole game in about a second."""
        plan, _ = self._plan(52, 3)
        if plan is None:
            self.skipTest("no decisive move in the fixture game")

        skim = [h for h, f in zip(plan.holds, plan.frames, strict=True) if f.accent != "bad"]
        self.assertGreater(
            sum(skim), 2.0, "the entire game passes in under two seconds"
        )

    def test_length_stays_short_shape(self):
        for plies, seed in ((40, 5), (90, 6), (170, 7)):
            with self.subTest(plies=plies):
                plan, _ = self._plan(plies, seed)
                if plan is None:
                    continue
                self.assertLessEqual(plan.duration, pacing.MAX_CLIP_SECONDS + 0.01)
                self.assertGreaterEqual(plan.duration, pacing.MIN_CLIP_SECONDS - 0.01)

    def test_a_long_game_drops_skim_frames_rather_than_speeding_up(self):
        """Length pressure must cost frames, not legibility."""
        short, _ = self._plan(40, 8)
        long, _ = self._plan(170, 9)
        if short is None or long is None:
            self.skipTest("no decisive move in a fixture game")

        def skim_holds(plan):
            return [h for h, f in zip(plan.holds, plan.frames, strict=True) if f.accent != "bad"]

        self.assertGreaterEqual(
            min(skim_holds(long)), min(skim_holds(short)) - 0.01,
            "a longer game was made to go faster rather than to cut frames",
        )

    def test_the_reveal_survives_trimming(self):
        plan, _ = self._plan(170, 9)
        if plan is None:
            self.skipTest("no decisive move in the fixture game")
        self.assertTrue(
            [f for f in plan.frames if f.arrow], "trimming removed the reveal"
        )


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class TurnLabelTests(unittest.TestCase):
    START = analysis.load_pgn_file  # keeps the import honest for the skip guard

    def test_white_to_move(self):
        side, _ = clip.turn_label("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual(side, "ходят белые")

    def test_black_to_move(self):
        side, _ = clip.turn_label("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        self.assertEqual(side, "ходят чёрные")

    def test_named_from_headers(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        side, name = clip.turn_label(
            fen, {"White": "Anderssen", "Black": "Kieseritzky"}
        )
        self.assertEqual(side, "ходят белые")
        self.assertEqual(name, "Anderssen")

        side, name = clip.turn_label(
            fen, {"White": "Anderssen", "Black": "Kieseritzky"}
        )
        self.assertEqual(name, "Anderssen")

    def test_black_name_follows_the_turn(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        _, name = clip.turn_label(fen, {"White": "Anderssen", "Black": "Kieseritzky"})
        self.assertEqual(name, "Kieseritzky")

    def test_missing_headers_keep_the_side(self):
        """The side is the half that carries the meaning."""
        for headers in (None, {}, {"Event": "x"}):
            with self.subTest(headers=headers):
                side, name = clip.turn_label(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", headers
                )
                self.assertEqual(side, "ходят белые")
                self.assertEqual(name, "")

    def test_unparseable_fen_is_not_guessed(self):
        self.assertEqual(clip.turn_label("not a fen"), ("", ""))

    def test_every_frame_names_a_side(self):
        game = _random_game(plies=60, seed=12)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")

        plan = pacing.plan_for_highlight(game, decisions[0])
        for frame in plan.frames:
            self.assertTrue(
                frame.info.turn_side,
                f"frame at ply {frame.info.progress} does not say whose move it is",
            )

    def test_the_skim_alternates_sides(self):
        """A label that never changes reads as a bug rather than as information."""
        game = _random_game(plies=60, seed=13)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")

        plan = pacing.plan_for_highlight(game, decisions[0])
        skim = [f.info.turn_side for f in plan.frames if f.accent != "bad"]
        self.assertGreater(
            len(set(skim)), 1, "every skim frame named the same side to move"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
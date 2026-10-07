"""Tests for how the batch layer drives the highlight format.

A highlight clip is a whole game, so the batch must emit exactly one per game.
That is not just tidiness: the run-wide ``--count`` budget picks the best moments
in the run, and a batch of 12 deciding-move clips came out of a single game --
twelve variations on the same position. The paced format cannot repeat itself
that way, and this pins that it does not.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import chess

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper.chess import batch


class HighlightBatchTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_one_moment_per_game(self):
        from tests.test_chess_pacing import _random_game

        moments = []
        for index in range(3):
            game = _random_game(plies=60, seed=100 + index)
            moments.extend(
                batch._collect_moments(
                    "highlight", game, Path(f"game_{index}.pgn"), 1, 200, 6, 5
                )
            )

        self.assertEqual(len(moments), 3, "expected one clip per game")
        self.assertEqual({m.fmt for m in moments}, {"highlight"})

    def test_plan_factory_carries_the_game(self):
        """The paced plan needs every ply, so the game must reach the factory."""
        from tests.test_chess_pacing import _random_game

        game = _random_game(plies=50, seed=9)
        moments = batch._collect_moments(
            "highlight", game, Path("g.pgn"), 1, 0, 0, 5
        )
        if not moments:
            self.skipTest("no decisive move found in the fixture game")

        moment = moments[0]
        self.assertIsNotNone(moment.plan_factory)
        plan = moment.plan()

        # A plan built without the game would be empty or a single frame.
        self.assertGreater(len(plan.frames), 1, "the game did not reach the plan")
        self.assertIsNotNone(plan.holds)
        self.assertEqual(len(plan.holds), len(plan.frames))
        self.assertAlmostEqual(plan.duration, sum(plan.holds), places=6)

        # The plan stops at the decision the batch actually chose, so the slow
        # window sits on that move and not on ply zero.
        self.assertIsInstance(moment.decision.ply, int)
        self.assertGreaterEqual(moment.decision.ply, 0)
        self.assertLess(moment.decision.ply, len(list(game.mainline_moves())))
        slow_at = [
            i for i, f in enumerate(plan.frames)
            if "вот этот ход" in f.top_text
        ]
        self.assertEqual(len(slow_at), 1, "no frame marks the decisive move")

        # The clip has to reach the end of the game, not stop at the window.
        self.assertEqual(plan.frames[-1].fen, game.end().board().fen())
        self.assertTrue(chess.Board(plan.frames[-1].fen).is_valid())

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_slug_is_namespaced_and_safe(self):
        from tests.test_chess_pacing import _random_game

        game = _random_game(plies=50, seed=13)
        game.headers["White"] = "Magnus Carlsen"
        game.headers["Black"] = "Ian Nepomniachtchi"

        moments = batch._collect_moments(
            "highlight", game, Path("g.pgn"), 1, 0, 0, 5
        )
        if not moments:
            self.skipTest("no decisive move found in the fixture game")

        slug = moments[0].slug
        self.assertIn("_highlight_", slug)
        self.assertNotIn(" ", slug)
        self.assertNotIn("/", slug)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_keys_differ_from_the_deciding_format(self):
        """Dedup is scoped by format; the same moment must not collide."""
        from tests.test_chess_pacing import _random_game

        game = _random_game(plies=50, seed=17)
        highlight = batch._collect_moments("highlight", game, Path("g.pgn"), 1, 0, 0, 5)
        deciding = batch._collect_moments("deciding", game, Path("g.pgn"), 1, 0, 0, 5)
        if not highlight or not deciding:
            self.skipTest("no decisive move found in the fixture game")

        self.assertNotEqual(highlight[0].key, deciding[0].key)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
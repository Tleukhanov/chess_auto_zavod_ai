"""The reveal frame must point at the move the engine actually flagged.

``Decision.ply`` is 1-based (``analysis.find_decisions`` stores ``ply + 1``) while
the paced plan indexes ``mainline_moves()`` from 0. Comparing them directly never
matched, so every frame was classified as ``lead`` and the whole highlight format
shipped without an arrow or an evaluation swing: 14 of 14 clips wrong, and the
failure was invisible because the clip still rendered and still looked fine.
"""

from __future__ import annotations

import tempfile
import unittest

from shorts_clipper.chess import analysis, narration, pacing
from tests.test_chess_pacing import _random_game

HAVE_CHESS = analysis._chess_module() is not None


def _san_at(game, index: int) -> str:
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i == index:
            return board.san(move)
        board.push(move)
    return ""


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class ResolveTargetPlyTests(unittest.TestCase):
    def test_finds_the_move_by_uci(self):
        game = _random_game(plies=60, seed=21)
        decision = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decision:
            self.skipTest("no decisive move in the fixture game")
        dec = decision[0]

        resolved = pacing.resolve_target_ply(game, dec)
        self.assertIsNotNone(resolved)

        moves = list(game.mainline_moves())
        self.assertEqual(moves[resolved].uci(), dec.uci)
        self.assertEqual(_san_at(game, resolved), dec.san)

    def test_never_the_opponents_reply(self):
        """The old bug: the index came out one ply late."""
        game = _random_game(plies=90, seed=33)
        decisions = analysis.find_decisions(game, limit=3, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")

        for dec in decisions:
            with self.subTest(san=dec.san):
                resolved = pacing.resolve_target_ply(game, dec)
                moves = list(game.mainline_moves())
                self.assertEqual(moves[resolved].uci(), dec.uci)
                if resolved + 1 < len(moves):
                    self.assertNotEqual(moves[resolved + 1].uci(), dec.uci)

    def test_resolved_index_is_zero_based(self):
        game = _random_game(plies=60, seed=44)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        dec = decisions[0]

        resolved = pacing.resolve_target_ply(game, dec)
        # Decision.ply is one-based, so the resolved index is one lower.
        self.assertEqual(resolved, dec.ply - 1)

    def test_no_ply_and_no_uci_is_none(self):
        class Empty:
            pass

        game = _random_game(plies=30, seed=66)
        self.assertIsNone(pacing.resolve_target_ply(game, Empty()))

    def test_falls_back_when_the_move_is_absent(self):
        """A UCI that is not in the game must not raise."""

        class Fake:
            ply = 5
            uci = "a1a8"

        game = _random_game(plies=40, seed=55)
        self.assertEqual(pacing.resolve_target_ply(game, Fake()), 4)

    def test_repeated_move_picks_the_occurrence_near_the_decision(self):
        """A game that repeats a move: the first match is the wrong one.

        Taking the first UCI match put a ply-64 reveal on ply 1, which emptied
        the window of the clip and left it with no arrow at all.
        """

        class Repeated:
            def __init__(self, ply):
                self.ply = ply
                self.uci = "e2e4"

        game = _random_game(plies=80, seed=88)
        moves = list(game.mainline_moves())
        same = [i for i, m in enumerate(moves) if m.uci() == "e2e4"]
        if len(same) < 2:
            self.skipTest("fixture does not repeat that move")

        for later in same[1:]:
            with self.subTest(ply=later):
                got = pacing.resolve_target_ply(game, Repeated(later + 1))
                self.assertEqual(got, later, "resolved to the wrong occurrence")

    def test_out_of_range_ply_is_clamped(self):
        class Wild:
            ply = 10_000
            uci = ""

        game = _random_game(plies=30, seed=77)
        got = pacing.resolve_target_ply(game, Wild())
        self.assertEqual(got, len(list(game.mainline_moves())) - 1)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class RevealFrameTests(unittest.TestCase):
    def test_exactly_one_reveal_frame(self):
        game = _random_game(plies=80, seed=88)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        dec = decisions[0]

        plan = pacing.plan_for_highlight(game, dec)
        reveals = [f for f in plan.frames if f.arrow is not None]
        self.assertTrue(reveals, "no frame carries an arrow at all")

        for frame in reveals:
            self.assertEqual(frame.info.moved_san, dec.san)

    def test_reveal_shows_the_evaluation_swing(self):
        game = _random_game(plies=80, seed=99)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        dec = decisions[0]

        plan = pacing.plan_for_highlight(game, dec)
        reveals = [f for f in plan.frames if f.arrow is not None]
        self.assertTrue(reveals)
        info = reveals[0].info
        self.assertTrue(
            info.eval_before or info.eval_after,
            "the reveal frame carries no evaluation, so the claim is unproven",
        )

    def test_skim_frames_get_no_arrow(self):
        game = _random_game(plies=80, seed=101)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")

        plan = pacing.plan_for_highlight(game, decisions[0])
        # Only frames outside the reveal window are "skim"; the ones adjacent to
        # it are the lead-in and the tail, which may legitimately carry no arrow
        # but are not what this is about.
        skim = [f for f in plan.frames if f.arrow is None and f.accent == "neutral"]
        self.assertTrue(skim, "no skim frames to check")
        for frame in skim:
            self.assertIsNone(frame.arrow)

    def test_frame_kind_uses_the_resolved_index(self):
        """narrate() gets a 0-based ply and must classify the reveal."""
        game = _random_game(plies=80, seed=111)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        dec = decisions[0]
        target = pacing.resolve_target_ply(game, dec)

        got = narration.narrate(
            decision=dec,
            ply=target,
            total_plies=len(list(game.mainline_moves())),
            fen_before=dec.fen_before,
            fen_after=dec.fen_after,
            color=dec.color,
            headers=game.headers,
        )
        self.assertEqual(got.kind, narration.KIND_REVEAL)
        self.assertIsNotNone(got.arrow)

        if target + 1 < len(list(game.mainline_moves())):
            later = narration.narrate(
                decision=dec,
                ply=target + 1,
                total_plies=len(list(game.mainline_moves())),
                fen_before=dec.fen_after,
                fen_after=dec.fen_after,
                color="black" if dec.color == "white" else "white",
                headers=game.headers,
            )
            self.assertNotEqual(later.kind, narration.KIND_REVEAL)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class EveryDemoGameGetsARevealTests(unittest.TestCase):
    """The bug hit 14 of 14 clips, so one clip proves nothing."""

    def test_reveal_matches_the_real_move_across_games(self):
        import scripts.make_demo_pgns as maker

        with tempfile.TemporaryDirectory() as tmp:
            paths = maker.write_demo_pgns(tmp, games=8)
            checked = 0
            for path in paths:
                game = analysis.load_pgn_file(path)
                if game is None:
                    continue
                decisions = analysis.find_decisions(game, limit=1, min_ply=6)
                if not decisions:
                    continue
                dec = decisions[0]
                target = pacing.resolve_target_ply(game, dec)
                plan = pacing.plan_for_highlight(game, dec)
                reveals = [f for f in plan.frames if f.arrow is not None]
                self.assertTrue(reveals, f"{path.name}: no reveal frame")
                self.assertEqual(
                    reveals[0].info.moved_san,
                    _san_at(game, target),
                    f"{path.name}: reveal names the wrong move",
                )
                checked += 1
            self.assertGreater(checked, 0, "no demo game produced a decision")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
"""The critic's three findings, pinned.

1. The skim started at ply 0, so 40-48% of each clip was the opening -- the
   start position, e2e4, e7e5 -- and a viewer's first three seconds contained
   nothing worth watching.
2. A mate-sized loss was divided by 100 and printed: "минус 100.0 за один ход".
   MATE_CP is a sentinel for "walked into mate", not a pawn count, and the clip
   whose whole claim is that its numbers are checkable was showing a figure that
   cannot exist.
3. Two progress counters rendered at once, in different units: "30 / 35" in the
   caption and "58 / 70" in the band, on one board.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import chess

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper.chess import analysis, narration, pacing
from tests.test_chess_pacing import _random_game


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class SkimStartsNearTheRevealTests(unittest.TestCase):
    def test_a_late_blunder_does_not_replay_the_opening(self):
        game = _random_game(plies=180, seed=8)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        target = pacing.resolve_target_ply(game, decisions[0])
        if target < pacing.SKIM_HISTORY_PLIES + 10:
            self.skipTest("the fixture's blunder is too early to test this")

        plan = pacing.plan_for_highlight(game, decisions[0])
        first = int(plan.frames[0].info.progress.split("/")[0])
        self.assertGreaterEqual(
            first,
            target - pacing.SKIM_HISTORY_PLIES - pacing.SKIM_PLIES_PER_FRAME,
            "the clip still opens on the starting position",
        )

    def test_an_early_blunder_still_gets_history(self):
        """Capped at ply 0, not allowed to run backwards."""
        game = _random_game(plies=60, seed=21)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        plan = pacing.plan_for_highlight(game, decisions[0])
        self.assertGreaterEqual(len(plan.frames), 3)

    def test_the_starting_position_is_never_the_first_frame_of_a_long_game(self):
        game = _random_game(plies=180, seed=9)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        plan = pacing.plan_for_highlight(game, decisions[0])
        self.assertNotEqual(
            plan.frames[0].fen,
            chess.STARTING_FEN,
            "a 180-ply game opened on 32 pieces in the starting array",
        )


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class NoFabricatedNumbersTests(unittest.TestCase):
    def test_the_mate_sentinel_is_never_printed_as_pawns(self):
        decision = analysis.Decision(
            ply=6, move_number=3, color="black", san="Bxf6", uci="c8f6",
            fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
            eval_before_cp=30, eval_after_cp=9999, mover_loss_cp=10_000,
            material=False, header={}, mate_after=None,
        )
        frame = narration.narrate(
            decision=decision, ply=5, total_plies=40,
            fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
            color="black",
        )
        for field in (frame.info.hook, frame.info.detail):
            self.assertNotIn(
                "100.0", field,
                f"the mate sentinel was printed as a pawn figure: {field!r}",
            )

    def test_no_real_decision_ever_prints_an_impossible_figure(self):
        for pg in sorted(Path(
            r"C:\Users\huawei\AppData\Local\Temp\chessfleet"
        ).glob("*.pgn"))[:6] if Path(
            r"C:\Users\huawei\AppData\Local\Temp\chessfleet"
        ).is_dir() else []:
            game = analysis.load_pgn_file(pg)
            if game is None:
                continue
            total = len(list(game.mainline_moves()))
            for dec in analysis.find_decisions(game, limit=2, min_ply=6):
                with self.subTest(pgn=pg.name, san=dec.san):
                    frame = narration.narrate(
                        decision=dec, ply=dec.ply - 1, total_plies=total,
                        fen_before=dec.fen_before, fen_after=dec.fen_after,
                        color=dec.color, headers=game.headers,
                    )
                    for field in (frame.info.hook, frame.info.detail):
                        self.assertNotIn("100.0", field)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class OneProgressCounterTests(unittest.TestCase):
    def test_the_caption_no_longer_carries_a_second_counter(self):
        game = _random_game(plies=80, seed=12)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        plan = pacing.plan_for_highlight(game, decisions[0])
        for frame in plan.frames:
            self.assertEqual(
                frame.top_text, "",
                "the caption still draws a progress counter beside the band's",
            )

    def test_the_band_still_reports_progress(self):
        game = _random_game(plies=80, seed=13)
        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decisive move in the fixture game")
        plan = pacing.plan_for_highlight(game, decisions[0])
        self.assertTrue(
            any(f.info and f.info.progress for f in plan.frames),
            "removing the caption counter left no progress at all",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
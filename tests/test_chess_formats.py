"""Tests for the opening and endgame content formats.

One real bug is pinned here: _piece_complex compared against material *values*
instead of python-chess ``piece_type`` codes and left the kings in the lists, so
every endgame position returned None and the format silently produced nothing.
"""

from __future__ import annotations

import unittest

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper.chess import clip, formats


def _game_from(sans: list[str]):
    game = chess.pgn.Game()
    node = game
    for san in sans:
        node = node.add_variation(node.board().parse_san(san))
    text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    from shorts_clipper.chess import analysis

    return analysis.load_pgn(text)


class OpeningTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_main_lines_classify(self):
        cases = [
            (["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"], "Сицилианская"),
            (["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], "Испанская"),
            (["e4", "e5", "Nf3", "Nc6", "Bc4"], "Итальянская"),
            (["d4", "Nf6", "c4", "g6", "Nc3", "d5"], "Кислова"),
            (["d4", "d5", "c4", "c6"], "Славянская"),
            (["e4", "c6", "d4", "d5"], "Каро-Канн"),
            (["e4", "e6", "d4", "d5"], "Французская"),
            (["e4", "e5", "Bc4", "Nf6"], "Итальянская"),
            (["d4", "d5", "c4", "e6"], "Славы"),
        ]
        for sans, expected in cases:
            with self.subTest(line=" ".join(sans)):
                got = formats.classify_opening(_game_from(sans))
                self.assertIsNotNone(got, f"no name for {' '.join(sans)}")
                self.assertIn(expected, got)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_specific_line_wins_over_the_general_one(self):
        """1.e4 c5 must not be reported as a plain open game."""
        self.assertIn("Сицилианская", formats.classify_opening(_game_from(["e4", "c5", "Nf3"])))
        self.assertIn("королевский", formats.classify_opening(_game_from(["e4", "e5", "Nf3"])).lower())

    def test_none_game(self):
        self.assertIsNone(formats.classify_opening(None))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_unknown_opening_returns_none(self):
        self.assertIsNone(formats.classify_opening(_game_from(["Nh3", "a6", "Rg1"])))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_opening_moment_feeds_the_clip_planner_unchanged(self):
        """The whole point: formats must not require editing clip.py."""
        game = _game_from(
            ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6",
             "Be3", "e5", "Nb3", "Be6", "f3", "Be7"]
        )
        from shorts_clipper.chess import analysis

        decisions = analysis.find_decisions(game, limit=1, min_ply=6)
        if not decisions:
            self.skipTest("no decision in this line")
        moment = formats.opening_moment(decisions[0], formats.classify_opening(game))
        self.assertIsNotNone(moment)
        self.assertIn("Сицилианская", moment.caption())
        plan = clip.plan_for_decision(moment)
        self.assertEqual(len(plan.frames), 3)

    def test_opening_moment_none_decision(self):
        self.assertIsNone(formats.opening_moment(None, "x"))


class EndgameTests(unittest.TestCase):
    K_R = "8/8/8/3k4/8/3K4/3R4/8 w - - 0 1"
    K_Q = "8/8/8/3k4/8/3K4/3Q4/8 w - - 0 1"
    K_B = "8/8/8/3k4/8/3K4/3B4/8 w - - 0 1"
    K_B_VS_KN = "8/8/8/3k4/8/3K4/3B4/8 w - - 0 1"
    R_VS_B = "8/8/8/3b4/8/3K4/3R4/8 w - - 0 1"
    Q_VS_R = "8/8/8/3r4/8/3K4/3Q4/8 w - - 0 1"
    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_named_complexes(self):
        """Regression: piece_type codes, kings stripped -- everything used to be None."""
        cases = [
            (self.K_R, "Ладья против короля"),
            (self.K_Q, "Ферзь против короля"),
            (self.K_B, "Слон против короля"),
            (self.R_VS_B, "Ладья против слона"),
            (self.Q_VS_R, "Ферзь против ладьи"),
        ]
        for fen, expected in cases:
            with self.subTest(fen=fen):
                got = formats.classify_endgame(fen)
                self.assertIsNotNone(got, f"no complex for {fen}")
                self.assertEqual(got[0], expected)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_opening_position_is_not_an_endgame(self):
        self.assertIsNone(formats.classify_endgame(self.START))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_piece_count_limit(self):
        """Above the limit the position is a middlegame and must be declined."""
        busy = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
        self.assertIsNone(formats.classify_endgame(busy))
        self.assertGreater(formats.ENDGAME_PIECE_LIMIT, 0)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_unsupported_complex_returns_none_rather_than_guessing(self):
        # King + two pawns against king + pawn: real, but not in the table.
        two_sided = "8/5p1k/8/8/8/7K/6PP/8 w - - 0 1"
        self.assertIsNone(formats.classify_endgame(two_sided))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_verdict_is_chess_correct_when_an_engine_is_present(self):
        """K+B vs K is a draw; K+R vs K is a win. A wrong answer here is worse
        than no answer, because the channel would state something false."""
        from shorts_clipper.chess import engine as engine_mod

        if not engine_mod.has_engine():
            self.skipTest("no engine installed")
        r = formats.classify_endgame(self.K_R)
        b = formats.classify_endgame(self.K_B)
        self.assertEqual(r[1], "выигрыш", "K+R vs K is won")
        self.assertEqual(b[1], "ничья", "K+B vs K is drawn")

    def test_endgame_moment_none_on_bad_fen(self):
        self.assertIsNone(formats.endgame_moment("not a fen"))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_endgame_moment_holds_one_position(self):
        moment = formats.endgame_moment(self.K_Q)
        self.assertIsNotNone(moment)
        self.assertIn("Ферзь", moment.caption())
        plan = clip.plan_for_decision(moment)
        self.assertEqual(len(plan.frames), 3)
        # A null move flips the side to move and resets the clock, so the FENs
        # differ there by design. The position itself must not change.
        before = chess.Board(plan.frames[0].fen)
        after = chess.Board(plan.frames[2].fen)
        self.assertEqual(before.board_fen(), after.board_fen())
        self.assertNotEqual(plan.frames[0].fen, plan.frames[2].fen)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_determinism(self):
        first = formats.classify_endgame(self.K_R)
        second = formats.classify_endgame(self.K_R)
        self.assertEqual(first, second)


class FormatRegistryTests(unittest.TestCase):
    def test_available_formats(self):
        self.assertIn("deciding-move", formats.available_formats())
        self.assertIn("opening", formats.available_formats())
        self.assertIn("endgame", formats.available_formats())


if __name__ == "__main__":
    unittest.main()
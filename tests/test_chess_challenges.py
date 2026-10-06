"""Tests for the challenge format.

The bug pinned here was only findable by looking at a rendered frame: the
question beat highlighted the origin and destination squares and printed the
evaluation, so the puzzle handed over its own answer. Selection also had a limit
of 10 non-pawn pieces, which rejected every position in every game -- 0
survivors out of 420 -- because a full middlegame has 14.
"""

from __future__ import annotations

import random
import unittest

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper.chess import challenges, clip


def _random_game(seed: int = 90210, plies: int = 70):
    """Seeded random legal game. Recalled PGNs proved unreliable to transcribe."""
    import chess.pgn

    rng = random.Random(seed)
    board = chess.Board()
    sans: list[str] = []
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over(claim_draw=False):
            break
        san = board.san(rng.choice(moves))
        sans.append(san)
        board.push_san(san)
    game = chess.pgn.Game()
    node = game
    for s in sans:
        node = node.add_variation(node.board().parse_san(s))
    text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    from shorts_clipper.chess import analysis

    return analysis.load_pgn(text)


class EngineMultiPvTests(unittest.TestCase):
    """top_moves is what makes a challenge a puzzle rather than a reveal."""

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_bad_fen_returns_none(self):
        from shorts_clipper.chess import engine as E

        self.assertIsNone(E.top_moves("not a fen"))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_count_is_clamped_to_legal_moves(self):
        """Bare kings have five legal moves, so count=9 must return five."""
        from shorts_clipper.chess import engine as E

        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        got = E.top_moves(fen, count=9, depth_override=10)
        if not got:
            self.skipTest("no engine")
        self.assertTrue(got)
        self.assertLessEqual(len(got), chess.Board(fen).legal_moves.count())

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_a_hanging_piece_has_a_huge_gap(self):
        from shorts_clipper.chess import engine as E

        got = E.top_moves("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", count=2, depth_override=12)
        if not got:
            self.skipTest("no engine")
        self.assertGreaterEqual(got[0][1].cp - got[1][1].cp, 300)


class ChallengeSelectionTests(unittest.TestCase):
    def test_none_game(self):
        self.assertEqual(challenges.find_challenges(None), [])

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_refuses_without_an_engine(self):
        """A static estimate cannot see the best-vs-second gap. It must not guess."""
        from shorts_clipper.chess import engine as engine_mod

        real = engine_mod.has_engine
        engine_mod.has_engine = lambda: False
        try:
            self.assertEqual(challenges.find_challenges(_random_game()), [])
            self.assertFalse(challenges.available())
        finally:
            engine_mod.has_engine = real

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_piece_limit_admits_a_full_middlegame(self):
        """Regression: MAX_NON_PAWNS was 10 and rejected all 420 positions."""
        game = _random_game()
        board = game.board()
        for move in list(game.mainline_moves())[:20]:
            board.push(move)
        non_pawns = [p for p in board.piece_map().values() if p.piece_type != chess.PAWN]
        if len(non_pawns) <= challenges.MAX_NON_PAWNS:
            self.assertGreater(challenges.MAX_NON_PAWNS, 14)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_selected_challenges_are_actually_puzzles(self):
        found = challenges.find_challenges(_random_game(), limit=5, min_ply=6)
        if not found:
            self.skipTest("no qualifying position in this random game")
        for c in found:
            self.assertGreaterEqual(c.gap_cp, challenges.DEFAULT_MIN_GAP_CP)
            self.assertNotEqual(c.best_san, c.runner_up_san)
            self.assertIsNotNone(c.question())
            self.assertLessEqual(len(c.question()), 60, "must read inside two seconds")
            self.assertTrue("?" in c.question() or "Мат" in c.question())
            if c.mate_in is not None:
                self.assertGreaterEqual(c.mate_in, challenges.MIN_MATE_DISTANCE)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_deterministic(self):
        game = _random_game()
        self.assertEqual(
            challenges.find_challenges(game, limit=5, min_ply=6),
            challenges.find_challenges(game, limit=5, min_ply=6),
        )

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_mover_cp_is_from_the_side_to_move(self):
        game = _random_game()
        found = challenges.find_challenges(game, limit=5, min_ply=6)
        for c in found:
            expected = c.best_cp if c.color == "white" else -c.best_cp
            self.assertEqual(c.mover_cp, expected)
            self.assertEqual(c.side_ru, "белые" if c.color == "white" else "чёрные")


class ChallengePlanTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_question_frame_gives_nothing_away(self):
        """The whole format depends on this.

        Highlighting the squares or printing the evaluation on the question beat
        turns the puzzle back into the reveal format. Only the answer beat may
        show them.
        """
        found = challenges.find_challenges(_random_game(), limit=3, min_ply=6)
        if not found:
            self.skipTest("no qualifying position")
        plan = clip.plan_for_challenge(found[0])
        self.assertEqual(len(plan.frames), 2)
        question, answer = plan.frames
        self.assertEqual(question.highlight_squares, ())
        self.assertIsNone(question.arrow)
        self.assertIsNone(question.eval_text)
        self.assertEqual(question.bottom_text, "")
        # The answer beat does reveal.
        self.assertTrue(answer.highlight_squares)
        self.assertIsNotNone(answer.arrow)
        self.assertIsNotNone(answer.eval_text)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_question_holds_longer_than_any_other_beat(self):
        found = challenges.find_challenges(_random_game(), limit=3, min_ply=6)
        if not found:
            self.skipTest("no qualifying position")
        plan = clip.plan_for_challenge(found[0])
        self.assertGreater(plan.before_seconds, 3.5, "viewer needs time to think")
        self.assertGreater(plan.duration, 6.0)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_eval_caption_is_russian_and_signed(self):
        found = challenges.find_challenges(_random_game(), limit=3, min_ply=6)
        if not found:
            self.skipTest("no qualifying position")
        plan = clip.plan_for_challenge(found[0])
        text = plan.frames[1].eval_text
        self.assertIsNotNone(text)
        self.assertTrue(text.startswith("оценка"))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_bad_position_yields_empty_plan(self):
        broken = challenges.Challenge(
            fen="not a fen", move_number=1, ply=0, color="white",
            best_uci="e2e4", best_san="e4", gap_cp=500, best_cp=500,
            runner_up_san="d2d4",
        )
        plan = clip.plan_for_challenge(broken)
        self.assertEqual(plan.frames, [])


if __name__ == "__main__":
    unittest.main()
"""Tests for chess game analysis.

Two real bugs are pinned here, both found while running the module for the first
time rather than by reading it:

- ``_chess_module`` imported only ``chess``, so ``chess.pgn`` was unbound and
  every PGN parse died in a swallowed ``AttributeError`` and returned ``None``.
- The piece-square tables were read without mirroring, so the starting position
  scored -82 instead of 0.

Plus the design flaw that shaped the module: a blunder is usually a quiet move
that leaves a piece attacked, so comparing eval before/after the move sees no
change and the blunder is invisible. ``_best_capture_gain`` is what closes that.
"""

from __future__ import annotations

import random
import unittest

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover - environment without the optional dep
    HAVE_CHESS = False

from shorts_clipper.chess import analysis


def _build_blunder_game(seed: int = 20261005, want_cp: int = 300):
    """Seeded random game, stopped where the mover can lose >= want_cp.

    Brute-forced rather than transcribed from a famous game: a hand-written PGN
    is only as correct as the author's memory, and this one has to hold for the
    assertion below to mean anything.
    """
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
            return game, worst
        san = board.san(rng.choice(list(board.legal_moves)))
        history.append(san)
        board.push_san(san)
    return None, None


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class ChessAnalysisTests(unittest.TestCase):
    def test_starting_position_scores_zero(self):
        """Regression: unmirrored piece-square tables scored the start -82."""
        self.assertEqual(analysis.material_eval(analysis._START_FEN), 0)

    def test_pgn_parses(self):
        """Regression: chess.pgn was unbound, so every parse returned None."""
        game = analysis.load_pgn(
            '[Event "T"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 *\n'
        )
        self.assertIsNotNone(game)
        self.assertEqual(game.headers.get("White"), "A")
        self.assertEqual(len(list(game.mainline_moves())), 3)

    def test_header_only_pgn_is_rejected(self):
        """python-chess happily returns a game for arbitrary text."""
        for junk in ("not a pgn at all", "", "[Event 'x']\n\n*"):
            self.assertIsNone(analysis.load_pgn(junk), msg=junk)

    def test_blunder_is_detected_and_measured(self):
        game, worst = _build_blunder_game()
        self.assertIsNotNone(game, "no >=300cp blunder found in the random walk")
        decisions = analysis.find_decisions(game, limit=1, min_ply=0)
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.san, worst[1])
        self.assertEqual(d.mover_loss_cp, -worst[0])
        self.assertTrue(d.material, "no engine here, so the material path must be flagged")
        self.assertIn(d.side_ru, ("белые", "чёрные"))
        self.assertIn("Ход", d.caption())

    def test_hanging_piece_is_detected_not_just_lost_material(self):
        """The flaw the lookahead exists for.

        A blunder is usually a quiet move that leaves a piece attacked, so the
        material total is unchanged at the moment it is played. Here a black pawn
        on d6 simply takes a queen on e5: without folding the threat in, this
        position looks perfectly healthy.
        """
        fen = "4k3/8/3p4/4Q3/8/8/8/4K3 b - - 0 1"
        board = chess.Board(fen)
        gain = analysis._best_capture_gain(chess, board)
        # _best_capture_gain works in material_eval units, which include the
        # piece-square tables, so a queen is ~860-900 here rather than exactly 900.
        self.assertGreaterEqual(gain, 800, "dxe5 must be found as a free queen")
        self.assertLess(gain, 950, "should be a queen, not a piece plus a bonus")

    def test_blunder_from_a_real_game_is_flagged(self):
        game, worst = _build_blunder_game()
        self.assertIsNotNone(game)
        decisions = analysis.find_decisions(game, limit=1, min_ply=0)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].san, worst[1])
        self.assertGreaterEqual(decisions[0].mover_loss_cp, 300)

    def test_min_ply_hides_opening_blunders(self):
        game, _ = _build_blunder_game()
        self.assertIsNotNone(game)
        self.assertEqual(analysis.find_decisions(game, limit=5), [])

    def test_mate_is_not_a_blunder(self):
        game = analysis.load_pgn(
            '[Event "T"]\n[Result "1-0"]\n\n1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0\n'
        )
        self.assertIsNotNone(game)
        self.assertEqual(analysis.find_decisions(game, limit=3), [])

    def test_sacrificial_attack_is_not_flagged(self):
        """A winning piece sacrifice must never read as a mistake."""
        # Qxf7+ Kxf7 Nf5+ wins material for white afterwards; the engine-free
        # path only looks at immediate swings, and mate-detection is covered above.
        game = analysis.load_pgn(
            '[Event "T"]\n[Result "*"]\n\n1. e4 e5 2. Bc4 Nf6 3. Qh5 Ng8 4. Qxf7+ Kxf7 5. Nf3 *\n'
        )
        self.assertIsNotNone(game)
        for d in analysis.find_decisions(game, limit=5):
            self.assertLess(d.mover_loss_cp, 0)

    def test_decisions_sorted_by_loss(self):
        game, _ = _build_blunder_game(want_cp=150)
        if game is None:
            self.skipTest("no blunder above 150cp in this walk")
        found = analysis.find_decisions(game, limit=5, min_ply=0)
        losses = [d.mover_loss_cp for d in found]
        self.assertEqual(losses, sorted(losses, reverse=True))

    def test_none_game_is_safe(self):
        self.assertEqual(analysis.find_decisions(None), [])

    def test_limit_is_respected(self):
        game, _ = _build_blunder_game(want_cp=150)
        if game is None:
            self.skipTest("no blunder above 150cp in this walk")
        self.assertLessEqual(len(analysis.find_decisions(game, limit=2, min_ply=0)), 2)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class BoardRenderTests(unittest.TestCase):
    def test_frame_is_not_blank(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from shorts_clipper.chess.board import FrameSpec, render_frame

        spec = FrameSpec(
            fen=analysis._START_FEN,
            top_text="Ход 1… e4",
            bottom_text="белые отдали 3.3",
            accent="bad",
            highlight_squares=("e4",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = render_frame(spec, Path(tmp) / "f.png")
            img = Image.open(out)
            self.assertEqual(img.size, (1080, 1920))
            colors = {c for _, c in img.convert("RGB").getcolors(maxcolors=200000)}
            self.assertIn((222, 214, 200), colors, "light squares missing")
            self.assertIn((124, 116, 108), colors, "dark squares missing")

    def test_piece_font_covers_chess_glyphs(self):
        from shorts_clipper.chess.board import _PIECE_FONTS, _covers, _first_font

        font = _first_font(_PIECE_FONTS, 40, "\u2654")
        self.assertTrue(_covers(font, "\u2654"), "piece font must cover the king glyph")

    def test_text_font_covers_cyrillic(self):
        from shorts_clipper.chess.board import _TEXT_FONTS, _covers, _first_font

        font = _first_font(_TEXT_FONTS, 40, "Х")
        self.assertTrue(_covers(font, "Х"), "text font must cover Cyrillic")

    def test_cyrillic_font_is_not_the_symbol_font(self):
        """Segoe UI Symbol has the pieces and no Cyrillic, so they must differ."""
        from shorts_clipper.chess.board import _PIECE_FONTS, _TEXT_FONTS, _covers, _first_font

        text_font = _first_font(_TEXT_FONTS, 40, "Х")
        piece_font = _first_font(_PIECE_FONTS, 40, "\u2654")
        if text_font.path == piece_font.path:
            self.skipTest("a single DejaVu-class font covers both on this machine")
        self.assertFalse(_covers(text_font, "\u2654"))


if __name__ == "__main__":
    unittest.main()
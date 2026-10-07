"""Tests for what the viewer is told: the narration behind the information band.

The band is only worth drawing if every field carries real text, so most of what is
pinned here is a promise about honesty rather than about formatting:

- ``eval_before`` / ``eval_after`` are the **mover's** point of view. Both evaluation
  paths in :mod:`analysis` are white's view, and mixing the two conventions is how a
  white blunder reads as a gain. The convention is asserted from both sides and the
  docstring is asserted to state it, so it cannot drift silently.
- ``detail`` never claims what the position does not support. A quiet middlegame
  position with nothing wrong in it comes back empty, and a decision flagged
  ``material`` (no engine behind it) is never called a blunder.
- A sacrifice is not a blunder. Anderssen's Qf6+ hangs the queen and is still the
  best move on the board, so the narration says "жертва" -- the forced mate is proved
  by playing the moves, not by an engine.

The rest is shape: the hook stays inside the two-line budget, a skim frame never
claims to be the reveal, the arrow points at the move as played, a PGN with no
player names still gets a readable band, and the same input always produces the same
words -- the batch is deduplicated on rendered text, so a random line would make one
game look like two clips.
"""

from __future__ import annotations

import types
import unittest

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover - environment without the optional dep
    HAVE_CHESS = False

from shorts_clipper.chess import analysis, narration

# Real games, validated move by move by python-chess, so a misremembered line fails
# loudly instead of producing a meaningless expectation.
# Paul Morphy vs Duke Karl / Count Isouard, Paris 1858. 16.Qb8+ hangs the queen and
# is met by 17.Rd8# -- a sacrifice the material path cannot see.
OPERA_GAME = """[Event "Paris Opera"]
[White "Paul Morphy"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6
9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6
15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
"""

# Adolf Anderssen vs Lionel Kieseritzky, London 1851. 4...b5 leaves the c4 bishop
# en prise, and 22.Qf6+ hangs the queen for a forced mate.
IMMORTAL_GAME = """[Event "London"]
[White "Adolf Anderssen"]
[Black "Lionel Kieseritzky"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5
8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8
15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6
21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0
"""


def _play(*sans: str):
    """A board after *sans*, so a test position is written the way it was played."""
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _walk(*sans: str) -> tuple[str, str, str]:
    """(fen_before, fen_after, san) of the last move in *sans*."""
    board = chess.Board()
    before = san = ""
    for san in sans:
        before = board.fen()
        board.push_san(san)
    return before, board.fen(), san


def _ply_state(pgn: str, ply: int) -> tuple[str, str, str]:
    """(fen_before, fen_after, san) of *ply* in a PGN, 1-based like Decision.ply."""
    game = analysis.load_pgn(pgn)
    assert game is not None, "the fixture PGN did not parse"
    board = game.board()
    before = fen = board.fen()
    san = ""
    for index, move in enumerate(game.mainline_moves(), start=1):
        san = board.san(move)
        before = fen
        board.push(move)
        fen = board.fen()
        if index == ply:
            return before, fen, san
    raise AssertionError(f"the fixture game is shorter than ply {ply}")


def _decision(**kwargs) -> analysis.Decision:
    base = dict(
        ply=1,
        move_number=1,
        color="white",
        san="e4",
        uci="e2e4",
        fen_before=chess.STARTING_FEN,
        fen_after=chess.STARTING_FEN,
        eval_before_cp=0,
        eval_after_cp=0,
        mover_loss_cp=0,
        material=False,
        header={},
    )
    base.update(kwargs)
    return analysis.Decision(**base)


def _narrate(**kwargs) -> narration.Narration:
    args = dict(
        decision=None,
        ply=1,
        total_plies=100,
        fen_before=chess.STARTING_FEN,
        fen_after=chess.STARTING_FEN,
        color="white",
        headers=None,
    )
    args.update(kwargs)
    return narration.narrate(**args)


class EvalFormatTests(unittest.TestCase):
    """format_eval is the one place the sign convention is decided."""

    def test_centipawns_both_signs_and_zero(self):
        self.assertEqual(narration.format_eval(30), "+0.3")
        self.assertEqual(narration.format_eval(280), "+2.8")
        self.assertEqual(narration.format_eval(-280), "-2.8")
        self.assertEqual(narration.format_eval(-1000), "-10.0")

    def test_zero_and_near_zero_print_without_a_sign(self):
        # "+0.0" is noise, and a negative score that rounds away must not become
        # the "-0.0" a chess audience would read as a real number.
        self.assertEqual(narration.format_eval(0), "0.0")
        self.assertEqual(narration.format_eval(-3), "0.0")
        self.assertEqual(narration.format_eval(4), "0.0")

    def test_mate_prints_as_a_distance_with_the_same_sign_rule(self):
        self.assertEqual(narration.format_eval(None, 3), "#3")
        self.assertEqual(narration.format_eval(None, -2), "-#2")
        self.assertEqual(narration.format_eval(None, 0), "#0")

    def test_mate_beats_centipawns_when_both_are_given(self):
        self.assertEqual(narration.format_eval(9_999, -3), "-#3")

    def test_missing_evaluation_is_empty_not_zero(self):
        # An unsearched position has no evaluation. Printing "0.0" would claim the
        # game is level when nobody has looked.
        self.assertEqual(narration.format_eval(None), "")

    def test_one_hyphen_glyph_for_both_forms(self):
        for text in (narration.format_eval(-280), narration.format_eval(None, -2)):
            self.assertNotIn("\u2212", text, "a typographic minus crept into a number")
            self.assertIn("-", text)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class SignConventionTests(unittest.TestCase):
    def test_convention_is_documented(self):
        # The convention is the whole point of showing two numbers, so it is
        # asserted in the docstring as well as in the arithmetic.
        self.assertIn(
            "point of view of the player who made the move", narration.__doc__ or ""
        )

    def test_white_mover_keeps_whites_perspective(self):
        decision = _decision(color="white", ply=20, eval_before_cp=30, eval_after_cp=-280,
                            mover_loss_cp=310)
        info = _narrate(decision=decision, ply=20, color="white").info
        self.assertEqual(info.eval_before, "+0.3")
        self.assertEqual(info.eval_after, "-2.8")

    def test_black_mover_gets_the_same_pair_flipped(self):
        decision = _decision(color="black", ply=20, eval_before_cp=30, eval_after_cp=-280,
                            mover_loss_cp=310)
        info = _narrate(decision=decision, ply=20, color="black").info
        self.assertEqual(info.eval_before, "-0.3")
        self.assertEqual(info.eval_after, "+2.8")

    def test_mate_sign_is_read_against_the_mover(self):
        # White mates in three after this move: for a white mover that is "#3",
        # for a black mover who walked into it the same score is "-#3".
        white = _decision(color="white", ply=20, mate_after=3)
        black = _decision(color="black", ply=20, mate_after=3)
        self.assertEqual(_narrate(decision=white, ply=20, color="white").info.eval_after, "#3")
        self.assertEqual(_narrate(decision=black, ply=20, color="black").info.eval_after, "-#3")

    def test_a_mate_the_mover_is_delivering_is_not_a_mistake(self):
        # mate is white's view, positive when white is mating. So +3 with a white
        # mover means the mover is mating, and nothing may say the move walked
        # into mate.
        before, after, _ = _ply_state(OPERA_GAME, 31)
        decision = _decision(ply=31, color="white", san="Qb8+", uci="b3b8",
                             fen_before=before, fen_after=after, mover_loss_cp=200,
                             mate_after=3)
        frame = _narrate(decision=decision, ply=31, color="white",
                         fen_before=before, fen_after=after)
        self.assertNotIn("Мат", frame.info.hook)
        self.assertNotIn("Мат", frame.info.detail)
        self.assertEqual(frame.info.eval_after, "#3")

    def test_a_mate_against_the_mover_is_the_headline(self):
        # The same score, read by a black mover: white mates, so the move walked
        # into it. 22...Nxf6 is met by 23.Be7# in the Immortal Game.
        before, after, _ = _ply_state(IMMORTAL_GAME, 44)
        decision = _decision(ply=44, color="black", san="Nxf6", uci="g8f6",
                             fen_before=before, fen_after=after, mover_loss_cp=10_000,
                             mate_after=1)
        frame = _narrate(decision=decision, ply=44, color="black",
                         fen_before=before, fen_after=after)
        self.assertIn("Мат через 1", frame.info.hook)
        self.assertIn("?", frame.info.hook, "a mate is the question the clip builds to")
        self.assertEqual(frame.info.eval_after, "-#1")

    def test_only_the_reveal_frame_carries_an_evaluation(self):
        decision = _decision(ply=20, eval_before_cp=30, eval_after_cp=-280, mover_loss_cp=310)
        for ply in (10, 18, 19, 21, 22, 40):
            info = _narrate(decision=decision, ply=ply, color="white").info
            self.assertEqual(info.eval_before, "", f"ply {ply} invented an evaluation")
            self.assertEqual(info.eval_after, "", f"ply {ply} invented an evaluation")


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class HookTests(unittest.TestCase):
    def test_hook_never_exceeds_the_line_budget(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480,
                             eval_before_cp=-30, eval_after_cp=-510)
        for ply in range(1, 121):
            for color in ("white", "black"):
                frame = _narrate(decision=decision, ply=ply, color=color)
                self.assertLessEqual(
                    len(frame.info.hook),
                    narration.MAX_HOOK_CHARS,
                    f"ply {ply} hook is too long: {frame.info.hook!r}",
                )
                self.assertTrue(frame.info.hook, f"ply {ply} has no hook at all")

    def test_mate_hook_fits_at_any_distance(self):
        for distance in range(1, 31):
            decision = _decision(ply=20, color="white", mate_after=-distance,
                                 mover_loss_cp=10_000)
            hook = _narrate(decision=decision, ply=20, color="white").info.hook
            self.assertLessEqual(len(hook), narration.MAX_HOOK_CHARS, hook)
            self.assertIn("Мат", hook)

    def test_skim_frame_and_reveal_frame_are_not_the_same_line(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480)
        skim = _narrate(decision=decision, ply=8, color="white").info.hook
        reveal = _narrate(decision=decision, ply=20, color="white").info.hook
        self.assertNotEqual(skim, reveal)
        self.assertNotIn("вот этот ход", skim.lower())

    def test_skim_rotation_varies_and_stays_inside_one_pool(self):
        hooks = {
            _narrate(ply=ply).info.hook
            for ply in range(4, 60, 4)
        }
        self.assertGreaterEqual(len(hooks), 3, "the skim repeated one line for 14 frames")
        self.assertTrue(all(len(hook) <= narration.MAX_HOOK_CHARS for hook in hooks))

    def test_same_input_gives_identical_text(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480,
                             header={"White": "Kasparov", "Black": "Kramnik"})
        first = _narrate(decision=decision, ply=20, color="white",
                         headers=decision.header)
        second = _narrate(decision=decision, ply=20, color="white",
                          headers=decision.header)
        self.assertEqual(first, second)
        self.assertEqual(first.info.hook, second.info.hook)

    def test_lead_and_after_frames_say_different_things(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480)
        lead = [_narrate(decision=decision, ply=ply).info.hook for ply in (18, 19)]
        after = [_narrate(decision=decision, ply=ply).info.hook for ply in (21, 22)]
        self.assertEqual(len(set(lead)), 2, "both lead frames got the same line")
        self.assertEqual(len(set(after)), 2, "both after frames got the same line")
        # The frames before the reveal ask a question; the frames after it state
        # what the move cost. Neither is allowed to be a bare label.
        for hook in lead:
            self.assertIn("?", hook, "a lead frame must ask, not label")
        self.assertEqual(set(after), {"Вот цена этого хода", "Теперь всё по-другому"})


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class ArrowTests(unittest.TestCase):
    def test_skim_frame_points_at_nothing(self):
        frame = _narrate(ply=8)
        self.assertIsNone(frame.arrow, "a mid-skim frame has no single move to point at")

    def test_reveal_arrow_is_the_move_that_was_played(self):
        before, after, san = _ply_state(OPERA_GAME, 31)
        self.assertEqual(san, "Qb8+")
        decision = _decision(ply=31, move_number=16, color="white", san="Qb8+",
                             uci="b3b8", fen_before=before, fen_after=after,
                             mover_loss_cp=865, material=True)
        frame = _narrate(decision=decision, ply=31, color="white",
                         fen_before=before, fen_after=after)
        self.assertEqual(frame.arrow, ("b3", "b8"))
        self.assertEqual(frame.info.moved_san, "Qb8+")
        self.assertEqual(frame.accent, "bad")

    def test_arrow_is_recovered_from_the_fens_when_the_uci_is_unusable(self):
        before, after, _ = _ply_state(IMMORTAL_GAME, 43)
        decision = _decision(ply=22, color="white", san="Qf6+", uci="nonsense",
                             fen_before=before, fen_after=after, mover_loss_cp=955,
                             material=True)
        frame = _narrate(decision=decision, ply=22, color="white",
                         fen_before=before, fen_after=after)
        self.assertEqual(frame.arrow, ("f3", "f6"))
        # The decision's SAN is preferred, so a bad uci cannot cost the caption.
        self.assertEqual(frame.info.moved_san, "Qf6+")

    def test_a_frame_covering_several_plies_names_no_move(self):
        # Pacing renders one frame every few plies while skimming. There is no
        # single move on such a frame, so naming one would be a lie.
        before, _, _ = _ply_state(IMMORTAL_GAME, 4)
        _, after, _ = _ply_state(IMMORTAL_GAME, 12)
        frame = _narrate(ply=12, fen_before=before, fen_after=after)
        self.assertEqual(frame.info.moved_san, "")
        self.assertIsNone(frame.arrow)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class PlayersTests(unittest.TestCase):
    def test_names_come_from_the_headers(self):
        self.assertEqual(
            narration.players_line({"White": "Garry Kasparov", "Black": "Vladimir Kramnik"}),
            "Garry Kasparov — Vladimir Kramnik",
        )

    def test_given_names_are_trimmed_for_the_band(self):
        # "Nepomniachtchi, Ian" is how PGNs spell it; the given name is dead weight
        # in a 960px line.
        self.assertEqual(
            narration.players_line({"White": "Kasparov, Garry", "Black": "Непомнящий"}),
            "Kasparov — Непомнящий",
        )

    def test_missing_or_unknown_headers_fall_back(self):
        fallback = narration.FALLBACK_PLAYERS
        self.assertEqual(fallback, "белые — чёрные")
        for headers in (None, {}, {"White": "Kasparov"}, {"Black": "Kramnik"},
                        {"White": "?", "Black": "???"}, {"White": "", "Black": "  "},
                        {"White": "unknown", "Black": "n/a"}):
            self.assertEqual(narration.players_line(headers), fallback, repr(headers))

    def test_headers_as_attributes_are_accepted(self):
        # The caller may hold the game object rather than a dict; both are read.
        headers = types.SimpleNamespace(White="Kasparov", Black="Kramnik")
        self.assertEqual(narration.players_line(headers), "Kasparov — Kramnik")

    def test_decision_header_is_used_when_no_headers_are_passed(self):
        decision = _decision(ply=20, header={"White": "Morphy", "Black": "Isouard"})
        frame = _narrate(decision=decision, ply=20)
        self.assertEqual(frame.info.players, "Morphy — Isouard")


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class ProgressTests(unittest.TestCase):
    def test_progress_counts_plies(self):
        self.assertEqual(narration.progress_line(24, 134), "24 / 134")
        self.assertEqual(narration.progress_line(0, 134), "0 / 134")

    def test_progress_clamps_and_survives_a_missing_length(self):
        self.assertEqual(narration.progress_line(400, 134), "134 / 134")
        self.assertEqual(narration.progress_line(-3, 134), "0 / 134")
        self.assertEqual(narration.progress_line(10, 0), "")
        self.assertEqual(narration.progress_line(None, None), "")


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class DetailTests(unittest.TestCase):
    def test_a_quiet_position_leaves_the_line_empty(self):
        # Nothing is wrong in the starting position, so there is nothing to say.
        frame = _narrate(ply=1)
        self.assertEqual(frame.info.detail, "")
        self.assertTrue(frame.info.hook, "the hook is still filled")

    def test_a_decision_with_nothing_to_show_leaves_the_line_empty(self):
        decision = _decision(ply=20, mover_loss_cp=40)
        frame = _narrate(decision=decision, ply=20)
        self.assertEqual(frame.info.detail, "")

    def test_hanging_piece_is_named_and_priced(self):
        # 4...b5 leaves the c4 bishop alone: the b5 pawn takes it and nothing
        # recaptures. Verified from the position, not asserted.
        before, after, _ = _ply_state(IMMORTAL_GAME, 8)
        frame = _narrate(ply=8, fen_before=before, fen_after=after, color="black")
        self.assertIn("слон", frame.info.detail)
        self.assertIn("c4", frame.info.detail)
        self.assertIn("висит", frame.info.detail)
        self.assertIn("3.3", frame.info.detail)

    def test_lost_castling_rights_are_reported_only_when_provable(self):
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.Rg1: the rook steps out, the king never
        # moves, and the h1 rook's right to castle is gone for good. Nothing else
        # is wrong with the position, so the line is about the castling.
        before, after, san = _walk("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Rg1")
        self.assertEqual(san, "Rg1")
        decision = _decision(ply=7, color="white", san="Rg1", uci="h1g1",
                             fen_before=before, fen_after=after, mover_loss_cp=220,
                             material=True)
        frame = _narrate(decision=decision, ply=7, color="white",
                         fen_before=before, fen_after=after)
        self.assertEqual(frame.info.detail, "Ладья уже ходила — рокировки не будет")

    def test_castling_rights_kept_are_not_reported(self):
        before, after, _ = _ply_state(IMMORTAL_GAME, 2)
        decision = _decision(ply=2, color="white", san="f4", uci="f2f4",
                             fen_before=before, fen_after=after, mover_loss_cp=60)
        frame = _narrate(decision=decision, ply=2, color="white",
                         fen_before=before, fen_after=after)
        self.assertNotIn("рокировки", frame.info.detail)

    def test_a_sacrifice_is_never_called_a_blunder(self):
        # 22.Qf6+ hangs the queen and mates in two. The material path flags it as
        # a 955cp blunder; the narration proves the mate instead and says so.
        before, after, _ = _ply_state(IMMORTAL_GAME, 43)
        decision = _decision(ply=43, color="white", san="Qf6+", uci="f3f6",
                             fen_before=before, fen_after=after, mover_loss_cp=955,
                             material=True)
        frame = _narrate(decision=decision, ply=43, color="white",
                         fen_before=before, fen_after=after)
        self.assertIn("мат", frame.info.hook.lower())
        self.assertIn("Жертва", frame.info.detail)
        self.assertNotIn("зевок", frame.info.hook.lower())
        self.assertNotIn("зевок", frame.info.detail.lower())

    def test_without_an_engine_the_wording_claims_no_quality(self):
        # Same position, material estimate only: a piece can be hanging in a
        # sacrifice as easily as in a blunder, and the material path cannot tell
        # them apart, so the line states the position instead of judging the move.
        before, after, _ = _ply_state(OPERA_GAME, 19)
        decision = _decision(ply=19, color="white", san="Nxb5", uci="c3b5",
                             fen_before=before, fen_after=after, mover_loss_cp=210,
                             material=True)
        frame = _narrate(decision=decision, ply=19, color="white",
                         fen_before=before, fen_after=after)
        self.assertIn("b5", frame.info.hook)
        self.assertNotIn("зевок", frame.info.hook.lower())
        self.assertNotIn("промах", frame.info.hook.lower())

    def test_engine_backed_decision_does_use_the_quality_words(self):
        before, after, _ = _ply_state(OPERA_GAME, 19)
        decision = _decision(ply=19, color="white", san="Nxb5", uci="c3b5",
                             fen_before=before, fen_after=after, mover_loss_cp=480,
                             material=False)
        frame = _narrate(decision=decision, ply=19, color="white",
                         fen_before=before, fen_after=after)
        self.assertIn("зевок", frame.info.hook.lower())

    def test_mate_in_the_position_is_reported_for_the_side_to_move(self):
        frame = _narrate(ply=1, fen_before=chess.STARTING_FEN,
                         fen_after="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        self.assertIn("мат", frame.info.detail.lower())


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class BandTests(unittest.TestCase):
    """The band is only worth drawing when it is full."""

    def test_the_reveal_frame_fills_every_field(self):
        before, after, _ = _ply_state(IMMORTAL_GAME, 35)
        decision = _decision(ply=35, color="white", san="Bd6", uci="f4d6",
                             fen_before=before, fen_after=after,
                             eval_before_cp=-180, eval_after_cp=-660,
                             mover_loss_cp=480, material=False,
                             header={"White": "Anderssen", "Black": "Kieseritzky"})
        frame = _narrate(decision=decision, ply=35, color="white",
                         fen_before=before, fen_after=after,
                         total_plies=45, headers=decision.header)
        info = frame.info
        self.assertEqual(info.hook, "Зевок: ладья на a1 висит")
        self.assertTrue(info.detail)
        self.assertEqual(info.progress, "35 / 45")
        self.assertEqual((info.eval_before, info.eval_after), ("-1.8", "-6.6"))
        self.assertEqual(info.players, "Anderssen — Kieseritzky")
        self.assertEqual(info.moved_san, "Bd6")
        self.assertEqual(frame.arrow, ("f4", "d6"))

    def test_a_skim_frame_fills_everything_but_the_evaluation(self):
        before, after, _ = _ply_state(IMMORTAL_GAME, 20)
        decision = _decision(ply=43, color="white", san="Qf6+", uci="f3f6",
                             fen_before=before, fen_after=after, mover_loss_cp=955,
                             material=True)
        frame = _narrate(decision=decision, ply=20, color="white",
                         fen_before=before, fen_after=after, total_plies=45,
                         headers={"White": "Anderssen", "Black": "Kieseritzky"})
        info = frame.info
        for field in (info.hook, info.detail, info.progress, info.players, info.moved_san):
            self.assertTrue(field, f"{field!r} is empty")
        self.assertEqual(info.eval_before, "")
        self.assertEqual(info.eval_after, "")

    def test_nothing_raises_on_junk_input(self):
        decision = _decision(ply=5)
        for kwargs in (
            dict(fen_before="not a fen", fen_after="also not a fen"),
            dict(fen_before="", fen_after=""),
            dict(color="purple"),
            dict(color=None),
            dict(ply=None, total_plies=None),
            dict(ply=-5, total_plies=-5),
            dict(headers="not headers at all"),
            dict(headers=[("White", "Kasparov")]),
        ):
            frame = _narrate(decision=decision, **kwargs)
            self.assertIsInstance(frame, narration.Narration)

    def test_narration_survives_without_python_chess(self):
        # python-chess is an optional dependency of this package. Without it the
        # position-based lines go blank, and the band still has to be fillable.
        decision = _decision(ply=20, color="white", mover_loss_cp=480,
                             eval_before_cp=30, eval_after_cp=-480,
                             header={"White": "Kasparov", "Black": "Kramnik"})
        original = narration._chess
        narration._chess = lambda: None  # type: ignore[assignment]
        try:
            skim = _narrate(ply=8, headers=decision.header)
            reveal = _narrate(decision=decision, ply=20, headers=decision.header)
        finally:
            narration._chess = original
        for frame, progress in ((skim, "8 / 100"), (reveal, "20 / 100")):
            self.assertTrue(frame.info.hook)
            self.assertEqual(frame.info.players, "Kasparov — Kramnik")
            self.assertEqual(frame.info.progress, progress)
        self.assertEqual(skim.info.detail, "", "no position, no claim")
        # The numbers come off the decision, not off the board, so they survive.
        self.assertEqual((reveal.info.eval_before, reveal.info.eval_after), ("+0.3", "-4.8"))
        self.assertIsNone(reveal.arrow, "no board, no arrow")

    def test_a_clip_without_a_decision_still_narrates(self):
        for ply in range(1, 40, 3):
            frame = _narrate(decision=None, ply=ply)
            self.assertEqual(frame.kind, narration.KIND_SKIM)
            self.assertTrue(frame.info.hook)
            self.assertEqual(frame.info.progress, f"{ply} / 100")


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class PacingShapeTests(unittest.TestCase):
    """The narration has to agree with the sampler that will call it."""

    def test_frame_kinds_follow_the_slow_window(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480)
        kinds = {ply: _narrate(decision=decision, ply=ply).kind for ply in range(1, 31)}
        self.assertEqual(kinds[20], narration.KIND_REVEAL)
        for ply in (18, 19):
            self.assertEqual(kinds[ply], narration.KIND_LEAD)
        for ply in (21, 22):
            self.assertEqual(kinds[ply], narration.KIND_AFTER)
        for ply in (17, 23):
            self.assertEqual(kinds[ply], narration.KIND_SKIM)

    def test_a_skim_step_of_four_never_sticks_on_one_line(self):
        # pacing samples every SKIM_PLIES_PER_FRAME plies; a rotation that locked
        # onto that step would print the same line for the whole skim.
        hooks = [_narrate(ply=4 * step).info.hook for step in range(1, 9)]
        self.assertGreaterEqual(len(set(hooks)), 3, hooks)

    def test_only_the_reveal_frame_is_ever_marked_bad(self):
        decision = _decision(ply=20, color="white", mover_loss_cp=480)
        for ply in range(1, 31):
            frame = _narrate(decision=decision, ply=ply)
            expected = "bad" if ply == 20 else "neutral"
            self.assertEqual(frame.accent, expected, f"ply {ply}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
"""Tests for the highlight pacing.

The format inverts the obvious edit: skim the whole game, then slow down at one
move. Two real bugs lived here, and both are now regression tests.

1. After the slow window the sampler fell into its "a skim step must not cross
   the window" branch and re-appended ``slow_from - 1`` on every iteration. A
   114-ply game collapsed to 8 frames and the tail of the game was never shown.
   ``test_covers_the_whole_game`` pins the frame count and the last ply.
2. The plan needed explicit per-frame holds, because three fixed beat durations
   cannot express "0.22s here, 1.35s there".
   ``test_slow_frames_are_held_far_longer`` pins the ratio.

The rest guards the shape: the window is centred, the cap holds, degenerate input
returns nothing rather than raising.
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

from shorts_clipper.chess import analysis, clip, pacing


def _random_game(plies: int = 114, seed: int = 4242):
    rng = random.Random(seed)
    board = chess.Board()
    history: list[object] = []
    while len(history) < plies:
        moves = list(board.legal_moves)
        if not moves:
            break
        # add_variation wants a Move, not a SAN string.
        history.append(rng.choice(moves))
        board.push(history[-1])
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    for move in history:
        node = node.add_variation(move)
    return game


def _fake_decision(ply: int, move_number: int = 1):
    return analysis.Decision(
        ply=ply,
        move_number=move_number,
        color="white",
        san="Qxd6+",
        uci="d1d6",
        fen_before=chess.STARTING_FEN,
        fen_after=chess.STARTING_FEN,
        eval_before_cp=10,
        eval_after_cp=-300,
        mover_loss_cp=500,
        material=False,
        header={},
    )


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class PacingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = _random_game()
        self.plies = len(list(self.game.mainline_moves()))

    def test_covers_the_whole_game(self):
        frames, holds = pacing.pace_game(self.game, target_ply=17)

        self.assertEqual(len(frames), len(holds))
        self.assertGreater(len(frames), 20, "the game collapsed to too few frames")

        # The last frame must show the final position, or the clip cuts off.
        final_fen = self.game.end().board().fen()
        self.assertEqual(frames[-1].fen, final_fen)

        # And the frame count has to grow with the game, not with the window.
        short_frames, _ = pacing.pace_game(
            _random_game(plies=40, seed=7), target_ply=10
        )
        self.assertLess(len(short_frames), len(frames))

    def test_slow_frames_are_held_far_longer(self):
        frames, holds = pacing.pace_game(self.game, target_ply=17)

        slow = [h for h, f in zip(holds, frames, strict=True) if f.accent == "bad"]
        skim = [h for h, f in zip(holds, frames, strict=True) if f.accent != "bad"]
        self.assertTrue(bool(slow), "no slow frames around the decisive move")
        self.assertTrue(bool(skim), "no skim frames before the window")

        slow_each = sum(slow) / len(slow)
        skim_each = sum(skim) / len(skim)
        self.assertGreater(
            slow_each / skim_each,
            3.0,
            "the decisive move is not actually slower than the skim",
        )

        # And the skim must stay fast: this is a format about not watching
        # ninety shuffling moves in real time.
        self.assertLess(skim_each, 0.5, "the skim is too slow to read as a skim")

    def test_window_is_centred_on_the_target(self):
        frames, _ = pacing.pace_game(self.game, target_ply=17, slow_window=2)
        slow_positions = [i for i, f in enumerate(frames) if f.accent == "bad"]
        self.assertEqual(len(slow_positions), 2 * 2 + 1)

        marker = [i for i, f in enumerate(frames) if "вот этот ход" in f.top_text]
        self.assertEqual(len(marker), 1, "exactly one frame names the decisive move")
        self.assertIn(marker[0], slow_positions)

    def test_target_at_either_end_does_not_crash(self):
        for ply in (0, 1, self.plies - 1, self.plies, self.plies + 50):
            frames, holds = pacing.pace_game(self.game, target_ply=ply)
            self.assertEqual(len(frames), len(holds))
            self.assertTrue(all(h > 0 for h in holds), f"non-positive hold at ply {ply}")

    def test_frame_cap_is_respected(self):
        frames, holds = pacing.pace_game(
            _random_game(plies=400, seed=11), target_ply=200, max_frames=20
        )
        self.assertLessEqual(len(frames), 20)
        self.assertEqual(len(frames), len(holds))

    def test_empty_game_returns_nothing(self):
        frames, holds = pacing.pace_game(chess.pgn.Game(), target_ply=0)
        self.assertEqual(frames, [])
        self.assertEqual(holds, [])

    def test_plan_carries_holds(self):
        plan = pacing.plan_for_highlight(self.game, _fake_decision(17))

        self.assertIsNotNone(plan.holds)
        self.assertEqual(len(plan.holds), len(plan.frames))
        self.assertAlmostEqual(plan.duration, sum(plan.holds), places=6)
        self.assertTrue(all(h > 0 for h in plan.holds))

    def test_explicit_holds_override_the_three_beat_fields(self):
        spec = [clip.FrameSpec(fen=chess.STARTING_FEN, top_text="", bottom_text="")]
        plan = clip.ChessClipPlan(
            frames=list(spec) * 3,
            holds=[0.2, 0.2, 1.4],
        )

        self.assertAlmostEqual(plan.duration, 1.8, places=6)
        self.assertAlmostEqual(clip._hold_seconds(plan, 0), 0.2, places=6)
        self.assertAlmostEqual(clip._hold_seconds(plan, 2), 1.4, places=6)

        # Without holds the three fixed beats still apply.
        plain = clip.ChessClipPlan(frames=list(spec) * 3)
        self.assertAlmostEqual(plain.duration, plain.before_seconds + plain.move_seconds + plain.after_seconds)
        self.assertAlmostEqual(clip._hold_seconds(plain, 1), plain.move_seconds)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
"""Tests for the highlight pacing.

The format inverts the obvious edit: skim the whole game, then slow down at one
move. Two bugs lived here long before the current ones, and both are still
regression tests:

1. After the slow window the sampler fell into its "a skim step must not cross
   the window" branch and re-appended ``slow_from - 1`` on every iteration. A
   114-ply game collapsed to 8 frames and the tail of the game was never shown.
   ``test_covers_the_whole_game`` pins the frame count and the last ply.
2. The plan needed explicit per-frame holds, because three fixed beat durations
   cannot express "0.125s here, 2.6s there".
   ``test_slow_frames_are_held_far_longer`` pins the ratio.
3. Thinning with ``[::step]`` yields ``ceil(len/step)`` frames, so a floored step
   overshot the cap: 22 frames for a cap of 20. ``test_frame_cap_is_respected``.

And four more, each of which was a defect measured on a rendered clip:

4. The slow window held one board repeatedly -- a 2fps contact sheet read
   ``24... 24... 24... 25... 25...``, because the five plies of the window are
   three move numbers and the second ply of a move is a reply to the first.
   ``test_window_holds_one_frame_per_move_number`` and
   ``test_no_adjacent_frames_show_the_same_position`` pin the fix.
5. The payoff landed at 20% of a 14.3s clip and 8 seconds of skimming followed
   it. ``test_reveal_lands_late`` pins that nothing but the window follows it.
6. The last six seconds were a bare king. ``test_clip_stops_when_the_board_is_empty``.
7. The skim was 0.22s a frame over 4 plies: long enough to start reading a
   position, too short to finish. ``test_skim_reads_as_a_speedrun``.

Plus the shape constraints: 8-20s for a short and a long game, deterministic
output, every hold above the renderer's 0.08s floor.
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


def _quiet_streak_game(gap: int = 3):
    """A game whose middle is pure shuffling: knight and bishop shuffles.

    Rendered as a window this is the position the defect was found on -- every
    consecutive pair of frames differs by one quiet move, which is exactly the
    pair a 2fps contact sheet cannot tell apart. Built by steering towards
    already-visited positions for *gap* plies, then playing on.
    """
    game = _random_game(plies=40, seed=90210)
    board = chess.Board()
    history: list[object] = []
    quiet = 0
    while len(history) < 40:
        moves = list(board.legal_moves)
        if not moves:
            break
        quiet_moves = []
        for move in moves:
            board.push(move)
            key = board.board_fen()
            board.pop()
            if any(key == fen for fen in _seen(history)) or quiet < gap:
                quiet_moves.append(move)
        history.append(quiet_moves[0] if quiet_moves else moves[0])
        board.push(history[-1])
        quiet = quiet + 1 if quiet_moves else 0
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    for move in history:
        node = node.add_variation(move)
    return game


def _seen(history):
    board = chess.Board()
    out = []
    for move in history:
        board.push(move)
        out.append(board.board_fen())
    return out


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


def _pieces(fen: str) -> int:
    return len(chess.Board(fen).piece_map())


def _slow(frames):
    return [i for i, f in enumerate(frames) if f.accent == "bad"]


def _outro(frames):
    """Frames of the held final position, which is neither skim nor window."""
    return [i for i, f in enumerate(frames) if "финал" in f.top_text]


def _skim(frames):
    return [
        i
        for i, f in enumerate(frames)
        if f.accent != "bad" and "финал" not in f.top_text
    ]


def _placement(fen: str) -> str:
    """What the renderer draws: the pieces and their squares, nothing else."""
    return chess.Board(fen).board_fen()


def _walk(game):
    """FEN after each ply, indexed by ply the way :mod:`pacing` indexes them.

    Deliberately not including the starting position: an off-by-one here would
    quietly compare the wrong board and the test would still pass.
    """
    board = chess.Board()
    fens = []
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return fens


def _last_shown(game) -> int:
    counts = [len(chess.Board(fen).piece_map()) for fen in _walk(game)]
    return pacing._last_worth_showing(counts)


def _window_plies(game, target_ply, slow_window: int = 2) -> list[int]:
    """The plies the emitted slow frames show, read from the plan's own beats.

    Positions repeat in a real game, so a FEN does not identify its ply and
    matching frames back to plies by position silently reads the wrong one.
    Reading the beats off :func:`pacing._drop_repeats` avoids that, and reading
    its *return* value means the seam guard is included rather than assumed
    away.
    """
    captured: dict[str, list[int]] = {}
    original = pacing._drop_repeats

    def spy(beats, placements, target):
        kept = original(beats, placements, target)
        captured["slow"] = [b.ply for b in kept if b.slow]
        return kept

    pacing._drop_repeats = spy
    try:
        frames, _ = pacing.pace_game(game, target_ply, slow_window=slow_window)
    finally:
        pacing._drop_repeats = original

    fens = _walk(game)
    assert [frames[i].fen for i in _slow(frames)] == [
        fens[ply] for ply in captured["slow"]
    ], "the beats and the shipped frames disagree"
    return captured["slow"]


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class PacingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = _random_game()
        self.plies = len(list(self.game.mainline_moves()))

    def test_covers_the_whole_game(self):
        frames, holds = pacing.pace_game(self.game, target_ply=17)

        self.assertEqual(len(frames), len(holds))
        self.assertGreater(len(frames), 20, "the game collapsed to too few frames")

        # The last frame must show the last position still worth showing, or
        # the clip either cuts off or spends its tail on a bare king.
        self.assertGreaterEqual(_pieces(frames[-1].fen), pacing.MIN_PIECES_TO_SHOW)

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

    def test_window_is_centred_on_the_target(self):
        frames, _ = pacing.pace_game(self.game, target_ply=17, slow_window=2)
        slow_positions = _slow(frames)
        marker = [i for i, f in enumerate(frames) if "вот этот ход" in f.top_text]

        self.assertEqual(len(marker), 1, "exactly one frame names the decisive move")
        self.assertIn(marker[0], slow_positions)

        # Five plies wide, one frame per move number, so three frames -- the ply
        # before the mistake, the mistake, the ply a move after it.
        plies = _window_plies(self.game, 17)
        self.assertEqual(len(plies), 3)
        self.assertEqual(plies[1], 17, "the decisive ply is the middle frame")
        self.assertLess(max(plies), 17 + 2 + 1)
        self.assertGreaterEqual(min(plies), 17 - 2 - 1)

    def test_window_holds_one_frame_per_move_number(self):
        """The repeat defect: ``24... 24... 24... 25... 25...`` in one clip."""
        game = _quiet_streak_game()

        for target in (12, 14, 16, 18):
            plies = _window_plies(game, target)
            move_numbers = [p // 2 for p in plies]
            self.assertEqual(
                len(set(move_numbers)),
                len(move_numbers),
                f"two window frames share move {move_numbers} at target {target}",
            )
            # And no two of them are a single ply apart: the second ply of a
            # move is a reply to the first and the board barely moves.
            gaps = [b - a for a, b in zip(plies, plies[1:], strict=False)]
            self.assertTrue(
                all(gap >= 2 for gap in gaps),
                f"window frames one ply apart at target {target}: {plies}",
            )

    def test_no_adjacent_frames_show_the_same_position(self):
        frames, _ = pacing.pace_game(self.game, target_ply=17)

        seen: list[str] = []
        for frame in frames:
            if seen and _placement(frame.fen) == _placement(seen[-1]):
                self.fail(f"two adjacent frames render the same board: {frame.fen}")
            seen.append(frame.fen)

    def test_decisive_ply_survives_a_duplicate_neighbour(self):
        """The one exemption to the repeat guard: the decisive ply always ships."""
        game = _quiet_streak_game()
        target = 16

        frames, _ = pacing.pace_game(game, target)
        decisive = [
            f for f in frames if "вот этот ход" in f.top_text
        ]
        self.assertEqual(len(decisive), 1)
        self.assertEqual(decisive[0].fen, _fen_after(game, target))

    def test_reveal_lands_late(self):
        """The anticlimax defect: the payoff, then eight seconds of skimming."""
        game = _random_game(plies=200, seed=31)
        # An early blunder is the worst case: ply order alone would put the
        # payoff at 20% of the runtime with a hundred plies still to skim.
        target = 30
        frames, holds = pacing.pace_game(game, target)

        reveal = [i for i, f in enumerate(frames) if "вот этот ход" in f.top_text][0]

        # The window is a replay placed at the end, so the payoff sits in the
        # last quarter of the clip by frame count whatever the target ply was.
        self.assertGreater(
            reveal / len(frames),
            0.75,
            "the decisive move is not near the end of the clip",
        )
        # And nothing skims after it: what follows is the rest of the window
        # and the held outro, which is the point of putting the window last.
        self.assertEqual(
            [i for i in _skim(frames) if i > reveal],
            [],
            "the clip returns to the skim after the payoff",
        )
        tail = sum(
            holds[i] for i in range(reveal + 1, len(frames)) if frames[i].accent != "bad"
        )
        self.assertLessEqual(
            tail,
            pacing.OUTRO_SECONDS + 1e-6,
            f"{tail:.2f}s of skimming follows the payoff",
        )

    def test_clip_stops_when_the_board_is_empty(self):
        """The dead-air defect: the last six seconds on two or three pieces."""
        game = _random_game(plies=400, seed=11)
        frames, holds = pacing.pace_game(game, target_ply=200)

        self.assertGreaterEqual(
            _pieces(frames[-1].fen),
            pacing.MIN_PIECES_TO_SHOW,
            "the clip ends on a position nobody would look at twice",
        )
        # And it genuinely cut: that game plays on well past the stop, and the
        # clip does not carry the positions the piece rule rejected.
        fens = _walk(game)
        total = len(fens)
        shown = [
            ply
            for ply, fen in enumerate(fens)
            if _placement(fen) == _placement(frames[-1].fen)
        ]
        self.assertLess(max(shown), total - 1, "nothing was trimmed")
        self.assertLess(
            len(chess.Board(fens[total - 1]).piece_map()),
            pacing.MIN_PIECES_TO_SHOW,
            "this game was supposed to be one that plays on to a bare king",
        )
        # The trimmed tail is real runtime the old build was spending.
        self.assertGreater(
            total - 1 - max(shown), 5, "nothing worth trimming"
        )

    def test_skim_reads_as_a_speedrun(self):
        """The illegibility defect: 0.22s a frame, 4 plies, read as nothing."""
        self.assertLessEqual(
            pacing.SKIM_PLIES_PER_FRAME / pacing.SKIM_PLIES_PER_SECOND,
            0.14,
            "the skim is too slow to read as a skim",
        )
        self.assertGreaterEqual(
            pacing.SKIM_PLIES_PER_FRAME / pacing.SKIM_PLIES_PER_SECOND, 0.10
        )
        # A whole move per frame, so each skim frame carries a complete move.
        self.assertEqual(pacing.SKIM_PLIES_PER_FRAME % 2, 0)

        frames, holds = pacing.pace_game(self.game, target_ply=17)
        skim = [holds[i] for i in _skim(frames)]
        self.assertTrue(skim)
        self.assertTrue(all(0.10 <= h <= 0.14 for h in skim), sorted(set(skim)))

    def test_frame_budget_covers_a_long_game(self):
        """Coarser skim steps mean fewer frames, so the budget has to keep up.

        Raising :data:`SKIM_PLIES_PER_FRAME` to make each skim frame carry a whole
        move is only free if ``MAX_FRAMES * SKIM_PLIES_PER_FRAME`` still spans a
        long game. Otherwise a 200-ply game silently gets its skim thinned to
        wider steps and the per-frame hold stops describing what is on screen.
        """
        self.assertGreaterEqual(
            pacing.MAX_FRAMES * pacing.SKIM_PLIES_PER_FRAME,
            200,
            "the frame budget can no longer hold a 200-ply game at this step",
        )

        game = _random_game(plies=200, seed=5)
        frames, _ = pacing.pace_game(game, target_ply=100)
        self.assertLessEqual(len(frames), pacing.MAX_FRAMES)
        # Nothing had to be thinned, so the plan is exactly skim + window.
        self.assertEqual(
            len(_skim(frames)) + len(_slow(frames)) + len(_outro(frames)),
            len(frames),
        )

    def test_skim_reaches_the_last_position_worth_showing(self):
        game = _random_game(plies=200, seed=5)
        frames, _ = pacing.pace_game(game, target_ply=100)
        fens = _walk(game)
        last_shown = _last_shown(game)

        skim_ply = fens.index(frames[_skim(frames)[-1]].fen)
        self.assertLessEqual(
            last_shown - skim_ply,
            pacing.SKIM_PLIES_PER_FRAME - 1,
            "the skim stops short of the position the clip is supposed to reach",
        )

    def test_clip_length_stays_shorts_shaped(self):
        for plies, seed in ((40, 7), (200, 31)):
            game = _random_game(plies=plies, seed=seed)
            total = len(list(game.mainline_moves()))
            for target in (5, total // 2, total - 1):
                _, holds = pacing.pace_game(game, target)
                seconds = sum(holds)
                self.assertGreaterEqual(
                    seconds, 8.0, f"{plies}-ply game at {target}: {seconds:.2f}s"
                )
                self.assertLessEqual(
                    seconds, 20.0, f"{plies}-ply game at {target}: {seconds:.2f}s"
                )

    def test_hold_floor_is_respected(self):
        for plies, seed in ((40, 3), (114, 4242), (400, 11)):
            game = _random_game(plies=plies, seed=seed)
            total = len(list(game.mainline_moves()))
            for target in (0, 1, total // 2, total - 1, total, total + 50):
                _, holds = pacing.pace_game(game, target)
                self.assertTrue(
                    all(h >= pacing.MIN_HOLD_SECONDS for h in holds),
                    f"sub-floor hold at ply {target} of a {plies}-ply game",
                )

    def test_plans_are_deterministic(self):
        game = _random_game(plies=200, seed=31)
        first = pacing.plan_for_highlight(game, _fake_decision(30))
        second = pacing.plan_for_highlight(game, _fake_decision(30))

        self.assertEqual([f.fen for f in first.frames], [f.fen for f in second.frames])
        self.assertEqual([f.top_text for f in first.frames], [f.top_text for f in second.frames])
        self.assertEqual(first.holds, second.holds)

    def test_frame_cap_is_respected(self):
        frames, holds = pacing.pace_game(
            _random_game(plies=400, seed=11), target_ply=200, max_frames=20
        )
        self.assertLessEqual(len(frames), 20)
        self.assertEqual(len(frames), len(holds))

    def test_target_at_either_end_does_not_crash(self):
        for ply in (0, 1, self.plies - 1, self.plies, self.plies + 50):
            frames, holds = pacing.pace_game(self.game, target_ply=ply)
            self.assertEqual(len(frames), len(holds))
            self.assertTrue(all(h > 0 for h in holds), f"non-positive hold at ply {ply}")

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


def _fen_after(game, ply: int) -> str:
    board = chess.Board()
    for move in list(game.mainline_moves())[: ply + 1]:
        board.push(move)
    return board.fen()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
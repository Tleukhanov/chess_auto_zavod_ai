"""Tests for the highlight pacing.

The format is a lead-in and a window: a short run-up to the decisive move, then
slow motion through it. Two bugs lived in the old skim for a long time, and both
are still regression tests:

1. After the slow window the sampler fell into its "a skim step must not cross
   the window" branch and re-appended ``slow_from - 1`` on every iteration. A
   114-ply game collapsed to 8 frames and the tail of the game was never shown.
   ``test_the_phase_is_covered`` pins the frame count and the span.
2. The plan needed explicit per-frame holds, because fixed beat durations cannot
   express "0.45s here, 2.6s there". ``test_window_frames_are_held_far_longer``.
3. Thinning with ``[::step]`` yields ``ceil(len/step)`` frames, so a floored step
   overshot the cap: 22 frames for a cap of 20. ``test_frame_cap_is_respected``.

And four more, each of which was a defect measured on a rendered clip:

4. The slow window held one board repeatedly -- a 2fps contact sheet read
   ``24... 24... 24... 25... 25...``, because the five plies of the window are
   three move numbers and the second ply of a move is a reply to the first.
   ``test_window_holds_one_frame_per_move_number`` and
   ``test_no_adjacent_frames_show_the_same_position`` pin the fix.
5. The payoff landed at 20% of a 14.3s clip and 8 seconds of skimming followed
   it. ``test_reveal_lands_late`` pins that nothing at all follows it.
6. The last six seconds were a bare king. ``test_clip_stops_when_the_board_is_empty``.
7. The clip opened at move 1 and the viewer said so: "почему с 1 хода все идет".
   ``test_the_clip_opens_a_fixed_distance_before_the_move`` and
   ``test_the_phase_is_the_clip_not_the_game`` pin the fix.

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


def _lead(frames):
    """Frames of the run-up to the decisive move: neither window nor reveal."""
    return [i for i, f in enumerate(frames) if f.accent != "bad"]


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


def _beat_plies(game, target_ply, **kwargs) -> list[int]:
    """Every ply the emitted plan shows, read from the plan's own beats.

    Positions repeat in a real game, so a FEN does not identify its ply and
    matching frames back to plies by position silently reads the wrong one --
    on the 300-ply fixture it reports an opening frame on ply 159 that is
    really on ply 198. Reading the beats off :func:`pacing._drop_repeats` avoids
    that, and reading its *return* value means the seam guard is included rather
    than assumed away.
    """
    captured: dict[str, list[int]] = {}
    original = pacing._drop_repeats

    def spy(beats, placements, target):
        kept = original(beats, placements, target)
        captured["slow"] = [b.ply for b in kept if b.slow]
        captured["all"] = [b.ply for b in kept]
        return kept

    pacing._drop_repeats = spy
    try:
        frames, _ = pacing.pace_game(game, target_ply, **kwargs)
    finally:
        pacing._drop_repeats = original

    fens = _walk(game)
    assert [f.fen for f in frames] == [
        fens[ply] for ply in captured["all"]
    ], "the beats and the shipped frames disagree"
    return captured


def _window_plies(game, target_ply, slow_window: int = 2) -> list[int]:
    """The plies the emitted slow frames show. See :func:`_beat_plies`."""
    return _beat_plies(game, target_ply, slow_window=slow_window)["slow"]


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class PacingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = _random_game()
        self.plies = len(list(self.game.mainline_moves()))

    def test_the_phase_is_covered(self):
        """Coverage now means "the decisive phase", not "the whole game".

        The old definition was the whole game, and holding it is what produced
        the clip that opened on the starting position. A lead-in is short by
        construction, so what is pinned here is the *span*: the clip runs from
        the run-up's first ply to the decisive ply and not one frame further.
        """
        frames, holds = pacing.pace_game(self.game, target_ply=17)

        self.assertEqual(len(frames), len(holds))
        fens = _walk(self.game)
        self.assertEqual(
            frames[0].fen, fens[17 - pacing.LEAD_IN_PLIES], "the run-up moved"
        )
        self.assertEqual(frames[-1].fen, fens[17], "the clip does not end on the move")
        # And the last frames must be the window, not more run-up.
        self.assertEqual(
            frames[-1].accent, "bad", "the clip does not end on the reveal"
        )

        # The last frame must show the last position still worth showing, or
        # the clip either cuts off or spends its tail on a bare king.
        self.assertGreaterEqual(_pieces(frames[-1].fen), pacing.MIN_PIECES_TO_SHOW)

    def test_the_clip_opens_a_fixed_distance_before_the_move(self):
        """"Why does it start at move 1" -- measured back from the move.

        The distance is from the decisive ply, never from the start of the game,
        so it is the same number of plies whether the game is 40 plies or 300.
        """
        fens = _walk(self.game)
        total = len(fens)
        for target in (17, total // 2, total - 1):
            with self.subTest(target=target):
                frames, _ = pacing.pace_game(self.game, target_ply=target)
                self.assertEqual(
                    frames[0].fen,
                    fens[target - pacing.LEAD_IN_PLIES],
                    "the opening frame is not LEAD_IN_PLIES before the move",
                )

        # And on a long game the opening frame is deep into it, not ply 0.
        game = _random_game(plies=300, seed=5)
        long_fens = _walk(game)
        frames, _ = pacing.pace_game(game, target_ply=150)
        self.assertEqual(frames[0].fen, long_fens[150 - pacing.LEAD_IN_PLIES])
        self.assertNotEqual(
            frames[0].fen,
            long_fens[0],
            "a 300-ply game opened on its first move again",
        )

    def test_window_frames_are_held_far_longer(self):
        frames, holds = pacing.pace_game(self.game, target_ply=17)

        slow = [h for h, f in zip(holds, frames, strict=True) if f.accent == "bad"]
        lead = [h for h, f in zip(holds, frames, strict=True) if f.accent != "bad"]
        self.assertTrue(bool(slow), "no slow frames around the decisive move")
        self.assertTrue(bool(lead), "no lead-in frames before the window")

        slow_each = sum(slow) / len(slow)
        lead_each = sum(lead) / len(lead)
        self.assertGreater(
            slow_each / lead_each,
            3.0,
            "the decisive move is not actually slower than the run-up",
        )

    def test_window_ends_on_the_target(self):
        """Nothing follows the reveal.

        A trailing window frame sat on the ply that answers the decisive move,
        which put a neutral skim line ("30 ходов сыграно") on the last image: the
        clip looked like it ended on bookkeeping instead of on the point. The
        reveal is the last thing shown.
        """
        frames, _ = pacing.pace_game(self.game, target_ply=17, slow_window=2)
        self.assertEqual(
            frames[-1].accent, "bad", "the clip does not end on the window"
        )

        plies = _window_plies(self.game, 17)
        self.assertEqual(max(plies), 17, "the window renders a ply after the move")

        # pace_game on its own narrates nothing -- no arrow without a decision --
        # so the marker is checked through the plan, which is what the batch uses.
        decision = analysis.find_decisions(self.game, limit=1, min_ply=6)
        if not decision:
            self.skipTest("no decisive move in the fixture game")
        plan = pacing.plan_for_highlight(self.game, decision[0])
        marker = [i for i, f in enumerate(plan.frames) if f.arrow is not None]
        self.assertEqual(len(marker), 1, "no frame marks the decisive move")
        self.assertEqual(marker[0], len(plan.frames) - 1, "the reveal is not last")

    def test_window_is_centred_on_the_target(self):
        frames, _ = pacing.pace_game(self.game, target_ply=17, slow_window=2)
        slow_positions = _slow(frames)

        # One frame per move number, all of them before the move.
        plies = _window_plies(self.game, 17)
        self.assertEqual(len(plies), 3)
        self.assertEqual(plies[-1], 17, "the decisive ply is the last frame")
        self.assertEqual(len(set(p // 2 for p in plies)), len(plies))
        self.assertTrue(slow_positions)
        # The window reaches back far enough to set the move up, and stops there.
        self.assertLessEqual(min(plies), 17 - 2)
        self.assertEqual(max(plies), 17)

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
        decisive = [f for f in frames if f.accent == "bad"]
        self.assertTrue(decisive, "the window has no frames at all")
        # The decisive ply is the last window frame.
        self.assertEqual(decisive[-1].fen, _fen_after(game, target))

    def test_reveal_lands_late(self):
        """The anticlimax defect: the payoff, then eight seconds of skimming."""
        game = _random_game(plies=200, seed=31)
        # An early blunder is the worst case: ply order alone would put the
        # payoff at 20% of the runtime with a hundred plies still to skim.
        target = 30
        frames, holds = pacing.pace_game(game, target)

        # The window is identified by its accent: the caption moved into the band
        # and top_text is empty, so "вот этот ход" is no longer the marker.
        reveal = [i for i, f in enumerate(frames) if f.accent == "bad"][-1]

        # The clip is the phase, so the payoff is in its last quarter by frame
        # count whatever the target ply was.
        self.assertGreater(
            reveal / len(frames),
            0.75,
            "the decisive move is not near the end of the clip",
        )
        # And nothing at all follows it. Not the rest of the window, not a held
        # final position, not a bookmark.
        self.assertEqual(
            reveal,
            len(frames) - 1,
            "the clip does not end on the payoff",
        )
        self.assertEqual(
            sum(holds[reveal + 1 :]),
            0.0,
            f"{sum(holds[reveal + 1:]):.2f}s follows the payoff",
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

    def test_the_run_up_is_one_beat_a_frame(self):
        """The run-up has to move, and it has to stay on the music.

        The old skim ran at 0.35s a frame, which cannot exist on a 0.45s beat
        grid at all -- it is shorter than a beat, so it either lands between the
        beats or has to be stretched by a sixth every fourth frame. Paying for
        that by shortening the reveal is the one thing a short must not do.

        So the run-up is one frame per ply at exactly one beat, and what the
        format lost in *speed* it made back in *frame count*.
        """
        beat = 60 / 133
        self.assertAlmostEqual(
            pacing.LEAD_IN_SECONDS,
            round(beat, 2),
            places=2,
            msg="a lead-in frame is not one beat at 133 BPM",
        )
        # Readable as a position, not a flicker.
        self.assertGreaterEqual(pacing.LEAD_IN_SECONDS, 0.30)

        frames, holds = pacing.pace_game(self.game, target_ply=17)
        lead = [holds[i] for i in _lead(frames)]
        self.assertTrue(lead)
        self.assertEqual(
            sorted(set(round(h, 3) for h in lead)),
            [pacing.LEAD_IN_SECONDS],
            "the run-up drifted off the beat",
        )

    def test_the_phase_is_the_clip_not_the_game(self):
        """Frame count and length track the phase, not the game's length.

        A 40-ply game and a 300-ply game whose blunders both sit 12 plies before
        their end come out identical. That is the point of the format: the clip
        is as long as the decisive phase is long, and the rest of the game costs
        the viewer nothing at all.
        """
        shapes = set()
        for plies, seed in ((40, 7), (90, 6), (180, 8), (300, 5)):
            game = _random_game(plies=plies, seed=seed)
            total = len(list(game.mainline_moves()))
            target = min(60, total - 1)
            with self.subTest(plies=plies):
                frames, holds = pacing.pace_game(game, target_ply=target)
                shapes.add((len(frames), round(sum(holds), 2)))
        self.assertEqual(
            len(shapes),
            1,
            f"clip length tracks the game rather than the phase: {sorted(shapes)}",
        )

    def test_an_over_long_run_up_is_capped_not_speeded_up(self):
        """Length pressure must cost frames, not legibility.

        The frame cap and the length ceiling between them decide what a caller
        who asks for a 200-ply run-up gets. It has to be a shorter clip at the
        same readable hold, never the same clip played faster.
        """
        game = _random_game(plies=300, seed=5)
        frames, holds = pacing.pace_game(game, target_ply=200, lead_in_plies=200)

        self.assertLessEqual(len(frames), pacing.MAX_FRAMES)
        self.assertEqual(len(frames), len(holds))
        self.assertEqual(frames[-1].accent, "bad", "the cap removed the reveal")
        lead = [holds[i] for i in _lead(frames)]
        self.assertTrue(lead)
        self.assertTrue(
            all(h >= pacing.LEAD_IN_SECONDS for h in lead),
            sorted(set(lead)),
        )
        self.assertLessEqual(sum(holds), pacing.MAX_CLIP_SECONDS + 1e-6)

    def test_the_run_up_covers_the_whole_phase(self):
        """The run-up reaches the window, and nothing sits past the move.

        The clip is a stretch of the game ending on the decisive ply, and the
        only plies allowed to be missing from it are the ones the repeat guard
        dropped because the board did not change -- a viewer cannot tell those
        apart, so showing them costs a frame and says nothing.
        """
        game = _random_game(plies=200, seed=5)
        target = 100
        frames, _ = pacing.pace_game(game, target_ply=target)
        fens = _walk(game)
        plies = _beat_plies(game, target)
        shown = plies["all"]

        self.assertEqual(shown[-1], target, "the clip does not end on the move")
        self.assertEqual(max(shown), target, "a frame sits after the decisive ply")
        # Every frame is inside the phase: nothing before the run-up, nothing
        # after the move.
        self.assertGreaterEqual(
            min(shown),
            target - pacing.LEAD_IN_PLIES,
            "a frame sits outside the decisive phase",
        )
        self.assertEqual(
            shown,
            sorted(shown),
            "the clip is not in ply order",
        )
        # The run-up reaches the window: no gap between the last lead-in frame
        # and the first window frame wider than the window's own reach.
        lead_last = max(p for p in shown if p not in plies["slow"])
        self.assertLessEqual(
            min(plies["slow"]) - lead_last,
            2 * pacing.SLOW_WINDOW_PLIES,
            "there is a gap between the run-up and the window",
        )
        # And the piece rule still clamps a decisive move that lands in the
        # tail the clip is not allowed to reach.
        self.assertGreaterEqual(_last_shown(game), target)
        frames, _ = pacing.pace_game(game, target_ply=_last_shown(game) + 25)
        self.assertEqual(frames[-1].fen, fens[_last_shown(game)])

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
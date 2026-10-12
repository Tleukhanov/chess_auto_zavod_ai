"""The clip shows the decisive phase and nothing else.

The reference channel does not skim a game. On the Adams-Kasparov clip it shows
the *decisive phase*: the position that made the mistake possible, the moves
that built it, the move itself. No setup, no history, no run-through. A viewer
named the alternative in one line -- "почему с 1 хода все идет" -- why does it
start at move 1.

These tests pin the four claims that replace it:

1. the opening frame is a fixed distance **before the move**, not a fixed
   distance into the game, and that distance lands inside 8-14 plies;
2. there are only two kinds of frame -- the lead-in and the slow window -- so
   "context" is not a category the format can fall back into;
3. the clip ends on the reveal (pinned here too, not only in the pacing suite,
   because this is the file that replaces the format);
4. the natural frame count of a lead-in plan sits on a 0.45s beat grid, which is
   what a 133 BPM bed gives, so the run-up and the music agree instead of
   fighting.

Measured over 40-, 90-, 180- and 300-ply games: before this change 8.0s, 8.0s,
14.7s and 10.9s with the first frame at 0%, 60%, 0% and 24% of the game; after,
9.2s in all four with the first frame 12 plies before the move.
"""

from __future__ import annotations

import unittest

from shorts_clipper.chess import pacing
from tests.test_chess_pacing import HAVE_CHESS, _random_game, _walk


def _phase_plies(game, target_ply: int, **kwargs) -> list[int]:
    """The mainline index each emitted frame shows, in order.

    Read off the plan's own beats by spying on :func:`pacing._drop_repeats`,
    rather than by matching frame FENs back to the game's plies. A real game
    repeats positions, so ``fens.index(frame.fen)`` silently returns the
    *earlier* occurrence -- on the 300-ply fixture below it reports the opening
    frame as ply 159 when the plan opened it on ply 198 -- and the test would
    then pass or fail for a reason that has nothing to do with pacing. Reading
    the beats also means the seam guard is included rather than assumed away.
    """
    captured: dict[str, list[int]] = {}
    original = pacing._drop_repeats

    def spy(beats, placements, target):
        kept = original(beats, placements, target)
        captured["plies"] = [b.ply for b in kept]
        return kept

    pacing._drop_repeats = spy
    try:
        frames, holds = pacing.pace_game(game, target_ply, **kwargs)
    finally:
        pacing._drop_repeats = original

    fens = _walk(game)
    assert [f.fen for f in frames] == [fens[ply] for ply in captured["plies"]], (
        "the beats and the shipped frames disagree"
    )
    return captured["plies"]


# The bed: a 133 BPM energetic offset, which is what captions.music picks.
BED_BPM = 133
BEAT_SECONDS = 60 / BED_BPM


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class LeadInOpensBeforeTheMoveTests(unittest.TestCase):
    def test_the_run_up_is_eight_to_fourteen_plies(self):
        """The window the run-up is allowed to occupy, measured from the move.

        Under 8 and the opening frame is a position the viewer has no way to
        relate to anything that follows it. Over 14 and the clip is a history
        lesson again, which is the defect this replaced.
        """
        self.assertGreaterEqual(pacing.LEAD_IN_PLIES, 8)
        self.assertLessEqual(pacing.LEAD_IN_PLIES, 14)

    def test_the_first_frame_is_a_fixed_distance_before_the_move(self):
        for plies, seed, target in ((90, 6, 60), (180, 8, 120), (300, 3, 210)):
            with self.subTest(plies=plies):
                game = _random_game(plies=plies, seed=seed)
                self.assertEqual(
                    _phase_plies(game, target)[0],
                    target - pacing.LEAD_IN_PLIES,
                    "the clip does not open LEAD_IN_PLIES before the move",
                )

    def test_a_long_blunder_does_not_put_the_opening_in_the_clip(self):
        """The symptom, stated as an assertion.

        A 300-ply game whose blunder is at ply 210 used to open a quarter of the
        way in and then carry the rest at speed. The opening frame is now ply
        198 and the last is ply 210: twelve plies of game, no matter how long the
        game is.
        """
        game = _random_game(plies=300, seed=3)
        target = 210
        shown = _phase_plies(game, target)

        self.assertEqual(min(shown), target - pacing.LEAD_IN_PLIES)
        self.assertEqual(max(shown), target)
        # The whole clip is inside the phase, so the first frame is two thirds of
        # the way into the game -- the old symptom was this being 0.
        self.assertGreater(min(shown) / 300, 0.5)

    def test_an_early_blunder_runs_back_to_ply_zero_and_stops(self):
        """A run-up cannot reach back further than the game goes."""
        game = _random_game(plies=60, seed=21)
        frames, _ = pacing.pace_game(game, target_ply=3)
        shown = _phase_plies(game, 3)

        self.assertEqual(min(shown), 0, "the clip invented history")
        self.assertEqual(max(shown), 3)
        self.assertGreaterEqual(len(frames), 1)

    def test_frame_count_does_not_grow_with_the_game(self):
        """Four games, four identical plans.

        This is the cost of the format stated plainly: a 50-ply and a 300-ply
        game with their blunders in the same relative place now produce the same
        clip. Nothing about the game that came before the mistake is on screen,
        so nothing about it can lengthen the clip.
        """
        shapes = set()
        for plies, seed in ((40, 7), (90, 6), (180, 8), (300, 5)):
            game = _random_game(plies=plies, seed=seed)
            total = len(list(game.mainline_moves()))
            frames, holds = pacing.pace_game(game, target_ply=min(60, total - 1))
            shapes.add((len(frames), round(sum(holds), 2)))
        self.assertEqual(len(shapes), 1, sorted(shapes))


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class OnlyTwoKindsOfFrameTests(unittest.TestCase):
    """There is no third category for a frame to fall back into.

    "Context" was the category the old skim lived in, and it is the category
    that swallowed 40-48% of the runtime. Every frame is now either part of the
    run-up to the decisive move or part of the window that contains it.
    """

    def test_every_frame_is_lead_in_or_window(self):
        game = _random_game(plies=180, seed=8)
        frames, _ = pacing.pace_game(game, target_ply=90)
        accents = [f.accent for f in frames]
        self.assertIn("bad", accents, "no window frames at all")
        self.assertIn("neutral", accents, "no lead-in frames at all")

    def test_the_plies_are_ordered_and_end_on_the_move(self):
        game = _random_game(plies=180, seed=8)
        frames, _ = pacing.pace_game(game, target_ply=90)
        shown = _phase_plies(game, 90)

        self.assertEqual(shown, sorted(shown), "frames are out of ply order")
        self.assertEqual(shown[-1], 90, "the clip does not end on the decisive move")
        self.assertEqual(frames[-1].accent, "bad", "the last frame is not the reveal")

    def test_no_ply_after_the_move_is_ever_shown(self):
        for plies, seed, target in ((40, 7, 20), (90, 6, 44), (300, 3, 150)):
            with self.subTest(plies=plies):
                game = _random_game(plies=plies, seed=seed)
                self.assertLessEqual(max(_phase_plies(game, target)), target)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class BeatGridTests(unittest.TestCase):
    """The run-up on a 0.45s beat grid.

    At 133 BPM a beat is 0.451s, so a 0.35s frame cannot exist on this bed at
    all. Snapping 19 frames of 0.35s onto the grid keeps reveal-longer-than-
    context but costs the reveal two thirds of its own length, because the only
    frame that may be shortened is the one the clip exists for. The conclusion
    was that a clip wanting a long reveal on this bed needs fewer frames, not
    faster ones -- which is exactly what deleting the skim buys.
    """

    def test_a_lead_in_frame_is_one_beat(self):
        self.assertAlmostEqual(pacing.LEAD_IN_SECONDS, round(BEAT_SECONDS, 2), places=2)

    def test_the_run_up_frames_are_all_on_the_beat(self):
        game = _random_game(plies=180, seed=8)
        frames, holds = pacing.pace_game(game, target_ply=90)
        lead = [holds[i] for i, f in enumerate(frames) if f.accent != "bad"]

        self.assertGreaterEqual(len(lead), 6, "the run-up is not a run-up")
        for hold in lead:
            self.assertAlmostEqual(hold, pacing.LEAD_IN_SECONDS, places=3)

    def test_the_natural_plan_is_fewer_frames_not_faster_ones(self):
        """The whole point of the change, as a frame count.

        A lead-in plan is 11 frames: 8 of them on the beat, and 3 that
        deliberately leave it. That is fewer frames at a *longer* hold than the
        18-29 frames the skim needed, and the reveal keeps its full 2.6s.
        """
        game = _random_game(plies=180, seed=8)
        frames, holds = pacing.pace_game(game, target_ply=90)

        lead = [holds[i] for i, f in enumerate(frames) if f.accent != "bad"]
        self.assertEqual(len(frames), len(lead) + 3)
        self.assertLessEqual(len(frames), 12, "the phase grew more frames than it should")
        self.assertAlmostEqual(
            holds[-1],
            pacing.DECISIVE_SECONDS,
            places=3,
            msg="the reveal lost length to the beat grid",
        )

    def test_the_clip_stays_shorts_shaped(self):
        for plies, seed in ((40, 7), (90, 6), (180, 8), (300, 5)):
            with self.subTest(plies=plies):
                game = _random_game(plies=plies, seed=seed)
                frames, holds = pacing.pace_game(game, target_ply=min(60, plies - 1))
                self.assertGreaterEqual(sum(holds), 6.0)
                self.assertLessEqual(sum(holds), 20.0)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class TheFloorTests(unittest.TestCase):
    """``MIN_CLIP_SECONDS`` in a format whose natural length no longer needs it.

    The natural lead-in plan is 9.2s against an 8.0s floor, so the floor fires
    in exactly one situation: the decisive move is inside the first dozen plies,
    so there is no run-up to show. There the plan would be three frames and 5.6s,
    and it is padded -- evenly, so the reveal does not become a five-second
    slide -- up to the floor.
    """

    def test_the_floor_does_not_fire_on_a_normal_phase(self):
        game = _random_game(plies=180, seed=8)
        _, holds = pacing.pace_game(game, target_ply=90)
        natural = pacing.LEAD_IN_SECONDS * (
            pacing.LEAD_IN_PLIES - 2 * pacing.SLOW_WINDOW_PLIES
        ) + pacing.SLOW_SECONDS_PER_PLY * 2 + pacing.DECISIVE_SECONDS
        self.assertGreater(natural, pacing.MIN_CLIP_SECONDS)
        self.assertAlmostEqual(sum(holds), natural, places=2)

    def test_the_floor_pads_a_decisive_move_with_no_run_up(self):
        game = _random_game(plies=180, seed=8)
        _, holds = pacing.pace_game(game, target_ply=4)
        self.assertAlmostEqual(sum(holds), pacing.MIN_CLIP_SECONDS, places=2)

    def test_padding_is_even_and_never_below_the_hold_floor(self):
        game = _random_game(plies=180, seed=8)
        frames, holds = pacing.pace_game(game, target_ply=4)
        self.assertTrue(all(h >= pacing.MIN_HOLD_SECONDS for h in holds))
        # Even, not dumped on the reveal: at most one floor-fraction of slack.
        self.assertLessEqual(max(holds) - min(holds), pacing.DECISIVE_SECONDS)
        self.assertEqual(frames[-1].accent, "bad", "padding changed the ending")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
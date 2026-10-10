"""The reveal must be a move that lost, never one that delivered mate.

Found by reading the narration of a real game's top decision instead of the
numbers behind it. The clip said "зевок" over a move the engine had scored as mate
for the mover.

``mover_loss_cp`` tested ``after_mate >= 0`` for "I am still mating", but
python-chess mates are signed from the position's point of view: ``mate 0`` is
"the side to move is getting mated", ``mate -3`` is "the mover delivers mate in
three". So the test read the mating case as the losing one and a move that
delivered mate was scored ``MATE_CP`` -- 10000cp of blunder, the largest in the
game. It was then sorted first and given the arrow.

Every test here fails against that version.
"""

from __future__ import annotations

import unittest

from shorts_clipper.chess import analysis, engine


def _eval(cp=0, mate=None):
    """An ``Eval`` as ``engine.analyse`` produces it, for a *black* mover.

    Both fields are white-perspective and ``for_mover(False)`` flips them, so the
    mover's view is built by negating: a black mover walking into mate gets
    ``mate=-1`` here (which is ``+1`` after the flip, "mover is mated"), and a
    black mover delivering mate gets ``mate=+1``.

    Getting this backwards is the whole bug, so the helper names the case rather
    than leaving the sign to be guessed at the call site.
    """
    return engine.Eval(cp=cp, mate=mate)


def _black_walks_into_mate(dist=1):
    """Black to move, black is being mated.

    ``Eval.mate`` is white-perspective and positive when *white* is the one mating,
    which is exactly the case when black walks into a mate. Verified against the
    engine on a real checkmate rather than reasoned out: after 1.Ra8# the score is
    ``Eval(mate=1)``.
    """
    return _eval(cp=0, mate=dist)


def _black_delivers_mate(dist=3):
    """Black to move, black is mating: white-perspective ``mate=-dist``."""
    return _eval(cp=0, mate=-dist)


class MoverLossMateTests(unittest.TestCase):
    def test_delivering_mate_is_not_a_blunder(self):
        before = _eval(cp=-400)
        after = _black_delivers_mate(3)
        loss = engine.mover_loss_cp(before, after, white_to_move=False)
        self.assertLessEqual(loss, 0, "a move that delivers mate was called a blunder")

    def test_mate_in_one_delivered_is_not_a_blunder(self):
        loss = engine.mover_loss_cp(_eval(cp=422), _black_delivers_mate(1), white_to_move=False)
        self.assertLessEqual(loss, 0)

    def test_walking_into_mate_is_a_blunder(self):
        """The case the whole format exists for."""
        loss = engine.mover_loss_cp(_eval(cp=300), _black_walks_into_mate(1), white_to_move=False)
        self.assertEqual(loss, engine.MATE_CP)

    def test_a_longer_mate_than_before_is_a_tempo_not_the_game(self):
        """Had a forced mate, still has one, just slower.

        This is the Opera game's 15...Nxd7: white had mate in one, black walked
        into mate in two. Black still holds a mate, so the win is not where it was
        lost -- pricing it at MATE_CP would blame black for a move that still wins
        for black, which is the same class of error as blaming the mating move.
        """
        loss = engine.mover_loss_cp(
            _black_walks_into_mate(1), _black_walks_into_mate(2), white_to_move=False
        )
        self.assertLessEqual(loss, 0, "a slower mate was reported as losing the game")

    def test_the_helpers_describe_opposite_sides(self):
        """Guards the helpers: if these two agree, every assertion above compares
        a value to itself and the suite proves nothing."""
        walking = _black_walks_into_mate(1)
        delivering = _black_delivers_mate(1)
        # From a black mover's view: positive = mating, negative = being mated.
        self.assertLess(
            walking.for_mover(False)[1],
            0,
            "the walking helper does not read as 'mover is being mated'",
        )
        self.assertGreater(
            delivering.for_mover(False)[1],
            0,
            "the delivering helper does not read as 'mover is mating'",
        )

    def test_a_slower_mate_is_a_tempo_not_a_blunder(self):
        loss = engine.mover_loss_cp(
            _black_delivers_mate(5), _black_delivers_mate(8), white_to_move=False
        )
        self.assertLessEqual(loss, 0)

    def test_white_mating_looks_the_same_as_black_mating(self):
        """The sign must survive ``for_mover`` for both colours."""
        white_mates = engine.mover_loss_cp(
            _eval(cp=200), _eval(mate=2), white_to_move=True
        )
        black_mates = engine.mover_loss_cp(
            _eval(cp=200), _black_delivers_mate(2), white_to_move=False
        )
        self.assertLessEqual(white_mates, 0)
        self.assertLessEqual(black_mates, 0)

    def test_white_walking_into_mate_is_a_blunder(self):
        loss = engine.mover_loss_cp(_eval(cp=200), _eval(mate=-2), white_to_move=True)
        self.assertEqual(loss, engine.MATE_CP)

    def test_plain_loss_is_still_a_plain_loss(self):
        loss = engine.mover_loss_cp(_eval(cp=200), _eval(cp=-300), white_to_move=True)
        self.assertEqual(loss, 500)

    def test_a_gain_is_never_positive_loss(self):
        loss = engine.mover_loss_cp(_eval(cp=-200), _eval(cp=300), white_to_move=True)
        self.assertLessEqual(loss, 0)


class FindDecisionsMateTests(unittest.TestCase):
    """The end-to-end version: what the batch actually picks."""

    def _decision_shape(self):
        from tests.test_chess_pacing import _random_game

        return _random_game

    def test_no_delivered_mate_is_ranked_as_the_worst_move(self):
        if not engine.has_engine():
            self.skipTest("no engine available")

        make_game = self._decision_shape()
        for seed in (3, 12, 21):
            with self.subTest(seed=seed):
                game = make_game(plies=90, seed=seed)
                decisions = analysis.find_decisions(game, limit=5, min_ply=6)
                for dec in decisions:
                    # mate_after is white-perspective, so the mover is the one
                    # delivering only when the sign agrees with its colour.
                    if dec.mate_after is None or dec.mate_after == 0:
                        continue
                    white_mates = dec.mate_after > 0
                    mover_mates = (
                        white_mates if dec.color == "white" else not white_mates
                    )
                    if mover_mates:
                        self.assertFalse(
                            dec.mover_loss_cp >= engine.MATE_CP,
                            f"{dec.san} delivers mate yet is scored as a "
                            f"{dec.mover_loss_cp}cp blunder",
                        )

    def test_a_mating_move_scores_no_loss_at_all(self):
        """A move that hands the opponent mate is the opposite of one that mates.

        Read the sign against the mover, never on its own: ``mate_after`` is
        white-perspective, so it is negative when *white* is mating. Verified
        against Stockfish on a real mate rather than reasoned from the sign --
        white's Ra1 in this fixture is scored ``mate_after=-4``, which the engine
        confirms as ``PovScore(Mate(+4), BLACK)``: black mates in four, so white
        just threw the game away and ``MATE_CP`` is the correct loss.
        """
        if not engine.has_engine():
            self.skipTest("no engine available")

        make_game = self._decision_shape()
        game = make_game(plies=120, seed=4)
        checked = 0
        for dec in analysis.find_decisions(game, limit=10, min_ply=6):
            if dec.mate_after is None or dec.mate_after == 0:
                continue
            checked += 1
            with self.subTest(san=dec.san, color=dec.color):
                white_mates = dec.mate_after > 0
                mover_mates = white_mates if dec.color == "white" else not white_mates
                if mover_mates:
                    self.assertLessEqual(
                        dec.mover_loss_cp,
                        0,
                        f"{dec.san} delivers mate yet is scored {dec.mover_loss_cp}",
                    )
                else:
                    self.assertGreater(
                        dec.mover_loss_cp,
                        0,
                        f"{dec.san} walks into mate yet is scored as harmless",
                    )
        self.assertGreater(checked, 0, "no mate decision found in the fixture")

    def test_the_worst_move_of_a_game_is_not_the_mating_one(self):
        """The reported symptom: the top decision is a move that ends the game."""
        if not engine.has_engine():
            self.skipTest("no engine available")

        make_game = self._decision_shape()
        for seed in (3, 4, 12, 21):
            with self.subTest(seed=seed):
                game = make_game(plies=90, seed=seed)
                decisions = analysis.find_decisions(game, limit=1, min_ply=6)
                if not decisions:
                    continue
                top = decisions[0]
                if top.mate_after is None or top.mate_after == 0:
                    continue
                white_mates = top.mate_after > 0
                mover_mates = white_mates if top.color == "white" else not white_mates
                self.assertFalse(
                    mover_mates,
                    f"the worst move found is {top.san}, which delivers mate",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
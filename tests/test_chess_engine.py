"""Tests for the Stockfish engine path.

The old engine probe was untested because no engine was installed. Now one is
(``python -c "from shorts_clipper.chess import engine; engine.ensure_stockfish()"``
puts it in the gitignored models dir), so the engine tests below run for real
rather than being skipped -- but they still skip cleanly, because a checkout
without an engine must stay green and must fall back to the material estimate.

What is pinned here, in order of how much damage the bug would do:

- **Perspective.** Stockfish scores from the side to move; the old hand-rolled
  path fed those numbers straight into arithmetic that assumed white's view,
  which hid every white blunder and doubled every black one. The tests that do
  not need an engine drive :func:`analysis.find_decisions` through a stubbed
  :func:`engine.analyse`, so the arithmetic is pinned even with no binary
  present.
- **Mate.** ``score mate`` is a distance in moves, not centipawns. Walking into
  mate has to be reported as a loss without inventing a five-figure pawn count,
  and a winning attack must never read as a blunder.
- **Graceful degradation.** Anything that stops the engine working has to end in
  ``Decision.material = True``, never in a crash and never in results presented
  as analysis.
- **Cost.** Depth and movetime are configuration, and the default is a depth, so
  two runs on the same game agree exactly.
"""

from __future__ import annotations

import io
import os
import tarfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

try:
    import chess
    import chess.engine  # noqa: F401  (binds the submodule)
    import chess.pgn  # noqa: F401  (used by analysis.load_pgn)

    HAVE_CHESS = True
except ImportError:  # pragma: no cover - environment without the optional dep
    HAVE_CHESS = False

from shorts_clipper.chess import analysis, engine

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paul Morphy vs Duke Karl / Count Isouard, Paris 1858. python-chess validates
# every move while parsing, so a misremembered game fails loudly instead of
# quietly producing meaningless expectations.
OPERA_GAME = """[Event "Paris Opera"]
[White "Paul Morphy"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6
9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6
15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
"""

# 15...Nxd7 walks into 16.Qb8+! Nxb8 17.Rd8#, so the position after it is a
# forced mate for white -- a decision the material path cannot possibly see.
OPERA_BLUNDER = "Nxd7"
# 16.Qb8+! is the move that wins the game. The material path flags it as an 865cp
# blunder because black is up a queen on the board until white mates.
OPERA_WINNING_MOVE = "Qb8+"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Back-rank mate: black is stuck behind its own pawns, so Ra8# is mate in 1.
# mate() counts moves for the side to move, which is why this reads as Mate(1).
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
# The same position one ply later, after Ra8#: the side to move is already
# checkmated, which is the "score mate 0" case the old regex read as 0 centipawns.
BLACK_MATED_FEN = "R5k1/5ppp/8/8/8/8/8/7K b - - 0 1"
# White is a queen up. Scoring it with black to move is what proves the perspective
# is white's: a side-to-move-relative engine would report it as negative.
WHITE_WINNING_TO_MOVE = "4k3/ppp5/8/8/8/8/8/3QK3 w - - 0 1"
WHITE_WINNING_BLACK_TO_MOVE = "4k3/ppp5/8/8/8/8/8/3QK3 b - - 0 1"


def _opera_game():
    game = analysis.load_pgn(OPERA_GAME)
    assert game is not None, "the Opera Game PGN must parse"
    return game


def _ply_of(game, san: str) -> int:
    """Index of the ply played as *san*, so a test cannot silently pick the wrong move."""
    positions = analysis._positions(game)
    for index, (_, played, _, _) in enumerate(positions):
        if played == san:
            return index
    raise AssertionError(f"{san} is not in this game: {[p[1] for p in positions]}")


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class EngineResolutionTests(unittest.TestCase):
    """Resolution order: explicit path, then PATH, then the models cache."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for name in (engine.ENGINE_ENV, engine.MODELS_DIR_ENV, engine.DEPTH_ENV):
            os.environ.pop(name, None)
        self.addCleanup(self._env.stop)
        self.addCleanup(engine.reset_cache)

    def test_explicit_path_wins(self):
        with self._as_models_dir() as models:
            fake = models / "not-stockfish-at-all"
            fake.write_bytes(b"")
            os.environ[engine.ENGINE_ENV] = str(fake)
            self.assertEqual(engine.binary(), fake)

    def test_explicit_path_that_is_not_a_file_is_ignored(self):
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir():
                os.environ[engine.ENGINE_ENV] = str(Path("nope") / "stockfish")
                self.assertIsNone(engine.binary())

    def test_path_is_used_when_no_explicit_binary(self):
        with mock.patch("shutil.which", return_value="C:/bin/stockfish.exe"):
            with self._as_models_dir() as models:
                (models / engine.ENGINE_SUBDIR).mkdir(parents=True)
                cached = models / engine.ENGINE_SUBDIR / "stockfish-sf_99.exe"
                cached.write_bytes(b"")
                self.assertEqual(engine.binary(), Path("C:/bin/stockfish.exe"))

    def test_cached_download_is_the_last_resort(self):
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir() as models:
                root = models / engine.ENGINE_SUBDIR
                root.mkdir(parents=True)
                (root / "stockfish-sf_18.exe").write_bytes(b"old")
                newer = root / "stockfish-sf_19.exe"
                newer.write_bytes(b"new")
                os.utime(newer, (2_000_000_000, 2_000_000_000))
                self.assertEqual(engine.cached_binary(), newer)
                self.assertEqual(engine.binary(), newer)

    def test_archives_in_the_models_dir_are_not_mistaken_for_an_engine(self):
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir() as models:
                root = models / engine.ENGINE_SUBDIR
                root.mkdir(parents=True)
                (root / "stockfish-sf_19.zip").write_bytes(b"PK\x03\x04")
                (root / "stockfish-sf_19.json").write_text("{}")
                self.assertIsNone(engine.cached_binary())

    def test_nothing_is_resolved_without_an_engine(self):
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir():
                self.assertIsNone(engine.binary())
                self.assertFalse(engine.has_engine())

    def test_ensure_stockfish_never_downloads_when_one_is_cached(self):
        """The download must be explicit: tests and short renders cannot fetch 80MB."""
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir() as models:
                root = models / engine.ENGINE_SUBDIR
                root.mkdir(parents=True)
                cached = root / engine.installed_name("sf_19")
                cached.write_bytes(b"")
                with mock.patch.object(engine, "download_stockfish") as download:
                    self.assertEqual(engine.ensure_stockfish(), cached)
                download.assert_not_called()

    def test_ensure_stockfish_without_download_returns_none(self):
        with mock.patch("shutil.which", return_value=None):
            with self._as_models_dir():
                self.assertIsNone(engine.ensure_stockfish(download=False))

    def _as_models_dir(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "models"
        root.mkdir(parents=True, exist_ok=True)
        os.environ[engine.MODELS_DIR_ENV] = str(root)
        return _DirContext(root)


class _DirContext:
    def __init__(self, path: Path):
        self._path = path

    def __enter__(self) -> Path:
        return self._path

    def __exit__(self, *exc):
        return False


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class SettingsAndLimitsTests(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for name in (engine.DEPTH_ENV, engine.MOVETIME_ENV, engine.THREADS_ENV, engine.HASH_ENV):
            os.environ.pop(name, None)
        self.addCleanup(self._env.stop)

    def test_defaults_are_a_depth_limit_and_no_wall_clock_cap(self):
        self.assertEqual(engine.depth(), engine.DEFAULT_DEPTH)
        self.assertEqual(engine.movetime_ms(), 0)
        limit = engine.limits()
        self.assertEqual(limit.depth, engine.DEFAULT_DEPTH)
        self.assertIsNone(limit.time, "no time field means Stockfish runs to the depth")

    def test_env_vars_are_honoured(self):
        os.environ[engine.DEPTH_ENV] = "18"
        os.environ[engine.MOVETIME_ENV] = "750"
        os.environ[engine.THREADS_ENV] = "2"
        os.environ[engine.HASH_ENV] = "64"
        self.assertEqual(engine.depth(), 18)
        self.assertEqual(engine.movetime_ms(), 750)
        self.assertEqual(engine.threads(), 2)
        self.assertEqual(engine.hash_mb(), 64)
        limit = engine.limits()
        self.assertEqual(limit.depth, 18)
        self.assertAlmostEqual(limit.time, 0.75)

    def test_garbage_env_vars_fall_back_instead_of_raising(self):
        for value in ("", "  ", "deep", "-4", "0"):
            os.environ[engine.DEPTH_ENV] = value
            self.assertEqual(engine.depth(), engine.DEFAULT_DEPTH, msg=repr(value))
        os.environ[engine.MOVETIME_ENV] = "soon"
        self.assertEqual(engine.movetime_ms(), 0)

    def test_arguments_beat_the_environment(self):
        os.environ[engine.DEPTH_ENV] = "18"
        self.assertEqual(engine.depth(10), 10)
        os.environ[engine.MOVETIME_ENV] = "5000"
        self.assertEqual(engine.movetime_ms(200), 200)


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class EvalSemanticsTests(unittest.TestCase):
    """The arithmetic, exercised without needing a binary."""

    def test_white_blunder_is_a_loss_and_not_a_gain(self):
        """Regression: the two evals used to come from opposite sides.

        White to move and slightly better (+50), black to move and winning by 800
        after the move: white gave up 850cp. Subtracting the two raw engine scores
        gave -750, i.e. a *gain*, so no white blunder was ever reported.
        """
        before = engine.Eval(cp=50)
        after = engine.Eval(cp=-800)
        self.assertEqual(engine.mover_loss_cp(before, after, white_to_move=True), 850)

    def test_black_blunder_is_measured_from_blacks_point_of_view(self):
        before = engine.Eval(cp=-50)
        after = engine.Eval(cp=800)
        self.assertEqual(engine.mover_loss_cp(before, after, white_to_move=False), 850)

    def test_a_good_move_is_a_gain_never_a_loss(self):
        """Black improves from +400 to +500 while black is the mover."""
        before = engine.Eval(cp=-400)
        after = engine.Eval(cp=-500)
        self.assertEqual(engine.mover_loss_cp(before, after, white_to_move=False), -100)
        # And the same numbers for white, who is losing and gets worse.
        self.assertEqual(engine.mover_loss_cp(engine.Eval(cp=400), engine.Eval(cp=500), True), -100)

    def test_walking_into_mate_is_a_mate_sized_loss_not_a_huge_pawn_count(self):
        """White moves from +30 to getting mated, in 3 or in 30.

        The raw subtraction would be ~+20000 either way (mate in 30 is a *bigger*
        number than mate in 3), which is how a caption ends up printing five-figure
        pawn counts. The loss is saturated instead.
        """
        before = engine.Eval(cp=30)
        for distance in (3, 30):
            after = engine.Eval(cp=-(engine.MATE_CP - distance), mate=-distance)
            self.assertEqual(
                engine.mover_loss_cp(before, after, white_to_move=True),
                engine.MATE_CP,
                msg=f"mate in {distance}",
            )

    def test_black_walking_into_mate_is_reported_too(self):
        before = engine.Eval(cp=-30)
        after = engine.Eval(cp=engine.MATE_CP - 4, mate=4)
        self.assertEqual(engine.mover_loss_cp(before, after, white_to_move=False), engine.MATE_CP)

    def test_delivering_mate_is_never_a_blunder(self):
        """White had mate in two and plays it.

        ``mate_after`` is white-perspective, so the move that delivers mate carries
        a *negative* distance -- the sign that was previously read the other way
        round, which is what priced the mating move as a 10000cp blunder.
        """
        before = engine.Eval(cp=engine.MATE_CP - 2, mate=2)
        after = engine.Eval(cp=engine.MATE_CP, mate=-1)
        self.assertLessEqual(engine.mover_loss_cp(before, after, white_to_move=True), 0)

    def test_a_slower_mate_is_a_tempo_not_a_loss(self):
        before = engine.Eval(cp=engine.MATE_CP - 1, mate=1)
        after = engine.Eval(cp=engine.MATE_CP - 9, mate=9)
        self.assertLessEqual(engine.mover_loss_cp(before, after, white_to_move=True), 0)

    def test_throwing_away_a_forced_mate_is_a_loss(self):
        before = engine.Eval(cp=engine.MATE_CP - 1, mate=1)
        after = engine.Eval(cp=-40)
        self.assertEqual(engine.mover_loss_cp(before, after, white_to_move=True), engine.MATE_CP)

    def test_already_lost_positions_do_not_accumulate_fake_losses(self):
        before = engine.Eval(cp=-(engine.MATE_CP - 8), mate=-8)
        after = engine.Eval(cp=-(engine.MATE_CP - 7), mate=-7)
        self.assertLessEqual(engine.mover_loss_cp(before, after, white_to_move=True), 0)

    def test_for_mover_flips_both_cp_and_mate(self):
        evaluation = engine.Eval(cp=-120, mate=-2)
        self.assertEqual(evaluation.for_mover(white_to_move=False), (120, 2))
        self.assertEqual(evaluation.for_mover(white_to_move=True), (-120, -2))

    def test_shared_positions_scores_each_position_once(self):
        game = _opera_game()
        positions = analysis._positions(game)
        before = [p[2].fen() for p in positions]
        after = []
        for _, _, board, move in positions:
            board = board.copy()
            board.push(move)
            after.append(board.fen())
        chain = analysis._shared_positions(before, after)
        self.assertEqual(len(chain), len(before) + 1)
        self.assertEqual(chain[:-1], before)
        self.assertEqual(chain[1:], after)

    def test_shared_positions_refuses_to_pair_a_broken_chain(self):
        before = [START_FEN, START_FEN, START_FEN]
        after = [START_FEN, START_FEN, START_FEN]
        self.assertIsNotNone(analysis._shared_positions(before, after))
        after[1] = "8/8/8/8/8/8/8/4K2R w - - 0 1"  # not the position before ply 2
        self.assertIsNone(analysis._shared_positions(before, after))


class _StubEngine:
    """Stands in for ``chess.engine.analyse`` with a canned list of evals."""

    def __init__(self, evals):
        self.evals = None if evals is None else list(evals)
        self.calls: list[list[str]] = []

    def __call__(self, fens, depth_override=None, movetime_override=None, path=None):
        self.calls.append(list(fens))
        return self.evals


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class AnalysisEngineSemanticsTests(unittest.TestCase):
    """find_decisions driven by canned engine evals: no binary needed."""

    def _find(self, evals, **kwargs):
        stub = _StubEngine(evals)
        with mock.patch.object(engine, "analyse", stub):
            decisions = analysis.find_decisions(_opera_game(), use_engine=True, **kwargs)
        return decisions, stub

    def _flat(self, count: int) -> list[engine.Eval]:
        return [engine.Eval(cp=0) for _ in range(count + 1)]

    def test_one_search_per_position_not_two_per_ply(self):
        game = _opera_game()
        count = len(analysis._positions(game))
        decisions, stub = self._find(self._flat(count), limit=3, min_ply=0)
        self.assertEqual(decisions, [])
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(len(stub.calls[0]), count + 1)

    def test_a_white_blunder_is_reported_with_the_right_loss(self):
        """White hangs material: +40 before, -700 after is 740cp given up."""
        game = _opera_game()
        target = _ply_of(game, "d4")
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=40)
        evals[target + 1] = engine.Eval(cp=-700)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        hits = [d for d in decisions if d.san == "d4"]
        self.assertEqual(len(hits), 1, f"expected 3.d4 to be flagged, got {decisions}")
        self.assertEqual(hits[0].color, "white")
        self.assertEqual(hits[0].mover_loss_cp, 740)
        self.assertFalse(hits[0].material)
        self.assertIsNone(hits[0].mate_before)
        self.assertIn("Ход", hits[0].caption())

    def test_a_black_blunder_is_reported_with_the_right_loss(self):
        """15...Nxd7 without the mate being visible yet still measures correctly.

        -200 before means black was 200 up; +300 after means black is 300 down, so
        black gave up 500cp. Reading the same numbers as white's view would report
        a 100cp gain and miss it.
        """
        game = _opera_game()
        target = _ply_of(game, OPERA_BLUNDER)
        self.assertFalse(analysis._positions(game)[target][2].turn, "must be a black move")
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=-200)
        evals[target + 1] = engine.Eval(cp=300)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        hits = [d for d in decisions if d.san == OPERA_BLUNDER]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].color, "black")
        self.assertEqual(hits[0].mover_loss_cp, 500)

    def test_the_mate_played_by_the_clipped_move_is_not_a_decision(self):
        game = _opera_game()
        target = _ply_of(game, "Rd8#")
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=engine.MATE_CP - 1, mate=1)
        # mate_after is white-perspective: white delivering mate is a *negative*
        # distance, and the delivered mate-in-0 is normalised to -1 because the
        # engine's own ``#0`` has no sign.
        evals[target + 1] = engine.Eval(cp=engine.MATE_CP, mate=-1)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        self.assertEqual([d for d in decisions if d.san == "Rd8#"], [])

    def test_walking_into_mate_is_a_decision_that_says_mate(self):
        """15...Nxd7 loses to forced mate; the caption must say so."""
        game = _opera_game()
        target = _ply_of(game, OPERA_BLUNDER)
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=120)
        evals[target + 1] = engine.Eval(cp=engine.MATE_CP - 2, mate=2)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        hits = [d for d in decisions if d.san == OPERA_BLUNDER]
        self.assertEqual(len(hits), 1)
        found = hits[0]
        self.assertTrue(found.is_mated, "white mates after this move, so black is mated")
        self.assertEqual(found.mate_in, 2)
        self.assertEqual(found.mover_loss_cp, engine.MATE_CP)
        self.assertIn("мат", found.caption())
        self.assertIn("чёрные", found.caption())

    def test_a_win_that_becomes_mate_for_the_mover_is_still_reported(self):
        """The "decisive swing" skip must not swallow a walk into mate."""
        game = _opera_game()
        target = _ply_of(game, "Qb3")  # a white move
        self.assertTrue(analysis._positions(game)[target][2].turn)
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=1500)
        evals[target + 1] = engine.Eval(cp=-(engine.MATE_CP - 4), mate=-4)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        hits = [d for d in decisions if d.san == "Qb3"]
        self.assertEqual(len(hits), 1, "a winning position that walks into mate must be flagged")
        self.assertTrue(hits[0].is_mated)
        self.assertEqual(hits[0].mate_in, 4)

    def test_a_non_mate_swing_across_the_winning_line_is_not_reported(self):
        """+900 to -900 is a decided game, not a move that lost it."""
        game = _opera_game()
        target = _ply_of(game, "Qb3")
        evals = self._flat(len(analysis._positions(game)))
        evals[target] = engine.Eval(cp=900)
        evals[target + 1] = engine.Eval(cp=-900)
        decisions, _ = self._find(evals, limit=5, min_ply=0)
        self.assertEqual([d for d in decisions if d.san == "Qb3"], [])

    def test_engine_failure_falls_back_and_says_so(self):
        game = _opera_game()
        with mock.patch.object(engine, "analyse", _StubEngine(None)):
            decisions = analysis.find_decisions(game, use_engine=True, limit=3, min_ply=6)
        self.assertTrue(decisions, "the material fallback should still find something")
        for decision in decisions:
            self.assertTrue(decision.material, "no engine ran, so nothing may claim otherwise")
            self.assertIsNone(decision.mate_after)
            self.assertIsNone(decision.engine_depth)

    def test_material_path_is_unaffected_by_the_stub(self):
        game = _opera_game()
        decisions = analysis.find_decisions(game, use_engine=False, limit=3, min_ply=6)
        for decision in decisions:
            self.assertTrue(decision.material)


class _FakeResponse:
    """Minimal stand-in for what urlopen returns, so the download path is testable.

    ``read`` honours its size argument the way a real file object does: ignoring it
    turns ``shutil.copyfileobj`` into an infinite writer.
    """

    def __init__(self, payload: bytes):
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
class DownloadTests(unittest.TestCase):
    """Asset selection and install, without touching the network."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(engine.reset_cache)

    def test_pick_asset_prefers_this_platform(self):
        assets = [
            {"name": "stockfish-android-arm64-universal.tar.gz"},
            {"name": "stockfish-macos-universal.tar.gz"},
            {"name": "stockfish-windows-x86-64-avx2.zip"},
            {"name": "stockfish-windows-x86-64-universal.zip"},
        ]
        with mock.patch.dict(engine.__dict__, {"_platform_keys": lambda: ("windows", "x86-64")}):
            chosen = engine.pick_asset(assets)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["name"], "stockfish-windows-x86-64-avx2.zip")

    def test_pick_asset_falls_back_to_the_universal_build(self):
        assets = [
            {"name": "stockfish-linux-x86-64.tar.gz"},
            {"name": "stockfish-linux-x86-64-universal.tar.gz"},
        ]
        with mock.patch.dict(engine.__dict__, {"_platform_keys": lambda: ("linux", "x86-64")}):
            chosen = engine.pick_asset(assets)
        self.assertEqual(chosen["name"], "stockfish-linux-x86-64-universal.tar.gz")

    def test_pick_asset_has_no_answer_on_an_unknown_platform(self):
        assets = [{"name": "stockfish-windows-x86-64.zip"}]
        with mock.patch.dict(engine.__dict__, {"_platform_keys": lambda: None}):
            self.assertIsNone(engine.pick_asset(assets))
        self.assertIsNone(engine.pick_asset([]))

    def test_pick_asset_accepts_the_single_build_name_and_macos_layout(self):
        """Older releases had no flavor suffix, and macOS has no arch in the name."""
        windows = [{"name": "stockfish-windows-x86-64.zip"}, {"name": "stockfish-linux-x86-64.zip"}]
        with mock.patch.dict(engine.__dict__, {"_platform_keys": lambda: ("windows", "x86-64")}):
            self.assertEqual(engine.pick_asset(windows)["name"], "stockfish-windows-x86-64.zip")
        mac = [
            {"name": "stockfish-macos-universal.tar.gz"},
            {"name": "stockfish-linux-x86-64-universal.tar.gz"},
        ]
        with mock.patch.dict(engine.__dict__, {"_platform_keys": lambda: ("macos", "arm64")}):
            self.assertEqual(engine.pick_asset(mac)["name"], "stockfish-macos-universal.tar.gz")

    def test_install_archive_keeps_the_binary_and_drops_the_source(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "stockfish-windows-x86-64.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("stockfish/stockfish-windows-x86-64.exe", b"MZ fake engine")
                zf.writestr("stockfish/src/search.cpp", b"int main(){}")
                zf.writestr("stockfish/README.md", b"# source")
            installed = engine.install_archive(archive, "sf_19", root / "models")
            self.assertEqual(installed, root / "models" / engine.installed_name("sf_19"))
            self.assertTrue(installed.is_file())
            leftovers = {p.name for p in (root / "models").iterdir()}
            self.assertEqual(leftovers, {installed.name}, "no archive or source may be kept")

    def test_install_archive_handles_tarballs(self):
        import io
        import tempfile

        # The POSIX release ships one extension-less file; Windows ships .exe.
        member = "stockfish/stockfish.exe" if os.name == "nt" else "stockfish/stockfish"
        payload = b"fake posix engine"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "stockfish-linux-x86-64.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            installed = engine.install_archive(archive, "sf_19", root / "models")
            self.assertEqual(installed.read_bytes(), payload)

    def test_install_archive_rejects_an_archive_with_no_engine(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "stockfish.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("stockfish/README.md", b"# no binary here")
            with self.assertRaises(FileNotFoundError):
                engine.install_archive(archive, "sf_19", root / "models")

    def test_extraction_refuses_to_write_outside_the_destination(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "evil.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escaped.txt", "nope")
                zf.writestr("stockfish/stockfish", b"fine")
            dest = root / "dest"
            engine.extract_archive(archive, dest)
            self.assertFalse((root / "escaped.txt").exists())
            self.assertTrue((dest / "stockfish" / "stockfish").is_file())

    def test_archive_kind_is_read_from_magic_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zipped = root / "download.part"
            with zipfile.ZipFile(zipped, "w") as zf:
                zf.writestr("stockfish/stockfish", b"x")
            self.assertEqual(engine._archive_kind(zipped), "zip")
            tarred = root / "download2.part"
            with tarfile.open(tarred, "w:gz") as tf:
                info = tarfile.TarInfo("stockfish/stockfish")
                info.size = 1
                tf.addfile(info, __import__("io").BytesIO(b"x"))
            self.assertEqual(engine._archive_kind(tarred), "tar")

    def test_a_failed_download_never_leaves_an_unusable_engine_behind(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ[engine.MODELS_DIR_ENV] = str(root)
            with mock.patch.object(
                engine, "latest_release", side_effect=OSError("offline")
            ):
                self.assertIsNone(engine.ensure_stockfish())
            self.assertFalse((root / engine.ENGINE_SUBDIR / engine.installed_name("sf_19")).exists())

    def test_a_bad_checksum_is_a_failure_not_a_quiet_install(self):
        """The published sha256 is checked; a mismatch must raise, not warn."""
        import hashlib
        import io
        import tempfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("stockfish/stockfish.exe", b"MZ not what was published")
        payload = buffer.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ[engine.MODELS_DIR_ENV] = str(root)
            release = {
                "tag_name": "sf_19",
                "assets": [
                    {
                        "name": "stockfish-windows-x86-64.zip",
                        "browser_download_url": "https://example.invalid/stockfish.zip",
                        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            with mock.patch.object(engine, "latest_release", return_value=release):
                with mock.patch.object(engine, "_platform_keys", lambda: ("windows", "x86-64")):
                    with mock.patch(
                        "urllib.request.urlopen", return_value=_FakeResponse(payload)
                    ):
                        # The digest above matches, so this installs; flip it and it must fail.
                        installed = engine.download_stockfish()
                        self.assertTrue(installed.is_file())
                        installed.unlink()
                        bad = dict(release)
                        bad["assets"] = [dict(release["assets"][0], digest="sha256:" + "0" * 64)]
                        with mock.patch.object(engine, "latest_release", return_value=bad):
                            with mock.patch(
                                "urllib.request.urlopen", return_value=_FakeResponse(payload)
                            ):
                                with self.assertRaises(ValueError):
                                    engine.download_stockfish()
            self.assertFalse(
                (root / engine.ENGINE_SUBDIR / engine.installed_name("sf_19")).exists(),
                "a failed verification must not leave a binary behind",
            )
            leftovers = sorted(p.name for p in (root / engine.ENGINE_SUBDIR).iterdir())
            self.assertNotIn(".download-", "".join(leftovers), "temp downloads must be cleaned up")

    def test_models_dir_is_gitignored(self):
        """A ~100 MB binary must never be committable."""
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        patterns = {line.strip().rstrip("/") for line in ignore if line.strip()}
        self.assertTrue(
            {"models", "models/"} & patterns,
            ".gitignore must keep models/ out of git or the engine binary would be committed",
        )


@unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
@unittest.skipUnless(engine.has_engine(), "no stockfish binary available")
class LiveEngineTests(unittest.TestCase):
    """The real thing. Skipped rather than faked when no engine is installed."""

    @classmethod
    def setUpClass(cls):
        cls.binary = engine.binary()
        cls.evals = engine.analyse([START_FEN], depth_override=10)

    def test_engine_is_resolvable_and_reports_a_version(self):
        self.assertIsNotNone(self.binary)
        with engine.session() as proc:
            self.assertIsNotNone(proc)
            self.assertIn("stockfish", proc.id.get("name", "").lower())

    def test_start_position_is_about_equal(self):
        self.assertIsNotNone(self.evals)
        evaluation = self.evals[0]
        self.assertIsNone(evaluation.mate)
        self.assertFalse(evaluation.is_mate)
        self.assertLess(abs(evaluation.cp), 100, f"{evaluation} is not an equal start position")
        self.assertGreaterEqual(evaluation.depth, 10)
        self.assertIn(evaluation.best_move, ("e2e4", "d2d4", "g1f3", "c2c4", "b1c3"))

    def test_mate_in_one_is_reported_as_mate_not_as_centipawns(self):
        """The bug: `score mate 1` matched `score (cp|mate)` and became 1cp."""
        evals = engine.analyse([MATE_IN_ONE_FEN], depth_override=10)
        self.assertIsNotNone(evals)
        self.assertEqual(evals[0].mate, 1)
        self.assertEqual(evals[0].cp, engine.MATE_CP - 1)
        self.assertGreater(evals[0].cp, 1000)

    def test_score_mate_zero_is_zero_moves_not_zero_centipawns(self):
        """Black is already checkmated. The old regex read `score mate 0` as 0cp.

        From white's point of view that is a mate in zero moves -- mate() == 0 --
        but the centipawn stand-in must be the saturated value, not zero, or a
        position that is already over reads as dead level.
        """
        evals = engine.analyse([BLACK_MATED_FEN], depth_override=10)
        self.assertIsNotNone(evals)
        # ``mate 0`` carries no sign from the engine -- ``#-0`` and ``#+0`` are the
        # same zero -- so it is recovered from the position. Black is checkmated, so
        # white is the one mating and the white-perspective distance is +1.
        self.assertEqual(evals[0].mate, 1)
        self.assertEqual(evals[0].cp, engine.MATE_CP)
        self.assertNotEqual(evals[0].cp, 0)
        # Black's view of the same position: black is the mated side.
        black_cp, black_mate = evals[0].for_mover(white_to_move=False)
        self.assertEqual(black_cp, -engine.MATE_CP)
        self.assertEqual(black_mate, -1)
        # Nothing moves, so nothing is lost.
        self.assertEqual(engine.mover_loss_cp(evals[0], evals[0], white_to_move=True), 0)

    def test_evals_are_white_perspective_regardless_of_who_is_to_move(self):
        """A side-to-move-relative score would be negative when black is to move."""
        evals = engine.analyse(
            [WHITE_WINNING_TO_MOVE, WHITE_WINNING_BLACK_TO_MOVE], depth_override=10
        )
        self.assertIsNotNone(evals)
        white_moves, black_moves = evals
        self.assertGreater(white_moves.cp, 500, "white is a queen up")
        self.assertGreater(black_moves.cp, 500, "black to move does not make the score negative")
        self.assertIsNone(white_moves.mate)
        self.assertIsNone(black_moves.mate)
        self.assertEqual(black_moves.for_mover(True)[0], black_moves.cp)
        self.assertEqual(black_moves.for_mover(False)[0], -black_moves.cp)

    def test_two_runs_at_the_same_depth_agree(self):
        """Depth is the default limit precisely so this holds."""
        first = engine.analyse([START_FEN, MATE_IN_ONE_FEN], depth_override=11)
        second = engine.analyse([START_FEN, MATE_IN_ONE_FEN], depth_override=11)
        self.assertEqual(first, second)

    def test_a_time_limit_is_weaker_but_still_usable(self):
        """Documented tradeoff: a movetime cap bounds the worst position."""
        os.environ[engine.MOVETIME_ENV] = "50"
        self.addCleanup(os.environ.pop, engine.MOVETIME_ENV, None)
        self.assertEqual(engine.movetime_ms(), 50)
        evals = engine.analyse([START_FEN], depth_override=30)
        self.assertIsNotNone(evals)
        self.assertLess(abs(evals[0].cp), 100)

    def test_has_engine_is_true_and_the_analysis_surface_still_works(self):
        self.assertTrue(analysis.has_engine())
        self.assertEqual(analysis._engine_binary(), str(self.binary))
        self.assertEqual(engine.probe(), True)

    def test_the_famous_opera_blunder_is_found_by_the_engine(self):
        decisions = analysis.find_decisions(
            _opera_game(), use_engine=True, limit=6, min_ply=6, critical_cp=200, depth=12
        )
        self.assertTrue(decisions)
        self.assertFalse(any(d.material for d in decisions), "an engine ran here")
        matching = [d for d in decisions if d.san == OPERA_BLUNDER]
        self.assertTrue(
            matching,
            f"15...Nxd7 walks into mate and must be reported; got "
            f"{[(d.san, d.mover_loss_cp) for d in decisions]}",
        )
        self.assertTrue(matching[0].is_mated)
        self.assertEqual(matching[0].color, "black")

    def test_the_engine_does_not_call_the_winning_move_a_blunder(self):
        """16.Qb8+! is the move that wins the game; the material path flags it."""
        game = _opera_game()
        material = {d.san for d in analysis.find_decisions(game, use_engine=False, limit=6, min_ply=6)}
        self.assertIn(OPERA_WINNING_MOVE, material, "the material path's known false positive")
        found = analysis.find_decisions(
            game, use_engine=True, limit=6, min_ply=6, critical_cp=200, depth=12
        )
        self.assertNotIn(OPERA_WINNING_MOVE, {d.san for d in found})

    def test_a_short_game_is_analysed_in_seconds_not_minutes(self):
        import time

        started = time.perf_counter()
        analysis.find_decisions(_opera_game(), use_engine=True, limit=3, min_ply=6, depth=12)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 60.0, "a short render cannot wait a minute for analysis")


if __name__ == "__main__":
    unittest.main()
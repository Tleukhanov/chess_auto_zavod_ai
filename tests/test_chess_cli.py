"""Tests for the ``chess`` subcommand and its batch runner.

The batch is meant to run unattended over a library, so the failures these
cover are the ones that would otherwise only show up in an overnight run:
a PGN that is not a PGN, a file that cannot be read, a file holding several
games, a second run re-cutting the same moments, and a render failure. The
exit code is the contract -- a silent zero on an empty batch is indistinguishable
from success, which is exactly the bug this guards.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

try:
    import chess
    import chess.pgn

    HAVE_CHESS = True
except ImportError:  # pragma: no cover
    HAVE_CHESS = False

from shorts_clipper import __main__ as cli
from shorts_clipper.chess import analysis, batch
from shorts_clipper.core.settings import Settings
from shorts_clipper.pipeline.stock_dedup import record_used


def _blunder_pgn(seed: int = 20261005, want_cp: int = 400, white: str = "A") -> str | None:
    """A legal game whose last move throws away at least *want_cp*."""
    import random

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
            game = chess.pgn.Game(headers={
                "White": white, "Black": f"{white}ko", "Result": "*", "Event": "Unit Test",
            })
            node = game
            for s in [*history, worst[1]]:
                node = node.add_variation(node.board().parse_san(s))
            return str(game)
        san = board.san(rng.choice(list(board.legal_moves)))
        history.append(san)
        board.push_san(san)
    return None


class _Args(argparse.Namespace):
    """Mirror of the chess subparser namespace for direct command calls."""

    def __init__(self, **kw):
        super().__init__(
            path=[], output=None, count=1, critical_cp=None, min_ply=None,
            music=None, no_music=False, seed=7, used_path=None, clear_used=False,
            continue_on_error=False,
        )
        for k, v in kw.items():
            setattr(self, k, v)


def _settings(tmp: Path, **kw) -> Settings:
    base = {
        "chess_out_dir": tmp / "clips",
        "chess_used_path": tmp / "used.json",
        "music_dir": tmp / "music",
    }
    base.update(kw)
    return Settings(**base)


class ParserTests(unittest.TestCase):
    def test_chess_subcommand_is_registered(self):
        parser = cli.build_parser()
        args = parser.parse_args(["chess", "data/pgns", "--count", "3"])
        self.assertEqual(args.command, "chess")
        self.assertEqual(args.path, ["data/pgns"])
        self.assertEqual(args.count, 3)

    def test_chess_flag_defaults_match_settings_defaults(self):
        args = cli.build_parser().parse_args(["chess", "g.pgn"])
        self.assertIsNone(args.critical_cp)
        self.assertIsNone(args.min_ply)
        self.assertFalse(args.no_music)
        self.assertFalse(args.continue_on_error)

    def test_help_renders(self):
        parser = cli.build_parser()
        self.assertIn("chess", parser.format_help())
        chess_help = parser._subparsers._group_actions[0].choices["chess"].format_help()  # noqa: SLF001
        for flag in ("--clear-used", "--continue-on-error", "--critical-cp", "--no-music"):
            self.assertIn(flag, chess_help)

    def test_subcommand_help_exits_cleanly(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["chess", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class CommandExitCodeTests(unittest.TestCase):
    def test_no_path_exits_2_with_a_message(self):
        code = cli.main(["chess"])
        self.assertEqual(code, 2)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_missing_python_chess_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            pgn = Path(tmp) / "g.pgn"
            pgn.write_text(_blunder_pgn() or "", encoding="utf-8")
            original = analysis.available
            analysis.available = lambda: False  # type: ignore[method-assign]
            try:
                self.assertEqual(cli.main(["chess", str(pgn)]), 2)
            finally:
                analysis.available = original  # type: ignore[method-assign]

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_empty_directory_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _Args(path=[str(root)], no_music=True)
            self.assertEqual(cli._cmd_chess(args, _settings(root)), 1)

    def test_missing_directory_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _Args(path=[str(root / "absent")], no_music=True)
            self.assertEqual(cli._cmd_chess(args, _settings(root)), 1)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_nothing_yields_a_clip_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _blunder_pgn()
            if text is None:
                self.skipTest("no blunder found")
            (root / "g.pgn").write_text(text, encoding="utf-8")
            settings = _settings(root)
            result = batch.run_batch(
                [str(root / "g.pgn")],
                settings=settings,
                no_music=True,
                critical_cp=99_999,
            )
            self.assertEqual(result.clips, [])
            self.assertTrue(result.ok, "an above-threshold game is not a failure")

            args = _Args(path=[str(root / "g.pgn")], critical_cp=99_999, no_music=True)
            self.assertEqual(cli._cmd_chess(args, settings), 1)


class BatchSelectionTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_count_is_a_run_wide_budget_not_per_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = _blunder_pgn(seed=1, want_cp=600, white="Big")
            small = _blunder_pgn(seed=2, want_cp=350, white="Small")
            if big is None or small is None:
                self.skipTest("no blunders found")
            (root / "big.pgn").write_text(big, encoding="utf-8")
            (root / "small.pgn").write_text(small, encoding="utf-8")

            selected, results, skipped = batch.plan_batch(
                batch.collect_sources([str(root)]),
                count=1,
                critical_cp=200,
                min_ply=0,
                used=set(),
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.ok for r in results))

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_unreadable_and_non_pgn_are_results_not_exceptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "junk.pgn").write_text("this is not a game at all\n", encoding="utf-8")
            (root / "empty.pgn").write_text("", encoding="utf-8")
            selected, results, _ = batch.plan_batch(
                batch.collect_sources([str(root)]),
                count=1,
                critical_cp=200,
                min_ply=0,
                used=set(),
            )
            self.assertEqual(selected, [])
            self.assertEqual(len(results), 2)
            for r in results:
                self.assertFalse(r.ok)
                self.assertIn("not a PGN", r.error or "")

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_missing_path_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            ghost = Path(tmp) / "nope.pgn"
            selected, results, _ = batch.plan_batch(
                [ghost], count=1, critical_cp=200, min_ply=0, used=set()
            )
            self.assertEqual(selected, [])
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertIn("unreadable", results[0].error or "")

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_multi_game_file_reports_the_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = _blunder_pgn(seed=3, white="One")
            two = _blunder_pgn(seed=4, white="Two")
            if one is None or two is None:
                self.skipTest("no blunders found")
            (root / "multi.pgn").write_text(one + "\n\n" + two, encoding="utf-8")
            _, results, _ = batch.plan_batch(
                batch.collect_sources([str(root)]),
                count=2,
                critical_cp=200,
                min_ply=0,
                used=set(),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].games_in_file, 2, "multi-game file not detected")

    def test_collect_sources_expands_dirs_and_keeps_explicit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pgn").write_text("x", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "b.pgn").write_text("x", encoding="utf-8")
            (root / "notes.txt").write_text("x", encoding="utf-8")
            found = batch.collect_sources([str(root), str(root / "notes.txt"), str(root / "a.pgn")])
            names = {p.name for p in found}
            self.assertEqual(names, {"a.pgn", "b.pgn", "notes.txt"})
            self.assertEqual(len(found), 3, "an explicitly named file must not be listed twice")


class DedupTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_key_is_position_plus_move_not_the_caption(self):
        text = _blunder_pgn()
        if text is None:
            self.skipTest("no blunder found")
        game = analysis.load_pgn(text)
        dec = analysis.find_decisions(game, limit=1, min_ply=0)[0]
        self.assertEqual(batch.moment_key(dec), batch.moment_key(dec))
        # Same position, same move, reworded caption -> same moment.
        other = analysis.Decision(**{**dec.__dict__, "header": {}})
        self.assertEqual(batch.moment_key(other), batch.moment_key(dec))

    def test_used_file_round_trip_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "used.json"
            record_used(history, "abc")
            self.assertEqual(batch.load_used_moments(history), {"abc"})
            self.assertEqual(batch.clear_used(history), 1)
            self.assertEqual(batch.load_used_moments(history), set())
            self.assertEqual(batch.clear_used(history), 0)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_second_run_suppresses_the_same_moment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _blunder_pgn()
            if text is None:
                self.skipTest("no blunder found")
            (root / "g.pgn").write_text(text, encoding="utf-8")
            settings = _settings(root)
            history = Path(settings.chess_used_path)
            self.assertEqual(batch.load_used_moments(history), set())

            first, _, skipped = batch.plan_batch(
                batch.collect_sources([str(root)]),
                count=1, critical_cp=200, min_ply=0, used=set(),
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(skipped, 0)
            record_used(history, first[0].key)

            again, results2, skipped2 = batch.plan_batch(
                batch.collect_sources([str(root)]),
                count=1, critical_cp=200, min_ply=0,
                used=batch.load_used_moments(history),
            )
            self.assertEqual(again, [], "the same moment was offered a second time")
            self.assertEqual(skipped2, 1)
            self.assertEqual(results2[0].skipped_used, 1)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_render_failure_marks_the_source_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _blunder_pgn()
            if text is None:
                self.skipTest("no blunder found")
            (root / "g.pgn").write_text(text, encoding="utf-8")
            settings = _settings(root)
            original = batch.render_moment
            batch.render_moment = lambda *a, **k: None  # type: ignore[assignment]
            try:
                with self.assertRaises(batch.ChessBatchError):
                    batch.run_batch(
                        [str(root)],
                        settings=settings,
                        min_ply=0,
                        no_music=True,
                        used_file=root / "u.json",
                    )
                kept = batch.run_batch(
                    [str(root)],
                    settings=settings,
                    min_ply=0,
                    no_music=True,
                    continue_on_error=True,
                    used_file=root / "u.json",
                )
            finally:
                batch.render_moment = original  # type: ignore[assignment]
            self.assertFalse(kept.clips)
            self.assertFalse(kept.ok, "a failed render must not report success")
            self.assertEqual(kept.failed, 1)
            self.assertIn("ffmpeg", kept.results[0].error or "")

    def test_run_batch_raises_when_no_pgn_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(batch.ChessBatchError):
                batch.run_batch(
                    [str(root / "nothing_here")],
                    settings=_settings(root),
                    no_music=True,
                    used_file=root / "u.json",
                )

    def test_slug_is_filesystem_safe(self):
        decision = analysis.Decision(
            ply=11, move_number=6, color="white", san="Nf6", uci="g1f6",
            fen_before=analysis._START_FEN, fen_after=analysis._START_FEN,
            eval_before_cp=0, eval_after_cp=-300, mover_loss_cp=300,
            material=True, header={"White": "Иванов П.", "Black": "O'Brien, J."},
        )
        moment = batch.Moment(decision=decision, source=Path("x.pgn"))
        self.assertNotIn("/", moment.slug)
        self.assertNotIn(" ", moment.slug)
        self.assertIn("m6", moment.slug)

    def test_music_resolution_reports_a_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))
            self.assertIsNone(batch.resolve_music(None, True, settings))
            self.assertIsNone(batch.resolve_music(None, False, settings))
            (Path(tmp) / "music").mkdir()
            bed = Path(tmp) / "music" / batch.DEFAULT_CHESS_MUSIC
            bed.write_bytes(b"RIFF")
            self.assertEqual(batch.resolve_music(None, False, settings), bed)
            self.assertEqual(batch.resolve_music(bed, False, settings), bed)


if __name__ == "__main__":
    unittest.main()
"""Regression tests for the two disk-exhaustion fixes.

The factory stopped because the drive ran out of space, and both causes were
structural rather than accidental. Neither showed up in a unit test, so both are
pinned here.

1. ``render_clip`` wrote its frame PNGs to ``%TEMP%`` while the finished clip went
   to the output folder. Two halves of one job on two different volumes: with C
   nearly full, writing a PNG failed and the render returned None with no frames
   and no reason. Scratch now lives beside the output.
2. ``run_*`` folders accumulated forever. 45 of them had been left behind, so the
   space they held was never reclaimed before the next run needed it.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from shorts_clipper.chess import analysis, batch, clip

HAVE_CHESS = analysis._chess_module() is not None


class WorkDirTests(unittest.TestCase):
    def test_scratch_lands_beside_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out" / "clip.mp4"
            out.parent.mkdir(parents=True)
            # out.parent IS the output directory itself, so scratch sits in it.
            # The point is that it is not somewhere else entirely.
            work = clip._work_root(out)
            self.assertEqual(Path(work), out.parent)

    def test_env_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "scratch"
            out = Path(tmp) / "out" / "clip.mp4"
            out.parent.mkdir(parents=True)
            old = os.environ.get("SHORTS_CHESS_WORK_DIR")
            os.environ["SHORTS_CHESS_WORK_DIR"] = str(override)
            try:
                self.assertEqual(Path(clip._work_root(out)), override)
                self.assertTrue(override.is_dir(), "override dir was not created")
            finally:
                if old is None:
                    os.environ.pop("SHORTS_CHESS_WORK_DIR", None)
                else:
                    os.environ["SHORTS_CHESS_WORK_DIR"] = old

    def test_unusable_override_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            out = Path(tmp) / "out" / "clip.mp4"
            out.parent.mkdir(parents=True)
            old = os.environ.get("SHORTS_CHESS_WORK_DIR")
            os.environ["SHORTS_CHESS_WORK_DIR"] = str(blocker / "nested")
            try:
                # A path whose parent is a file cannot be made into a directory.
                work = clip._work_root(out)
                self.assertIsNotNone(work)
                self.assertNotEqual(Path(work), blocker / "nested")
            finally:
                if old is None:
                    os.environ.pop("SHORTS_CHESS_WORK_DIR", None)
                else:
                    os.environ["SHORTS_CHESS_WORK_DIR"] = old


class CleanupRendersTests(unittest.TestCase):
    def _seed(self, root: Path, count: int) -> list[Path]:
        made = []
        for i in range(count):
            run = root / f"run_2026100{i}_120000"
            run.mkdir(parents=True)
            (run / "clip.mp4").write_bytes(b"x" * 32)
            made.append(run)
        return made

    def test_keeps_the_newest_and_drops_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, 6)
            removed = batch.cleanup_renders(root, keep_runs=3)

            self.assertEqual(removed, 3)
            left = sorted(p.name for p in root.iterdir())
            self.assertEqual(len(left), 3)
            self.assertEqual(left[-1], "run_20261005_120000")

    def test_never_touches_loose_files(self):
        """An operator may point this at a folder of finished clips."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, 4)
            keep = root / "my_favourite_clip.mp4"
            keep.write_bytes(b"x")

            batch.cleanup_renders(root, keep_runs=1)

            self.assertTrue(keep.is_file(), "cleanup deleted a deliverable")

    def test_never_touches_other_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, 3)
            other = root / "notes"
            other.mkdir()
            (other / "todo.txt").write_text("keep", encoding="utf-8")

            batch.cleanup_renders(root, keep_runs=0)

            self.assertTrue((other / "todo.txt").is_file())

    def test_zero_keeps_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, 3)
            self.assertEqual(batch.cleanup_renders(root, keep_runs=0), 3)
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(batch.cleanup_renders(Path(tmp) / "nope"), 0)

    def test_fewer_runs_than_the_limit_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root, 2)
            self.assertEqual(batch.cleanup_renders(root, keep_runs=5), 0)
            self.assertEqual(len(list(root.iterdir())), 2)


class RenderMomentCleansUpTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_no_silent_intermediate_is_left_behind(self):
        """Two mp4 per clip doubled the output folder for no benefit."""
        from tests.test_chess_pacing import _random_game

        game = _random_game(plies=40, seed=5)
        moments = batch._collect_moments("highlight", game, Path("g.pgn"), 1, 0, 0, 5)
        if not moments:
            self.skipTest("no decisive move in the fixture game")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = moments[0].plan()
            if not plan.frames:
                self.skipTest("fixture produced no frames")
            got = clip.render_clip(plan, out / "probe.mp4")
            if got is None:
                self.skipTest("ffmpeg unavailable")

            leftovers = list(out.glob("*_silent.mp4"))
            self.assertEqual(leftovers, [], "a silent intermediate survived")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
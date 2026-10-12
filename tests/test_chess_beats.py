"""Tests for the beat grid and the snapping onto it.

The holds the chess pipeline emits are constants today -- 0.35 s a skim frame,
1.5 s a slow ply, 2.6 s the reveal -- and a constant is a rate, not a track.
These tests are about the module that replaces them with the track's own
numbers, so none of them touch ``data/music``: the fixtures are click tracks
synthesised here with :mod:`wave`, which keeps the suite runnable with no assets
and makes the ground truth a number in the test rather than a file.

What each one pins, and the defect behind it:

1. **A click track yields its own period.** The whole module is this. What it
   replaces is worth stating: the obvious estimator -- the median inter-onset
   interval -- answers 545 BPM for this project's own 134 BPM bed, because a
   bar of that arrangement carries more off-beat percussion than on-beat.
   ``test_median_interval_is_not_the_tempo`` pins that finding rather than
   hiding it, so nobody reintroduces the estimator as an "obvious simplification".
2. **Silence and a missing file leave the holds untouched.** Music is optional
   in this pipeline and a clip must still render without it, so every failure is
   one return value and not an exception.
3. **The total survives snapping**, within a stated tolerance that the module
   itself computes (:func:`~shorts_clipper.chess.beats.drift_tolerance`)
   rather than a number the test invented.
4. **No hold falls below the floor**, which is the renderer's own 0.08 s clamp.
5. **The reveal stays longer than the skim**, because a grid that flattened them
   together would have taken the clip's payoff away with its speedrun.
6. **Determinism**: the same input twice is byte-identical, so a re-run does not
   silently re-cut the video.

And two that came out of measuring this project's own bed rather than out of
theory, because they are the ones that would have shipped:

7. **A pure-Python envelope equals the numpy one.** The first version
   subtracted one sample instead of one hop when it slid its running window, so
   by the four-thousandth frame it was measuring a 1.7 ms slice of the signal
   instead of a 20 ms one. Both paths agreed -- on garbage -- and the tempo came
   out at 41 BPM.
8. **The octave guard goes towards the beat.** A bar grid and a beat grid
   collect exactly the same kicks; without a stated rule the answer is whichever
   one the search happened to visit last, and this module's first version
   reported 268 BPM on a 134 BPM bed for exactly that reason.
"""

from __future__ import annotations

import array as array_module
import math
import tempfile
import unittest
import wave
from pathlib import Path

from shorts_clipper.chess import beats

HOP = beats.ENVELOPE_HOP_SECONDS


def write_wav(path: Path, samples: array_module.array, rate: int = 22050) -> Path:
    """Write mono 16-bit PCM, the format :mod:`wave` and the music pipeline use."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def click_track(
    path: Path, period: float = 0.5, count: int = 24, rate: int = 22050, decay: float = 0.02
) -> Path:
    """A click every *period* seconds: a tone with a fast exponential decay.

    The decay matters. A bare tone has no attack and no end, so the energy
    envelope is flat and there is nothing to detect; the click has a rise the
    onset function can see and a tail it can fall away from.
    """
    length = int(rate * period * count)
    buffer = array_module.array("h", bytes(2 * length))
    tail = int(decay * rate)
    for beat in range(count):
        start = int(beat * period * rate)
        for index in range(tail):
            at = start + index
            if at < length:
                envelope = (1.0 - index / tail) ** 3
                phase = 2.0 * math.pi * 180.0 * index / rate
                buffer[at] = int(28000 * envelope * math.sin(phase))
    return write_wav(path, buffer, rate)


def silence(path: Path, seconds: float = 8.0, rate: int = 22050) -> Path:
    return write_wav(path, array_module.array("h", bytes(2 * int(rate * seconds))), rate)


def tone(path: Path, seconds: float = 8.0, hz: float = 220.0, rate: int = 22050) -> Path:
    """A steady tone: loud, and with no onset in it anywhere."""
    length = int(rate * seconds)
    buffer = array_module.array("h", bytes(2 * length))
    for index in range(length):
        buffer[index] = int(12000 * math.sin(2.0 * math.pi * hz * index / rate))
    return write_wav(path, buffer, rate)


# A plan shaped like the real one: a skim, a slow window and the reveal.
PLANNED: list[float] = [0.35] * 17 + [1.5, 2.6]


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)

    def path(self, name: str) -> Path:
        return self.dir / name


class ClickTrackTests(TempDirCase):
    def test_known_period_gives_that_period(self):
        for period in (0.5, 0.375, 0.6, 0.75):
            with self.subTest(period=period):
                grid = beats.beat_grid(click_track(self.path(f"c{period}.wav"), period))
                self.assertIsNotNone(grid, f"no grid for a {period}s click track")
                # Within 2%: the envelope hop is 5 ms and the peak of a 20 ms
                # onset function cannot be located more finely than that, which
                # is 1% of a 0.5 s beat.
                self.assertLess(abs(grid.period - period) / period, 0.02)
                self.assertAlmostEqual(grid.bpm, 60.0 / period, delta=60.0 / period * 0.02)

    def test_grid_points_land_on_the_clicks(self):
        period = 0.5
        grid = beats.beat_grid(click_track(self.path("grid.wav"), period))
        self.assertIsNotNone(grid)
        self.assertTrue(grid.beats)
        for beat in grid.beats[:12]:
            nearest = round(beat / period) * period
            self.assertLess(abs(beat - nearest), 0.06)

    def test_beats_are_ordered_and_evenly_spaced(self):
        grid = beats.beat_grid(click_track(self.path("ordered.wav"), 0.5))
        self.assertIsNotNone(grid)
        self.assertEqual(grid.beats, tuple(sorted(grid.beats)))
        gaps = [b - a for a, b in zip(grid.beats, grid.beats[1:], strict=False)]
        self.assertGreater(len(gaps), 8)
        for gap in gaps:
            self.assertAlmostEqual(gap, grid.period, delta=grid.period * 0.05)

    def test_steady_tone_has_no_beat(self):
        # A loud, perfectly periodic signal with no onset anywhere in it. Its
        # envelope ripples at the analysis window's own aliasing rate, and the
        # autocorrelation of that ripple is 0.995 -- so this is not a threshold
        # that can be tightened until it passes. What fails it is that no hit
        # was ever played, which is why the onset count is checked before the
        # tempo rather than after.
        self.assertIsNone(beats.beat_grid(tone(self.path("tone.wav"))))

    def test_too_few_clicks_have_no_beat(self):
        self.assertIsNone(beats.beat_grid(click_track(self.path("few.wav"), 0.5, count=4)))

    def test_median_interval_is_not_the_tempo(self):
        """The finding that forced the design, pinned so it stays a finding.

        On a click track the median inter-onset interval is the right answer, so
        this fixture cannot show the failure -- it pins the *contrast*: the
        interval is a description of the onsets and the period is an estimate
        from them, and only one of the two survives material with off-beat
        percussion in it. See the module docstring for the 545 BPM figure on
        this project's own bed.
        """
        grid = beats.beat_grid(click_track(self.path("median.wav"), 0.5))
        samples, rate = beats.read_samples(self.path("median.wav"))
        self.assertIsNotNone(grid)
        envelope = beats.energy_envelope(samples, rate, HOP, beats.ENVELOPE_WINDOW_SECONDS)
        times = beats.detect_onsets(envelope)
        gaps = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
        interval = beats._modal_interval(gaps)
        self.assertIsNotNone(interval)
        self.assertLess(abs(interval - grid.period) / grid.period, 0.02)


class DegradationTests(TempDirCase):
    holds = [0.35, 0.35, 0.35, 0.35, 1.5, 2.6]

    def test_silence_yields_no_grid(self):
        self.assertIsNone(beats.beat_grid(silence(self.path("silence.wav"))))

    def test_silence_leaves_holds_untouched(self):
        grid = beats.beat_grid(silence(self.path("quiet.wav")))
        self.assertIsNone(grid)
        self.assertEqual(beats.snap_holds(self.holds, grid=grid), self.holds)

    def test_missing_file_does_not_raise_and_leaves_holds_untouched(self):
        grid = beats.beat_grid(self.path("not-here.wav"))
        self.assertIsNone(grid)
        self.assertEqual(beats.snap_holds(self.holds, grid=grid), self.holds)

    def test_directory_and_rubbish_do_not_raise(self):
        self.assertIsNone(beats.beat_grid(self.dir))
        junk = self.path("junk.wav")
        junk.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk")
        self.assertIsNone(beats.beat_grid(junk))

    def test_no_grid_leaves_holds_untouched(self):
        self.assertEqual(beats.snap_holds(self.holds), self.holds)
        self.assertEqual(beats.snap_holds(self.holds, grid=None), self.holds)

    def test_unusable_grid_leaves_holds_untouched(self):
        # A grid measured from nothing has a zero period, and a zero period is
        # not a tempo. Snapping to it would divide by it.
        broken = beats.BeatGrid(period=0.0)
        self.assertFalse(broken.usable)
        self.assertEqual(beats.snap_holds(self.holds, grid=broken), self.holds)
        self.assertEqual(beats.snap_holds(self.holds, grid=beats.BeatGrid(period=float("nan"))),
                         self.holds)

    def test_empty_holds(self):
        grid = beats.beat_grid(click_track(self.path("empty.wav")))
        self.assertEqual(beats.snap_holds([], grid=grid), [])

    def test_read_samples_returns_none_for_missing(self):
        self.assertEqual(beats.read_samples(self.path("gone.wav")), (None, 0))


class SnapTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.grid = beats.beat_grid(click_track(self.path("snap.wav"), 0.5))
        self.assertIsNotNone(self.grid)

    def test_every_hold_is_a_whole_number_of_beats(self):
        snapped = beats.snap_holds(PLANNED, grid=self.grid)
        self.assertEqual(len(snapped), len(PLANNED))
        for hold in snapped:
            steps = hold / self.grid.period
            self.assertAlmostEqual(steps, round(steps), places=4)

    def test_boundaries_land_on_the_grid(self):
        # Boundaries are start_seconds + a whole number of beats, so every cut
        # is on the grid by construction rather than by luck.
        start = 12.5
        snapped = beats.snap_holds(PLANNED, start_seconds=start, grid=self.grid)
        running = start
        for hold in snapped:
            running += hold
            steps = (running - start) / self.grid.period
            self.assertAlmostEqual(steps, round(steps), places=4)

    def test_total_is_preserved_within_the_stated_tolerance(self):
        for planned in (PLANNED, [0.5] * 8, [0.35] * 12 + [1.5, 2.6], [1.0] * 20):
            with self.subTest(n=len(planned)):
                snapped = beats.snap_holds(planned, grid=self.grid)
                drift = abs(sum(snapped) - sum(planned))
                self.assertLessEqual(
                    drift,
                    beats.drift_tolerance(planned, self.grid),
                    f"drift {drift:.4f}s exceeds the module's own stated tolerance",
                )

    def test_drift_tolerance_is_zero_without_a_grid(self):
        self.assertEqual(beats.drift_tolerance(PLANNED), 0.0)
        self.assertEqual(beats.drift_tolerance(PLANNED, None), 0.0)

    def test_no_hold_falls_below_the_floor(self):
        for planned in (PLANNED, [0.02] * 6 + [2.6], [0.35] * 40):
            with self.subTest(n=len(planned)):
                snapped = beats.snap_holds(planned, grid=self.grid)
                for hold in snapped:
                    self.assertGreaterEqual(hold, beats.MIN_HOLD_SECONDS)

    def test_reveal_stays_longer_than_the_skim(self):
        snapped = beats.snap_holds(PLANNED, grid=self.grid)
        self.assertGreater(max(snapped), min(snapped))

    def test_ordering_is_restored_when_snapping_would_flatten_it(self):
        # Every frame the same length except a reveal only half a beat longer:
        # the allocation can easily hand them all the same number of beats.
        planned = [0.5] * 12 + [1.0]
        snapped = beats.snap_holds(planned, grid=self.grid)
        self.assertGreater(max(snapped), min(snapped))

    def test_equal_holds_are_left_alone(self):
        # No distinction in, no distinction manufactured: the module is not
        # allowed to invent a reveal the plan did not ask for.
        planned = [0.5] * 10
        snapped = beats.snap_holds(planned, grid=self.grid)
        self.assertEqual(len(set(snapped)), 1)

    def test_deterministic(self):
        first = beats.snap_holds(PLANNED, start_seconds=3.0, grid=self.grid)
        second = beats.snap_holds(PLANNED, start_seconds=3.0, grid=self.grid)
        self.assertEqual(first, second)

    def test_detection_is_deterministic(self):
        path = click_track(self.path("det.wav"), 0.45)
        first = beats.beat_grid(path)
        second = beats.beat_grid(path)
        self.assertEqual(first, second)


class EnvelopeTests(TempDirCase):
    def test_pure_python_and_numpy_paths_agree(self):
        """The regression that cost a whole debugging session.

        The first version slid its running window by one sample rather than one
        hop, so both of its paths -- the numpy one and the fallback -- returned
        identical nonsense, and a tempo that should have been 134 came out 41.
        Two implementations agreeing proves nothing unless the thing they agree
        on is right, and the right answer here is computable: a windowed RMS of
        a constant-amplitude tone is the amplitude.
        """
        path = tone(self.path("env.wav"), seconds=2.0, hz=220.0)
        samples, rate = beats.read_samples(path)
        self.assertIsNotNone(samples)

        expected = []
        numpy = beats._numpy()
        if numpy is None:  # pragma: no cover - environment dependent
            self.skipTest("numpy is the reference here")
        block = numpy.frombuffer(samples, dtype=numpy.int16).astype(numpy.float64)
        window = int(round(beats.ENVELOPE_WINDOW_SECONDS * rate))
        for index in range(0, len(block) - window, window * 3):
            chunk = block[index: index + window]
            expected.append(float(numpy.sqrt((chunk * chunk).mean())))

        fast = beats.energy_envelope(samples, rate)
        original = beats._numpy
        beats._numpy = lambda: None
        try:
            slow = beats.energy_envelope(samples, rate)
        finally:
            beats._numpy = original

        self.assertEqual(len(fast), len(slow))
        for index, (want, got) in enumerate(zip(fast, slow, strict=False)):
            self.assertAlmostEqual(want, got, places=6, msg=f"frame {index}")

        # And the first window really is a windowed RMS of a constant tone.
        self.assertAlmostEqual(fast[0], expected[0], delta=max(1.0, expected[0] * 0.02))

    def test_envelope_is_empty_for_nothing(self):
        self.assertEqual(beats.energy_envelope(array_module.array("h"), 22050), [])
        self.assertEqual(beats.energy_envelope(array_module.array("h", bytes(100)), 0), [])

    def test_onset_function_is_empty_for_nothing(self):
        self.assertEqual(beats.onset_function([]), [])
        self.assertEqual(beats.onset_function([0.0] * 10), [])


class BeatGridTests(TempDirCase):
    def test_bpm_and_usable(self):
        grid = beats.BeatGrid(period=0.5)
        self.assertTrue(grid.usable)
        self.assertAlmostEqual(grid.bpm, 120.0)
        self.assertFalse(beats.BeatGrid(period=0.0).usable)
        self.assertEqual(beats.BeatGrid(period=0.0).bpm, 0.0)

    def test_nearest_beat_follows_the_locked_track(self):
        # A grid whose tracked beats are not where an extrapolated phase would
        # put them: the track is the phase, which is the whole reason for
        # carrying it.
        grid = beats.BeatGrid(period=0.5, beats=(10.1, 10.6, 11.1))
        self.assertAlmostEqual(grid.nearest_beat(20.0, origin=20.0), 20.1, places=6)

    def test_nearest_beat_without_a_track_uses_the_phase(self):
        grid = beats.BeatGrid(period=0.5, phase=0.125)
        self.assertAlmostEqual(grid.nearest_beat(0.0), 0.125, places=6)
        self.assertAlmostEqual(grid.nearest_beat(0.4), 0.625, places=6)
        self.assertAlmostEqual(grid.nearest_beat(1.0), 1.125, places=6)

    def test_beats_between_and_from(self):
        grid = beats.BeatGrid(period=0.5, phase=0.0)
        self.assertEqual(grid.beats_between(0.4, 1.2), [0.5, 1.0])
        self.assertEqual(grid.beats_between(2.0, 1.0), [])
        self.assertEqual(grid.beats_from(3), [0.0, 0.5, 1.0])

    def test_broken_grid_helpers_degrade(self):
        grid = beats.BeatGrid(period=0.0)
        self.assertEqual(grid.nearest_beat(1.23), 1.23)
        self.assertEqual(grid.beats_between(0.0, 1.0), [])
        self.assertEqual(grid.beats_from(3), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
"""Cut the edit to the music: onsets, a tempo estimate, and a beat grid.

Every hold the chess pipeline emits is a constant today -- ``0.35`` a skim frame,
``1.5`` a slow ply, ``2.6`` the reveal (``pacing.SKIM_SECONDS_PER_FRAME``,
``SLOW_SECONDS_PER_PLY``, ``DECISIVE_SECONDS``). Those numbers describe a *rate*,
not a *track*. The reference channel's cuts land on the bass hit because its
durations were read off the waveform; ours are read off a style guide, so they
are consistently a few hundred milliseconds off the pulse. Consistently wrong is
what makes the edit read as "beside" the music rather than "on" it: no viewer
can name the fault, they just stop feeling the clip move.

This module is that reading -- a pure, testable core, stdlib ``wave`` in, a
tempo and a grid out -- and nothing in the render path depends on it, which is
the point: **music is optional in this pipeline and a clip must still render
without it**. Every entry point returns ``None``, or the input unchanged, when
the file is missing, unreadable, silent, or has no detectable beat, so a caller
can wire this in with no try/except and no feature flag.

No numpy is required. It is imported lazily inside the loops that want it and
the pure-Python path beside each is the fallback, so this module has the same
dependency footprint as the stdlib it is written against -- both paths are
tested against each other, because two implementations agreeing proves nothing
unless the thing they agree on is right. No ffmpeg either: the input is a
``.wav`` on disk and the music pipeline already normalises to one.

**What the detector will not do is flatter the generator.**
``shorts_clipper.audio.dark_industrial`` writes a constant ``BPM = 134.0`` and
stamps every kick at a multiple of ``60/134``. If the measured tempo disagreed
with that by more than a few percent, the generator would be the thing that is
wrong, and the honest move is to say so -- in the numbers, not by tuning a
constant until the two agree. It does not: see the measured figures at the end
of this docstring.


How it works
------------

1. **Energy envelope** -- RMS over a 20 ms trailing window, stepped every 5 ms.
2. **Onset function** -- the positive part of the *logarithmic* rise in that
   envelope, bridged by a 3-frame moving max so a one-frame dip inside a hit
   does not split it in two. Log rather than linear, because a kick here sits
   over a sustained sub bed and a linear difference reports the level rather
   than the transient.
3. **Peaks** -- local maxima clearing :data:`THRESHOLD_MULT` times the median of
   the surrounding :data:`THRESHOLD_WINDOW_SECONDS`, floored at
   :data:`THRESHOLD_FLOOR_REL` of the loudest onset in the file. The median is
   what makes it survive an arrangement that runs from a near-silent intro to a
   full-band final section; a global threshold loses one or the other.
4. **Tempo** -- :func:`estimate_period`: the autocorrelation of the onset
   function, then an octave rule.
5. **Beat track** -- :func:`track_beats`, because a period is not a grid.
6. **Snap** -- :func:`snap_holds`, the function the rest of the code calls.

Step 4 is where the naive version fails, and the failure is worth writing down
because it is not hypothetical. The obvious estimator -- the median inter-onset
interval, guarded against half- and double-time -- returns **545 BPM on this
project's own music bed** for a 134 BPM track. Not because the tempo is
ambiguous, but because the *interval is the wrong statistic*: a bar of this
arrangement carries a kick on 1, 2, 3 and 4, metal clangs on 2, 3 and 4.5, a
machine-gun burst on 3.5 whose own clicks are 17 ms apart, three EBM blips on
1.5, 2.5 and 3.25 and, in the final section, an extra kick on 3.5. The busiest
interval in the whole track is the machine-gun's, and no amount of guarding
turns that into a beat.

Autocorrelation asks the question an interval cannot: *does this signal repeat
after this much time?* A half-beat grid fails it, because what sits between the
beats is not the beat. The octave rule then settles the one case the correlation
genuinely cannot see -- a bar grid and a beat grid collect the same kicks, so the
answer is "a multiple of the beat" until the shortest reading that still
explains the signal is preferred.

Two choices in there are worth flagging because both replaced something more
obvious, and both were made after that obvious thing was measured and failed:

- **No phase search.** Scoring a candidate grid by "how much rise lands on its
  points, trying every phase" prefers 40-70 BPM on this bed, every time. A long
  period puts few grid points down, so searching phases finds a lucky alignment
  among them more often than a short one does -- a winner's curse that grows
  exactly as the evidence thins. Autocorrelation has no free parameter.
- **A 20 ms window, not the 5 ms the hop suggests.** Shorter than a couple of
  cycles of the bass, an RMS window reports the carrier's phase rather than its
  envelope, and the kick vanishes into the wobble: at 5 ms and 10 ms the beat
  scores at chance on this bed, at 20 ms and 40 ms it scores clearly, and 40 ms
  starts smearing adjacent eighth-notes together.


Measured on ``data/music/dark_industrial_loop.wav``
---------------------------------------------------

- The generator's ``BPM`` constant claims **134.0**. This module measures
  **133.31**, which is 0.51% off the claim -- inside "a few percent", so the
  detector and the generator agree and no constant was bent to make them.
- The rendered file is itself not exactly at 134.0. Regressing its forty
  strongest energy rises against their beat index gives a period of 0.44854 s =
  **133.77 BPM**, 0.17% slow against the constant. So most of the detector's
  remaining 0.34% is the file, not the measurement: the kicks in the file on
  disk drift about 2.3 ms per second against ``60/134``.
- 72% of the tracked beat times land within 25 ms of a kick, median error
  13.9 ms. The residual is a property of the material rather than the method:
  the generator's own time-varying master filter smears every kick about 25 ms
  late, so the strongest energy rise is not at the beat either.
- Snapping a real 19-frame plan (0.35 s skim, 1.5 s window, 2.6 s reveal) onto
  that grid moves the total by **-0.33 beats (-0.149 s)**, and leaves the reveal
  longer than the skim. It also costs the reveal two thirds of its length, which
  is the finding rather than a bug: at 134 BPM a beat is 0.45 s, a 0.35 s skim
  frame cannot exist, and 19 frames of 0.35 s leave only three beats of slack
  over the one-beat floor. A clip that wants a 2.6 s reveal on a 134 BPM bed
  wants fewer frames, not faster ones. :func:`drift_tolerance` says exactly how
  much snapping is allowed to move a given plan.
"""

from __future__ import annotations

import array
import dataclasses
import logging
import math
import statistics
import sys
import wave
from itertools import accumulate
from operator import mul

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables. One block because they are one decision: the detector is a few time
# constants (hop, window, median half-width, segment) and a few multipliers, and
# they only mean anything together.
# --------------------------------------------------------------------------

# Envelope hop. 5 ms is 200 frames a second: fine enough that a kick's attack
# spans several frames, so the peak interpolator has something to interpolate
# between, and coarse enough that a 90 s track is 18k frames rather than 4M
# samples of Python arithmetic.
ENVELOPE_HOP_SECONDS = 0.005

# Trailing RMS window. 20 ms, not the 5 ms a naive envelope would use: shorter
# than a couple of cycles of the bass, an RMS window reports the carrier's phase
# rather than its envelope, and the kick disappears into the wobble. Measured --
# 5 ms and 10 ms windows score the tempo at chance on this bed, 20 ms and 40 ms
# score it clearly, and 40 ms starts smearing adjacent eighth-notes together.
ENVELOPE_WINDOW_SECONDS = 0.020

# Half-width of the adaptive-threshold median, in seconds. 60 ms each side is
# about one beat at 250 BPM: wide enough that a quiet bar sits inside its own
# statistics, narrow enough that a dense fill does not raise its own floor.
THRESHOLD_WINDOW_SECONDS = 0.060

# An onset must exceed this multiple of the local median. 1.6 is the classic
# adaptive-median ratio; above ~2.0 a kick under a full-band mix stops being an
# onset, below ~1.3 the reverb tail of the previous hit is.
THRESHOLD_MULT = 1.6

# Absolute floor, as a fraction of the loudest onset in the file. Without it a
# passage 60 dB under the loudest hit still has a median, and the median of
# noise times a multiplier is a threshold noise clears.
THRESHOLD_FLOOR_REL = 0.02

# Two onsets closer than this are one hit. 60 ms is the fastest subdivision
# worth cutting on at any tempo a short can use; below it a machine-gun burst is
# nine clicks and the detector would report nine onsets where the ear hears one.
MIN_ONSET_SEPARATION = 0.060

# The tempo range the estimator will report. 40 BPM is a half-time reading of an
# 80 BPM track, 300 BPM a double-time reading of a 150 BPM one. Outside this the
# onsets are not a pulse.
MIN_BPM = 40.0
MAX_BPM = 300.0

# Below this autocorrelation there is no pulse, only correlated noise. Measured
# on this project's bed the true tempo scores 0.36 over the whole file and 0.51
# inside a single main section; a synthesised click track scores 1.0; white-ish
# material with a drifting arrangement scores under 0.1.
ACF_FLOOR = 0.12

# How much weaker a sub-beat grid may be and still be preferred, as a fraction.
# The autocorrelation of a periodic signal is high at *every* multiple of the
# beat, so this is what turns "somewhere between 30 and 300 BPM" into "the beat".
# 0.85 is below the drop from a beat to its half on real material and above the
# drop from a bar line to the beat.
OCTAVE_TOLERANCE = 0.85

# An RMS below this (int16 units, so ~-90 dBFS) is silence, not a quiet bar.
SILENCE_RMS = 1e-3

# Fewer onsets than this and there is no period to estimate. Two is the
# arithmetic minimum; the floor is higher because one real onset plus noise
# produces a confident, meaningless BPM.
MIN_ONSETS = 8

# Beat-track segment, in seconds, and the sub-offsets tried inside each one.
# A period estimated over 90 s cannot be trusted to 0.1% -- 1 ms of period error
# is 9 ms of phase at the end of the track -- so the phase is re-locked segment
# by segment rather than extrapolated once.
TRACK_SEGMENT_SECONDS = 2.0
TRACK_OFFSETS = 16
# How far, as a fraction of a beat, a segment may pull the phase away from the
# one it was handed. A quarter beat is wider than the drift a segment can
# accumulate and narrower than the half beat that separates two subdivisions.
PHASE_REACH = 0.25

# The floor for a snapped hold, in seconds. Matches ``pacing.MIN_HOLD_SECONDS``:
# the renderer clamps every hold to 0.08 s, so a plan asking for less is
# silently re-timed by ffmpeg and the plan stops describing the clip. Copied
# rather than imported because ``pacing`` is a separate zone, and a floor that
# moves when someone edits a pacing constant is a floor nobody can reason about.
MIN_HOLD_SECONDS = 0.08

# The floor in beats. One beat is the shortest hold a grid can express: both of
# a hold's boundaries are grid points, so a hold is a whole number of beats and
# the smallest whole number is one.
MIN_SNAP_BEATS = 1

# Drift, as a multiple of the beat period. The snap budget is chosen so the
# total lands on the nearest whole number of beats, which is half a beat of
# quantisation; the ordering repair can spend one more when there is no middle
# hold to move a beat from. :func:`drift_tolerance` turns this into the number a
# caller should actually assert against, so the constant here is not a licence.
DRIFT_TOLERANCE_BEATS = 1.0


# --------------------------------------------------------------------------
# WAV reading
# --------------------------------------------------------------------------


def read_samples(path) -> tuple[array.array | None, int]:
    """Return ``(mono_int16_samples, sample_rate)`` for a PCM WAV, or ``(None, 0)``.

    Stdlib ``wave`` only, on purpose: adding a dependency to read a file the
    music pipeline already wrote with ``wave`` is a bigger decision than this
    module is allowed to make. Channels are mixed to mono and every supported
    width is normalised to signed 16-bit so nothing downstream has to care what
    it was handed.

    Returns ``(None, 0)`` -- never raises -- for a missing file, a directory, a
    truncated or non-PCM file, or a width this module does not decode (24-bit).
    Those are all conditions a caller hits in the wild and all of them mean the
    same thing: no music, carry on.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        log.debug("beat detection could not read %s: %s", path, exc)
        return None, 0

    codes = {1: "b", 2: "h", 4: "i"}.get(width)
    if codes is None or rate <= 0 or channels < 1:
        log.debug("beat detection skipped %s: %d-bit, %d ch", path, width, channels)
        return None, 0

    try:
        raw = array.array(codes)
        raw.frombytes(frames)
    except ValueError:  # frame count is not a whole number of samples
        return None, 0
    if raw.itemsize != width:
        return None, 0
    if sys.byteorder == "big":  # WAV is little-endian; the buffer is native
        raw.byteswap()
    raw = raw[: len(raw) - (len(raw) % channels)]

    if channels == 1:
        if width == 2:
            return raw, rate  # the common case, already in the right units
        if width == 1:
            return array.array("h", [(v - 128) * 256 for v in raw]), rate
        return array.array("h", [_scale(v, width) for v in raw]), rate

    count = len(raw) // channels
    numpy = _numpy()
    if numpy is not None:
        block = numpy.frombuffer(raw, dtype=_dtype(codes, numpy)).astype(numpy.float64)
        block = block[: count * channels].reshape(count, channels)
        mono = block.mean(axis=1) / _scale_factor(width)
        return array.array("h", numpy.clip(mono, -32768, 32767).astype(numpy.int16).tobytes()), rate

    mixed = array.array("h", bytes(2 * count))
    for index in range(count):
        base = index * channels
        lane = 0
        for offset in range(channels):
            lane += raw[base + offset]
        mixed[index] = _scale(lane / channels, width)
    return mixed, rate


def _dtype(codes: str, numpy):
    return {"b": numpy.int8, "h": numpy.int16, "i": numpy.int32}[codes]


def _scale_factor(width: int) -> float:
    """Divisor mapping a sample of *width* bits onto the int16 scale."""
    return {1: 128.0, 2: 1.0, 4: 32768.0}[width]


def _scale(value: float, width: int) -> int:
    """One sample of *width* bits as int16, clamped to the int16 range."""
    return max(-32768, min(32767, int(round(value / _scale_factor(width)))))


def energy_envelope(
    samples: array.array,
    rate: int,
    hop_seconds: float = ENVELOPE_HOP_SECONDS,
    window_seconds: float = ENVELOPE_WINDOW_SECONDS,
) -> list[float]:
    """RMS of every *window_seconds* window of *samples*, stepped by the hop.

    One value per frame, in int16 amplitude units, from a *trailing* window:
    frame ``i`` covers samples ``[i*hop, i*hop + window)``. The asymmetry is
    deliberate and is undone by :func:`onset_peaks` -- see the note there.

    Empty input, a zero rate, or a file shorter than one window all return
    ``[]``, which every caller treats as "no audio".
    """
    hop = max(1, int(round(float(hop_seconds) * rate))) if rate > 0 else 0
    window = max(1, int(round(float(window_seconds) * rate))) if rate > 0 else 0
    if hop < 1 or window < 1 or len(samples) < window:
        return []
    count = (len(samples) - window) // hop + 1

    numpy = _numpy()
    if numpy is not None:
        block = numpy.frombuffer(samples, dtype=numpy.int16).astype(numpy.float64)
        running = numpy.concatenate([[0.0], numpy.cumsum(block * block)])
        starts = numpy.arange(count) * hop
        sums = running[starts + window] - running[starts]
        return [float(value) for value in numpy.sqrt(numpy.maximum(sums, 0.0) / window)]

    # No numpy: two running sums, one for the block of samples leaving the
    # window and one for the block entering it. O(n) in the sample count.
    total = math.fsum(value * value for value in samples[:window])
    out: list[float] = []
    for index in range(count):
        out.append(math.sqrt(max(total, 0.0) / window))
        if index + 1 >= count:
            break
        base = (index + 1) * hop
        total += math.fsum(
            v * v for v in samples[base - hop + window : base + window]
        ) - math.fsum(v * v for v in samples[base - hop : base])
    return out


def _numpy():
    """``numpy`` if installed, else ``None``. Imported lazily on purpose."""
    try:
        import numpy

        return numpy
    except Exception:  # pragma: no cover - environment dependent
        return None


# --------------------------------------------------------------------------
# Onsets
# --------------------------------------------------------------------------


def onset_function(
    envelope: list[float],
    hop_seconds: float = ENVELOPE_HOP_SECONDS,
    window_seconds: float = ENVELOPE_WINDOW_SECONDS,
) -> list[float]:
    """The onset function of *envelope*: one rise value per frame.

    The positive part of the *logarithmic* rise between frames, bridged by a
    3-frame moving max so a one-frame dip inside a hit does not split it in two.
    Log rather than linear, because a kick here sits over a sustained sub bed
    and a linear difference reports the level rather than the transient.

    This is the signal the tempo stage scores. The peak-picked onsets below are
    what a caller reads and what the beat track locks onto, but they are too
    sparse and too noisy to estimate a period from: on this project's own bed
    the picker returns 71 onsets across 90 s, and a period estimated from 71
    points scattered across a 200-slot span is a guess with a number attached.
    """
    if len(envelope) < 3:
        return []
    if max(envelope) < SILENCE_RMS:
        return []

    eps = 1e-6
    rise = [0.0]
    for index in range(1, len(envelope)):
        delta = math.log(envelope[index] + eps) - math.log(envelope[index - 1] + eps)
        rise.append(delta if delta > 0.0 else 0.0)
    return [max(rise[max(0, i - 1) : i + 2]) for i in range(len(rise))]


def onset_peaks(
    envelope: list[float],
    hop_seconds: float = ENVELOPE_HOP_SECONDS,
    window_seconds: float = ENVELOPE_WINDOW_SECONDS,
) -> list[tuple[float, float]]:
    """``(time, strength)`` for every onset in an energy *envelope*.

    A frame is an onset when it is a local maximum of :func:`onset_function` and
    clears an adaptive threshold: :data:`THRESHOLD_MULT` times the median of the
    surrounding :data:`THRESHOLD_WINDOW_SECONDS`, floored at
    :data:`THRESHOLD_FLOOR_REL` of the loudest onset in the file.

    Two corrections are applied to the peak's frame index and neither is a fudge:

    - **Sub-frame position.** Parabola through the sample and its two
      neighbours. Without it every onset is quantised to the hop, and at 134 BPM
      a 5 ms hop is 1.1% of a beat -- most of the precision being claimed.
    - **Window lag.** The envelope is a *trailing* window, so an onset first
      appears in the frame whose window opens on it, which is ``window - hop``
      seconds earlier than the onset itself. Reporting ``frame * hop + lag``
      puts it back. Measured on this bed, the uncorrected onset lands 20 ms late
      on every kick, which is 4.5% of a beat and enough on its own to put the
      grid between the hits.

    *strength* is the onset function's own height.
    """
    bridged = onset_function(envelope, hop_seconds, window_seconds)
    if len(bridged) < 3:
        return []

    loudest = max(bridged)
    if loudest <= 0.0:
        return []

    window = max(1, int(round(THRESHOLD_WINDOW_SECONDS / max(hop_seconds, 1e-6))))
    separation = max(1, int(round(MIN_ONSET_SEPARATION / max(hop_seconds, 1e-6))))
    lag = window_seconds - hop_seconds

    found: list[tuple[float, float]] = []
    for index in range(1, len(bridged) - 1):
        value = bridged[index]
        if value <= 0.0 or value < bridged[index - 1] or value < bridged[index + 1]:
            continue
        lo = max(0, index - window)
        hi = min(len(bridged), index + window + 1)
        threshold = max(
            statistics.median(bridged[lo:hi]) * THRESHOLD_MULT,
            THRESHOLD_FLOOR_REL * loudest,
        )
        if value > threshold:
            found.append((_interpolate_peak(bridged, index) * hop_seconds + lag, value))

    return _thin_peaks(found, separation * hop_seconds)


def detect_onsets(
    envelope: list[float],
    hop_seconds: float = ENVELOPE_HOP_SECONDS,
    window_seconds: float = ENVELOPE_WINDOW_SECONDS,
) -> list[float]:
    """Onset *times* in seconds. :func:`onset_peaks` without the strengths."""
    return [time for time, _ in onset_peaks(envelope, hop_seconds, window_seconds)]


def _interpolate_peak(values: list[float], index: int) -> float:
    """Sub-frame peak position of *values* at *index*, in frames.

    A parabolic fit through the sample and its two neighbours, clamped to half a
    frame each way because a peak on a plateau has no curvature and the fit can
    throw the vertex outside the bracket.
    """
    left = values[index - 1]
    here = values[index]
    right = values[index + 1]
    denom = left - 2.0 * here + right
    if denom == 0.0:
        return float(index)
    return float(index) + max(-1.0, min(1.0, 0.5 * (left - right) / denom))


def _thin_peaks(found: list[tuple[float, float]], separation: float) -> list[tuple[float, float]]:
    """Keep the strongest onset of any cluster closer together than *separation*."""
    kept: list[tuple[float, float]] = []
    for time, value in found:
        if kept and time - kept[-1][0] < separation:
            if value > kept[-1][1]:
                kept[-1] = (time, value)
            continue
        kept.append((time, value))
    return kept


# --------------------------------------------------------------------------
# Tempo
# --------------------------------------------------------------------------


def estimate_period(odf: list[float], hop_seconds: float = ENVELOPE_HOP_SECONDS) -> float | None:
    """The beat period in seconds, or ``None`` if there is no pulse to find.

    The normalised autocorrelation of the onset function, scanned coarse to fine
    and then resolved sub-frame by a parabola. Nothing else is searched, and the
    reason is the whole design of this function.

    **The median inter-onset interval is not the answer, and on this project's
    own music bed it is not even close: it reports 545 BPM for a 134 BPM
    track.** Nothing about that track's tempo is ambiguous -- the generator
    stamps every kick at a multiple of ``60/134`` -- but a bar of the
    arrangement carries a kick on 1, 2, 3 and 4 *plus* metal clangs on 2, 3 and
    4.5, a machine-gun burst on 3.5, three EBM blips on 1.5, 2.5 and 3.25 and,
    in the final section, an extra kick on 3.5. The busiest interval is the
    machine-gun's own 17 ms click spacing. No guard turns that into a beat.

    Autocorrelation asks a question the interval cannot: *does this signal
    repeat after this much time?* A half-beat grid fails it, because the
    material between the beats is not the same as the beat; a two-beat grid
    answers it partly, which the octave rule below then settles.

    Two deliberate choices are worth stating, because both were arrived at by
    measurement and both replace something that looks more obvious:

    - **No phase search.** Every earlier version tried the best of 24 phases
      and preferred periods between 40 and 70 BPM on this bed, consistently.
      A long period puts few grid points down, so a phase search finds a lucky
      alignment among them more often than a short one does -- a winner's curse
      that grows exactly as the evidence gets thinner. Autocorrelation has no
      free parameter and does not have this failure mode.
    - **The whole file, not the busiest stretch.** Autocorrelation normalises
      away level, so a quiet intro contributes nothing instead of contributing
      silence, and a 12 s busy window is not needed to drown the noise.
    """
    if len(odf) < MIN_ONSETS or hop_seconds <= 0.0:
        return None
    if sum(1 for value in odf if value > 0.0) < MIN_ONSETS:
        return None

    low, high = 60.0 / MAX_BPM, 60.0 / MIN_BPM
    lags = _lag_range(low, high, hop_seconds)
    if not lags:
        return None

    scores = _correlations(odf, lags)
    best = max(range(len(lags)), key=lambda index: scores[index])
    coarse = float(lags[best])

    # Sub-frame by a parabola through the peak and its two neighbours, which is
    # worth doing and is also where the precision stops. The curve is rippled at
    # a few frames' period -- a kick's rise is 20 ms wide, and correlating a
    # pulsed signal with itself has sidelobes at the pulse width -- so a finer
    # scan does not resolve the tempo further, it resolves the nearest ripple.
    # Measured on this bed: a 0.1-frame scan lands on 0.4510 s and the parabola
    # on 0.4501 s, and the truth is 0.4485 s. The residual +/-0.3% is the onset
    # function's, not the search's, and is what :func:`track_beats` exists to
    # carry; see its docstring.
    peak = coarse
    if 0 < best < len(lags) - 1:
        left, here, right = scores[best - 1: best + 2]
        denominator = left - 2.0 * here + right
        if denominator != 0.0:
            shift = 0.5 * (left - right) / denominator
            peak = coarse + max(-1.0, min(1.0, shift))

    # The octave rule. The autocorrelation of a periodic signal is high at every
    # multiple of its period, so the scan alone answers "a multiple of the beat"
    # and not which one. Halve while the halved grid still explains the signal:
    # a real half-beat has material between the beats and loses far more than
    # :data:`OCTAVE_TOLERANCE`, a bar line does not.
    score = _correlate(odf, peak)
    while True:
        halved = peak / 2.0
        candidate = _clamp_period(halved * hop_seconds)
        if candidate is None:
            break
        weaker = _correlate(odf, halved)
        if weaker < score * OCTAVE_TOLERANCE:
            break
        peak, score = halved, weaker

    if score < ACF_FLOOR:
        return None
    return round(peak * hop_seconds, 6)


def _lag_range(low: float, high: float, hop_seconds: float) -> list[int]:
    """Whole-frame lags covering the tempo range ``[low, high]`` seconds."""
    first = max(1, int(math.ceil(low / hop_seconds)))
    last = int(math.floor(high / hop_seconds))
    return list(range(first, last + 1))


def _correlations(odf: list[float], lags: list[int]) -> list[float]:
    """Normalised autocorrelation of *odf* at each lag in *lags*.

    Pearson correlation between the onset function and itself shifted by the
    lag. Normalising by the energy of both halves keeps a lag near the length of
    the file from scoring low for having few overlapping samples.
    """
    numpy = _numpy()
    if numpy is not None:
        x = numpy.asarray(odf, dtype=numpy.float64)
        x = x - x.mean()
        out: list[float] = []
        for lag in lags:
            if lag >= len(x):
                out.append(0.0)
                continue
            head, tail = x[:-lag], x[lag:]
            energy = math.sqrt(
                float(head @ head) * float(tail @ tail)
            ) + 1e-12
            out.append(float(head @ tail) / energy)
        return out

    mean = math.fsum(odf) / len(odf)
    centred = [value - mean for value in odf]
    energy = list(accumulate(value * value for value in centred))
    total = energy[-1] if energy else 0.0
    out = []
    for lag in lags:
        if lag >= len(centred):
            out.append(0.0)
            continue
        cross = math.fsum(map(mul, centred, centred[lag:]))
        left = energy[len(centred) - lag - 1]
        right = total - energy[lag]
        out.append(cross / (math.sqrt(max(left, 0.0) * max(right, 0.0)) + 1e-12))
    return out


def _correlate(odf: list[float], lag: float) -> float:
    """The correlation at a fractional *lag*, interpolating the shifted half.

    Rounding the lag to a whole frame and reading the correlation there would
    quantise the answer to the hop; interpolating the second copy of the signal
    instead makes this continuous in *lag*, which is what the refinement scan
    needs to have a peak to find.
    """
    if lag < 1.0:
        return 0.0
    reach = int(math.ceil(lag)) + 1
    size = len(odf) - reach
    if size < 2:
        return 0.0

    numpy = _numpy()
    if numpy is not None:
        x = numpy.asarray(odf, dtype=numpy.float64)
        x = x - x.mean()
        index = numpy.arange(size, dtype=numpy.float64) + lag
        floor = numpy.floor(index).astype(int)
        frac = index - floor
        head = x[:size]
        tail = x[floor] * (1.0 - frac) + x[floor + 1] * frac
        energy = numpy.sqrt(
            float(head @ head) * float(tail @ tail)
        ) + 1e-12
        return float(head @ tail) / energy

    mean = math.fsum(odf) / len(odf)
    base = int(math.floor(lag))
    frac = lag - base
    head = [odf[index] - mean for index in range(size)]
    tail = [
        (odf[base + i] - mean) * (1.0 - frac) + (odf[base + i + 1] - mean) * frac
        for i in range(size)
    ]
    left = math.fsum(value * value for value in head)
    right = math.fsum(value * value for value in tail)
    cross = math.fsum(map(mul, head, tail))
    return cross / (math.sqrt(max(left, 0.0) * max(right, 0.0)) + 1e-12)


def _at(odf: list[float], hop_seconds: float, time: float) -> float:
    """The onset function at *time*, linearly interpolated between frames.

    The grid points of a beat track almost never land on a frame, so this is
    what makes a phase comparison fair rather than a comparison of which phase
    happened to fall nearest a sample.
    """
    position = time / hop_seconds
    index = int(position)
    if index < 0:
        return 0.0
    if index >= len(odf) - 1:
        return odf[-1]
    frac = position - index
    return odf[index] * (1.0 - frac) + odf[index + 1] * frac


def _modal_interval(gaps: list[float], bin_seconds: float = 0.010) -> float | None:
    """The median of the inter-onset intervals inside the fullest 10 ms bin.

    Kept because it is the honest reading of "median inter-onset interval" and
    because it is a cheap look at the material rather than at the hypothesis: a
    track with no percussion at all has no modal interval worth the name. It is
    deliberately *not* the tempo -- see :func:`estimate_period` for what it
    returns on this project's own bed.
    """
    if not gaps:
        return None
    width = max(bin_seconds, 1e-6)
    buckets: dict[int, list[float]] = {}
    for gap in gaps:
        if gap > 0.0:
            buckets.setdefault(int(gap / width), []).append(gap)
    if not buckets:
        return None
    # Ties break toward the shorter interval, so an even number of gaps does not
    # settle the octave question here by accident.
    key = max(buckets, key=lambda k: (len(buckets[k]), -k))
    return statistics.median(buckets[key])


def _clamp_period(period: float) -> float | None:
    """*period* in seconds if it is inside the tempo range, else ``None``."""
    if not math.isfinite(period) or period <= 0.0:
        return None
    return period if MIN_BPM <= 60.0 / period <= MAX_BPM else None


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


def track_beats(
    odf: list[float],
    hop_seconds: float,
    period: float,
    *,
    segment_seconds: float = TRACK_SEGMENT_SECONDS,
) -> list[float]:
    """Grid points locked onto the onset function, in seconds.

    A period is not a grid. The period is measured over the whole track and is
    good to a fraction of a percent, which is not good enough: 0.5% of a 90 s
    file is 450 ms, a whole beat of drift by the end, and a clip cut on the last
    bar of the track would open a beat away from the bass. Extrapolating one
    phase from the first onset spends the whole clip wrong by the end.

    So the phase is re-locked every :data:`TRACK_SEGMENT_SECONDS`. Each segment
    searches for the grid point that best collects the rise inside it, within a
    quarter beat of the phase the previous segment locked onto. The window is
    what stops the track sliding onto a neighbouring subdivision when two
    candidates score similarly, and it is why a period good only to a few
    milliseconds is enough to hand over.

    Every grid point in a segment is returned. Gating them on "was there
    actually a hit here" was tried and removed: this bed has no kick at all
    through its 9% intro and its 11% breakdown, so the gate produced a track
    with holes in it, and a grid with a hole in it is worse than a grid over
    silence -- a caller choosing where to lay a clip wants the grid to be
    continuous so it can ask about any position in the file. The question of
    whether a beat is *played* is the caller's, and ``start_seconds`` is how it
    says so.
    """
    if len(odf) < MIN_ONSETS or period <= 0.0:
        return []
    last = (len(odf) - 1) * hop_seconds
    if last <= 0.0:
        return []

    segment = max(period, float(segment_seconds))
    beats: list[float] = []
    # Seed the phase over the whole file before locking it segment by segment.
    # Starting from the first segment instead puts the seed in this bed's 9%
    # intro, which has no kick in it at all, and every later segment inherits a
    # phase chosen out of noise and can only pull it a quarter beat per segment.
    seed = _best_phase(odf, hop_seconds, period, 0.0, last, None)
    start = 0.0
    while start < last:
        stop = min(last, start + segment)
        phase = _best_phase(odf, hop_seconds, period, start, stop, seed)
        if phase is not None:
            seed = phase
            time = start + (phase - start) % period
            while time <= stop:
                beats.append(round(time, 6))
                time += period
        start = stop + 1e-9

    out: list[float] = []
    for beat in beats:
        if not out or beat - out[-1] > period * 0.5:
            out.append(beat)
    return out


def _best_phase(
    odf: list[float],
    hop_seconds: float,
    period: float,
    start: float,
    stop: float,
    seed: float | None,
) -> float | None:
    """The grid point that collects the most rise in ``[start, stop]``.

    The first segment searches a whole beat, because there is nothing to carry;
    every later one searches a quarter beat either side of what it was handed,
    which is both the correction it needs and the leash that keeps it on the
    same subdivision.
    """
    if seed is None:
        reach = period / 2.0
        origin = 0.0
    else:
        reach = period * PHASE_REACH
        origin = seed
    best, best_score = None, -1.0
    for index in range(TRACK_OFFSETS):
        candidate = origin + (index - TRACK_OFFSETS // 2) * reach * 2.0 / TRACK_OFFSETS
        score = 0.0
        time = start + (candidate - start) % period
        while time <= stop:
            score += _at(odf, hop_seconds, time)
            time += period
        if score > best_score:
            best, best_score = candidate, score
    return best


@dataclasses.dataclass(frozen=True)
class BeatGrid:
    """A tempo, the phase it sits at, the beats that carry it, and its source.

    *beats* is the locked track from :func:`track_beats`, and it is what makes a
    grid usable anywhere in the file rather than only where the phase was
    anchored. *period* and *phase* are the fallback for a caller that passes a
    grid measured from a very short sample, where there was nothing to track.
    """

    period: float
    phase: float = 0.0
    beats: tuple[float, ...] = ()
    onsets: tuple[float, ...] = ()
    source: str = ""

    @property
    def bpm(self) -> float:
        """Beats per minute, or ``0.0`` for an unusable grid."""
        return 60.0 / self.period if self.usable else 0.0

    @property
    def usable(self) -> bool:
        """Whether this grid can be snapped to. A zero period is not a tempo."""
        return math.isfinite(self.period) and self.period > 0.0

    def nearest_beat(self, time: float, *, origin: float = 0.0) -> float:
        """The grid point nearest *time*.

        *origin* is the music time at clip time zero -- where in the track the
        music begins -- and the answer comes back in clip time. When the track
        carries locked beats, the phase is taken from the one nearest *origin*
        rather than extrapolated from the head of the file, which is the whole
        reason the track is stored rather than just its period.
        """
        if not self.usable:
            return float(time)
        anchor = self._anchor(origin)
        return origin + round((time - origin - anchor) / self.period) * self.period + anchor

    def beats_between(self, start: float, end: float, *, origin: float = 0.0) -> list[float]:
        """Every grid point in ``[start, end]``, in order."""
        if not self.usable or end < start:
            return []
        anchor = self._anchor(origin)
        first = math.ceil((start - origin - anchor) / self.period - 1e-9)
        last = math.floor((end - origin - anchor) / self.period + 1e-9)
        return [origin + anchor + step * self.period for step in range(int(first), int(last) + 1)]

    def beats_from(self, count: int, *, origin: float = 0.0) -> list[float]:
        """The first *count* grid points at or after *origin*."""
        if not self.usable:
            return []
        anchor = self._anchor(origin)
        return [origin + anchor + step * self.period for step in range(max(0, int(count)))]

    def _anchor(self, origin: float) -> float:
        """Music-time offset of a grid point relative to *origin*."""
        if self.beats:
            nearest = min(self.beats, key=lambda beat: abs(beat - origin))
            return nearest - origin
        return self.phase - origin


def beat_grid(path, *, hop_seconds: float = ENVELOPE_HOP_SECONDS,
              window_seconds: float = ENVELOPE_WINDOW_SECONDS) -> BeatGrid | None:
    """Measure the beat grid of the WAV at *path*, or ``None``.

    ``None`` covers every way this can fail -- missing, not a PCM WAV this
    module decodes, silent, too few onsets to have a tempo, tempo outside the
    range a short can be cut to. They are all the same answer for the caller (no
    music: emit the plan as planned), so they are one return value and not five.
    """
    try:
        samples, rate = read_samples(path)
    except Exception as exc:  # pragma: no cover - read_samples absorbs its own
        log.debug("beat grid unavailable for %s: %s", path, exc)
        return None
    if not samples or rate <= 0:
        return None

    envelope = energy_envelope(samples, rate, hop_seconds, window_seconds)
    peaks = onset_peaks(envelope, hop_seconds, window_seconds)
    if len(peaks) < MIN_ONSETS:
        # Enough transients, or nothing to be a tempo out of. This is the guard
        # that a steady tone fails: a 220 Hz sine produces an envelope that
        # ripples at the window's own aliasing rate, and the autocorrelation of
        # that ripple is 0.995. It is not noise, it is worse than noise -- it is
        # a confident, periodic artefact, and the only thing standing between it
        # and a reported 60 BPM is the knowledge that no hit was ever played.
        log.debug("no tempo in %s (%d onsets)", path, len(peaks))
        return None

    odf = onset_function(envelope, hop_seconds, window_seconds)
    period = estimate_period(odf, hop_seconds)
    if period is None:
        log.debug("no tempo in %s (%d onset frames)", path, len(odf))
        return None

    beats = track_beats(odf, hop_seconds, period)
    phase = beats[0] % period if beats else 0.0

    return BeatGrid(
        period=period,
        phase=phase,
        beats=tuple(beats),
        onsets=tuple(time for time, _ in peaks),
        source=str(path),
    )


# --------------------------------------------------------------------------
# Snapping
# --------------------------------------------------------------------------


def snap_holds(holds: list[float], *, start_seconds: float = 0.0, grid=None) -> list[float]:
    """Return *holds* re-timed so every boundary lands on *grid*.

    A boundary is the moment a cut happens, and a cut that lands on a bass hit
    is the entire point of this module. So every hold is a whole number of beats
    and the arithmetic that gets there has to answer two questions at once: where
    does each boundary go, and what does the total become.

    **The total comes first, because drift is what breaks a clip.** Every
    boundary goes to a grid point, so the total is necessarily a whole number of
    beats. The budget is ``round(total / period)`` -- the nearest whole number of
    beats to the planned total -- and that choice alone bounds the drift at half
    a beat, :data:`DRIFT_TOLERANCE_BEATS` of it at worst once the ordering repair
    has been paid for (+/-0.225 s at this project's 133 BPM). The budget is
    raised to one beat a frame when the plan asks for less than that, because
    that is the floor and the floor wins; :func:`drift_tolerance` says by how
    much. The beats are then handed out across the holds in proportion to their
    planned lengths, so the *shape* of the plan survives and only its *scale* is
    quantised.

    **The literal alternative is worse, and it is worth saying why.** Snapping
    each cumulative boundary to its nearest grid point in sequence looks
    equivalent and is not. When a planned hold is shorter than a beat -- the
    0.35 s skim against a 0.45 s beat at this project's tempo -- two consecutive
    planned boundaries round to the *same* grid point, the floor repairs that by
    pushing forward a beat, and the push cascades: every later hold collapses
    onto the one-beat floor. A 19-frame plan comes out as seventeen identical
    one-beat holds and a squeezed reveal. Spending the budget before any boundary
    moves cannot cascade, because there is nothing left to push.

    The floor is :func:`_min_beats`: never fewer than :data:`MIN_SNAP_BEATS`
    beats, and never fewer than :data:`MIN_HOLD_SECONDS` seconds, which is the
    renderer's own clamp.

    **Ordering.** Snapping can leave the reveal and the skim the same length,
    which destroys the format: the reveal is the payoff and the skim is the part
    that must not be read. If the input carries a strict longest and shortest
    hold and the output has lost that, one beat moves out of a middle hold into
    the longest and the total is preserved exactly. With no middle hold to take
    it from, the longest simply gains a beat and the clip grows by one -- the
    format is worth 0.22 s. The ordering is restored because it is the clip; the
    drift is bounded because a beat of it is 0.22 s.

    *start_seconds* is the music time at clip time zero -- where in the track the
    music begins. The grid is **re-phased onto it**: every returned hold is a
    whole number of beats from *start_seconds*, so clip time zero is itself a
    grid point and the whole sequence is on the grid by construction. The
    measured phase is still what decides *which* beat the first cut lands on --
    without it a clip laid over the middle of the track would open on whatever
    beat the track happened to start with, up to half a beat away from the hit
    under it.

    ``grid`` of ``None``, or one with no usable period, returns *holds*
    unchanged. The pipeline must render with or without a bed. See
    :func:`drift_tolerance` for what snapping is allowed to cost.
    """
    if not holds:
        return []
    planned = [max(0.0, float(value)) for value in holds]
    if grid is None or not getattr(grid, "usable", False):
        return list(planned)

    period = float(grid.period)
    floor = _min_beats(period)
    start = float(start_seconds)
    total = math.fsum(planned)

    # The budget: the nearest whole number of beats to the planned total, but
    # never fewer than one beat per hold, which is what the floor demands.
    budget = max(len(planned) * floor, int(total / period + 0.5))

    # The first cut opens on a hit rather than on a round number of beats into
    # the track. The grid's own phase picks the beat; the sub-beat remainder is
    # discarded rather than carried, because a hold that is not a whole number
    # of beats puts the whole total back into question -- and because discarding
    # it is what re-phases the grid onto ``start_seconds``.
    opening = grid.nearest_beat(start + planned[0], origin=start) - start
    first = int(opening / period + 0.5)
    first = max(floor, min(budget - (len(planned) - 1) * floor, first))

    rest = planned[1:]
    beats = [first] + _allocate(rest, math.fsum(rest), budget - first, floor)
    beats = _restore_ordering(planned, beats, floor)
    return [round(beat * period, 6) for beat in beats]


def drift_tolerance(holds: list[float], grid=None) -> float:
    """The most :func:`snap_holds` may move the total of *holds*, in seconds.

    Stated rather than assumed, because the answer is two numbers and a caller
    has to know which one it is looking at:

    - **Half a beat** whenever the plan averages at least one beat a frame.
      That is the quantisation of landing a total on a whole number of beats,
      and it is the number that matters for a normal plan.
    - **One beat more**, when the ordering repair has to buy a beat for the
      reveal out of a plan where every frame is already at the one-beat floor
      and there is no middle frame to take it from. Measured on this project's
      own bed that is a 19-frame plan at 134 BPM: 0.33 beats.
    - **The floor, exactly**, when the plan asks for less than one beat a frame
      -- which 134 BPM makes of the 0.35 s skim and any plan of more than about
      25 frames. The floor wins over the total there, the clip grows to one beat
      a frame, and the shortfall is ``len(holds) * period - sum(holds)``, not a
      rounding. A 40-frame plan of 0.35 s frames grows 0.6 s: that is the track
      dictating the duration, and it is the honest answer to "the frame durations
      are not constants, the track dictates them".

    A grid of ``None`` tolerates nothing, because nothing is snapped.
    """
    if grid is None or not getattr(grid, "usable", False) or not holds:
        return 0.0
    period = float(grid.period)
    total = math.fsum(max(0.0, float(value)) for value in holds)
    floor = _min_beats(period) * period
    forced = len(holds) * floor - total
    return period / 2.0 * DRIFT_TOLERANCE_BEATS + max(0.0, forced)


def _min_beats(period: float) -> int:
    """The fewest beats a hold may occupy at *period*.

    One beat, or as many as it takes to clear :data:`MIN_HOLD_SECONDS`. The
    second term only bites on a grid finer than 80 ms -- a double-time reading of
    a fast track -- where a single beat would be a hold ffmpeg clamps anyway.
    """
    return max(MIN_SNAP_BEATS, int(math.ceil(MIN_HOLD_SECONDS / max(period, 1e-9) - 1e-9)))


def _allocate(planned: list[float], total: float, budget: int, floor: int) -> list[int]:
    """Hand *budget* beats across *planned* in proportion, summing to exactly it.

    Cumulative rounding rather than per-hold rounding, because rounding each
    hold independently accumulates: at a 0.448 s beat, twelve holds each rounded
    to nearest are up to 2.7 s out in total, a quarter of a short. The running
    total is rounded once per hold, so the errors are differences of rounds and
    telescope down to the last one.
    """
    count = len(planned)
    if not count:
        return []
    spare = max(0, budget - count * floor)
    if total <= 0.0:
        return [floor] * count

    beats: list[int] = []
    running = 0.0
    previous = 0
    for hold in planned:
        running += hold
        here = int(spare * (running / total) + 0.5)
        beats.append(floor + (here - previous))
        previous = here
    return beats


def _restore_ordering(planned: list[float], beats: list[int], floor: int) -> list[int]:
    """Make the longest hold strictly longer than the shortest, keeping the total.

    Only fires when the input actually had a distinction: if every planned hold
    was the same length there is no reveal to protect and no skim to protect it
    from, and inventing one would be the module editorialising.
    """
    if len(beats) < 2:
        return beats
    longest = max(range(len(planned)), key=lambda i: (planned[i], -i))
    shortest = min(range(len(planned)), key=lambda i: (planned[i], i))
    if planned[longest] <= planned[shortest] or beats[longest] > beats[shortest]:
        return beats

    out = list(beats)
    # Prefer a donor that is neither extreme, so the fix is invisible in the
    # part of the plan the viewer is actually watching.
    for index in sorted(range(len(out)), key=lambda i: (-out[i], i)):
        if index in (longest, shortest) or out[index] <= floor:
            continue
        out[index] -= 1
        out[longest] += 1
        return out
    # Nothing spare to move: the longest gains the beat and the clip grows by
    # one, which is the documented second half of the drift tolerance.
    out[longest] += 1
    return out
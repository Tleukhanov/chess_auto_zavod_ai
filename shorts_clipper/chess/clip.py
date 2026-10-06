"""Assemble a chess short from rendered board frames.

A chess clip is a small set of still positions with a bed under them, not a
montage: there is no source footage, so every visual is a frame this module
draws. The clip therefore has three beats --

1. the position *before* the move, held
2. the move itself, arrow and accent, held
3. the position *after*, held long enough to read

-- which is also what makes it legible: a viewer needs the before-state to
understand why the move was bad, not just the after-state.

The music bed is optional and, when present, is mixed from an energetic offset
via captions.music.energetic_offset for the same reason the motivational
channel uses one: a produced track opens with an intro longer than a short.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from pathlib import Path

from shorts_clipper.chess.board import FrameSpec, render_frame

log = logging.getLogger(__name__)

FPS = 30
HOLD_BEFORE = 2.2
HOLD_MOVE = 2.4
HOLD_AFTER = 2.8
CROSSFADE = 0.4


@dataclasses.dataclass
class ChessClipPlan:
    frames: list[FrameSpec]
    before_seconds: float = HOLD_BEFORE
    move_seconds: float = HOLD_MOVE
    after_seconds: float = HOLD_AFTER

    @property
    def duration(self) -> float:
        return self.before_seconds + self.move_seconds + self.after_seconds


def plan_for_decision(decision, captions: str | None = None) -> ChessClipPlan:
    """Build the three-beat plan for one :class:`Decision`."""
    try:
        import chess
    except ImportError:
        return ChessClipPlan(frames=[])

    before = chess.Board(decision.fen_before)
    after = chess.Board(decision.fen_after)
    move = chess.Move.from_uci(decision.uci)
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)
    touched = tuple(
        chess.square_name(sq)
        for sq in chess.SQUARES
        if before.piece_at(sq) != after.piece_at(sq)
    )
    title = f"Ход {decision.move_number}… {decision.san}"

    return ChessClipPlan(
        frames=[
            FrameSpec(
                fen=decision.fen_before,
                top_text=title,
                bottom_text=captions or "смотри до хода",
                accent="neutral",
                highlight_squares=touched,
            ),
            FrameSpec(
                fen=decision.fen_before,
                top_text=title,
                bottom_text=captions or "слон встал под бой",
                accent="bad",
                highlight_squares=touched,
                arrow=(src, dst),
            ),
            FrameSpec(
                fen=decision.fen_after,
                top_text=title,
                bottom_text=decision.caption(),
                accent="bad",
                highlight_squares=touched,
            ),
        ]
    )


def plan_for_challenge(challenge, hold_question: float = 4.2, hold_answer: float = 3.0):
    """Two beats for a challenge: the question, held, then the answer.

    The question holds far longer than any other beat in this module. That is the
    point of the format -- the viewer needs time to actually find the move, and a
    two-second window would just be a reveal again.
    """
    try:
        import chess
    except ImportError:
        return ChessClipPlan(frames=[])

    try:
        chess.Board(challenge.fen)  # validates the FEN before we trust it
        move = chess.Move.from_uci(challenge.best_uci)
        src = chess.square_name(move.from_square)
        dst = chess.square_name(move.to_square)
    except Exception:
        log.debug("challenge plan could not parse its position", exc_info=True)
        return ChessClipPlan(frames=[])

    eval_line = _eval_caption(challenge)

    return ChessClipPlan(
        frames=[
            # The question frame must show the position untouched. Highlighting the
            # origin and destination -- or even the evaluation -- gives the answer
            # away, which turns the puzzle back into the reveal format. Verified by
            # looking at the rendered frame, not by reading this code.
            FrameSpec(
                fen=challenge.fen,
                top_text=challenge.question(),
                bottom_text="",
                accent="neutral",
            ),
            FrameSpec(
                fen=challenge.fen,
                top_text=challenge.question(),
                bottom_text=challenge.answer(),
                accent="good",
                highlight_squares=(src, dst),
                arrow=(src, dst),
                eval_text=eval_line,
            ),
        ],
        before_seconds=hold_question,
        move_seconds=hold_answer,
        after_seconds=0.0,
    )


def _eval_caption(challenge) -> str | None:
    """`оценка +0.6` from the mover's side, in Russian."""
    try:
        value = challenge.mover_cp / 100.0
    except Exception:
        return None
    if challenge.mate_in:
        return f"оценка: мат в {challenge.mate_in}"
    return f"оценка: {value:+.1f}"


def render_clip(
    plan: ChessClipPlan,
    out_path: str | Path,
    work_dir: str | Path | None = None,
    fps: int = FPS,
) -> Path | None:
    """Render *plan* to an mp4. ``None`` when FFmpeg is unavailable."""
    import tempfile

    from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

    if not plan.frames:
        log.warning("Chess clip has no frames")
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="chess_clip_")
        work = Path(tmp_ctx.name)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        images = []
        for i, spec in enumerate(plan.frames):
            png = render_frame(spec, work / f"frame_{i}.png")
            images.append(png)

        ffmpeg = ffmpeg_path()
        # Concat demuxer keeps the three stills as separate inputs so each can
        # hold for its own duration; a zoompan filter adds the slow Ken Burns
        # push that stops a still frame reading as a dead screenshot.
        concat_file = work / "frames.txt"
        lines = []
        for i, png in enumerate(images):
            dur = (
                plan.before_seconds if i == 0
                else plan.move_seconds if i == 1
                else plan.after_seconds
            )
            lines.append(f"file '{png.as_posix()}'")
            lines.append(f"duration {dur:.3f}")
        lines.append(f"file '{images[-1].as_posix()}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        zoom = (
            f"zoompan=z='min(zoom+0.0009,1.10)':d={fps * 2}"
            f":s=1080x1920:fps={fps}"
        )
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            # A silent stereo track is always muxed in. Chess clips are bed-driven
            # in practice, and a video-only mp4 makes the later mix fail on
            # "[0:a] matches no streams" -- which is exactly what happened.
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", f"{zoom},format=yuv420p",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-r", str(fps),
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            log.error("ffmpeg failed: %s", (proc.stderr or "")[-800:])
            return None
        log.info("Chess clip written to %s", out)
        return out
    except Exception:
        log.debug("Chess clip render failed", exc_info=True)
        return None
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def add_bed(
    clip_path: str | Path,
    music_path: str | Path | None,
    out_path: str | Path,
    start_seconds: float = 0.0,
    volume: float = 0.16,
) -> Path:
    """Mix a music bed under the clip. Copies the video when no music is given.

    The bed is duckable (``volume=0.16``) because the clip has no narration: the
    on-screen text is the message, so music must not compete with it.
    """
    import subprocess

    from shorts_clipper.utils.ffmpeg_path import ffmpeg_path

    clip_path = Path(clip_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if music_path is None or not Path(music_path).is_file():
        if clip_path == out:
            return out
        subprocess.run(
            [ffmpeg_path(), "-y", "-v", "error", "-i", str(clip_path), "-c", "copy", str(out)],
            capture_output=True, timeout=120, check=False,
        )
        # Returning a path that does not exist would leave the caller unable to
        # tell success from failure, so hand back the input instead.
        return out if out.is_file() else clip_path

    cmd = [
        ffmpeg_path(), "-y", "-v", "error",
        "-i", str(clip_path),
        "-ss", f"{float(start_seconds):.3f}", "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={volume}[bed];[0:a][bed]amix=inputs=2:duration=first:"
        "dropout_transition=0:normalize=0,alimiter=limit=0.97[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        log.error("bed mix failed: %s", (proc.stderr or "")[-800:])
        return clip_path
    return out
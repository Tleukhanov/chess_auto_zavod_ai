#!/usr/bin/env python3
"""Turn a PGN into a short vertical clip about the move that decided the game.

Usage:
    python scripts/make_chess_clip.py game.pgn
    python scripts/make_chess_clip.py game.pgn --out outputs/ --count 2
    python scripts/make_chess_clip.py game.pgn --music data/music/dark_industrial_loop.wav

Everything is drawn from the PGN: no footage, no licence questions. Exits non-zero
with a clear message when python-chess is missing, the PGN will not parse, or no
move clears the blunder threshold -- silence there would look like success.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path, help="PGN file of the game to analyse")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs", help="output directory")
    parser.add_argument("--count", type=int, default=1, help="how many moments to cut")
    parser.add_argument(
        "--critical-cp",
        type=int,
        default=200,
        help="minimum evaluation loss (centipawns) for a move to count (default: 200)",
    )
    parser.add_argument("--min-ply", type=int, default=10, help="ignore the opening (default: 10)")
    parser.add_argument(
        "--music",
        type=Path,
        default=None,
        help="music bed; defaults to dark_industrial_loop.wav under SHORTS_MUSIC_DIR",
    )
    parser.add_argument("--no-music", action="store_true", help="render without a bed")
    parser.add_argument("--seed", type=int, default=7, help="seed for the bed offset")
    args = parser.parse_args(argv)

    try:
        from shorts_clipper.chess import analysis, clip
    except ImportError as exc:
        print(f"chess module unavailable: {exc}", file=sys.stderr)
        return 2

    if not analysis.available():
        print(
            "python-chess is not installed. Install the extra: pip install -e \".[chess]\"",
            file=sys.stderr,
        )
        return 2

    game = analysis.load_pgn_file(args.pgn)
    if game is None:
        print(f"could not parse a game out of {args.pgn}", file=sys.stderr)
        return 2

    headers = dict(game.headers)
    print(f"game: {headers.get('White', '?')} vs {headers.get('Black', '?')}"
          f" ({headers.get('Result', '*')}, {headers.get('Event', '?')})")
    engine = "stockfish" if analysis.has_engine() else "material+piece-square (no engine)"
    print(f"evaluation: {engine}")

    decisions = analysis.find_decisions(
        game, critical_cp=args.critical_cp, min_ply=args.min_ply, limit=args.count
    )
    if not decisions:
        print(
            f"no move lost more than {args.critical_cp}cp after ply {args.min_ply}",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    music = None
    if not args.no_music:
        if args.music is not None:
            music = args.music
        else:
            from shorts_clipper.core.settings import Settings

            music = Path(Settings.from_env().music_dir) / "dark_industrial_loop.wav"
        if not Path(music).is_file():
            print(f"note: no music at {music}, rendering silent", file=sys.stderr)
            music = None

    written: list[Path] = []
    for i, dec in enumerate(decisions, 1):
        print(f"\n[{i}/{len(decisions)}] {dec.caption()}")
        plan = clip.plan_for_decision(dec)
        if not plan.frames:
            print("  could not build a frame plan", file=sys.stderr)
            return 2
        silent = clip.render_clip(plan, args.out / f"chess_{i}_silent.mp4")
        if silent is None:
            print("  ffmpeg render failed", file=sys.stderr)
            return 2
        offset = 0.0
        if music is not None:
            from shorts_clipper.captions.music import energetic_offset

            offset = energetic_offset(music, random.Random(args.seed + i), plan.duration)
            print(f"  bed starts at {offset:.1f}s into the track")
        final = clip.add_bed(
            silent, music, args.out / f"chess_{i}.mp4", start_seconds=offset
        )
        written.append(Path(final))
        print(f"  -> {final}")

    print(f"\ndone: {len(written)} clip(s) in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""CLI entry point for shorts-clipper.

Usage:
    python -m shorts_clipper clip <url> [options]
    python -m shorts_clipper autopilot [options]
    python -m shorts_clipper scout
    python -m shorts_clipper inspect <file.mp4>
    python -m shorts_clipper chess <game.pgn | dir-of-pgns> [options]

Examples:
    python -m shorts_clipper clip https://youtu.be/xyz --output ./clips/
    python -m shorts_clipper autopilot --log-level DEBUG
    python -m shorts_clipper scout --count 3
    python -m shorts_clipper clip --source URL1 URL2 --continue-on-error
    python -m shorts_clipper autopilot --batch-file sources.txt
    python -m shorts_clipper inspect outputs/stock_short_1.mp4
    python -m shorts_clipper chess data/pgns --count 3
    python -m shorts_clipper chess game.pgn --no-music
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shorts_clipper.core.logging import configure_logging
from shorts_clipper.core.settings import Settings


def _resolve_sources(args: argparse.Namespace) -> list[str]:
    """Explicit sources from ``--source`` / ``--batch-file`` / positional URL.

    ``--source`` wins over ``--batch-file``, which wins over the positional
    ``url``. ``--source`` value(s) may carry comma-separated URLs; the batch
    file holds one source per line with ``#`` comment lines allowed.
    Returns an empty list when no source is given.
    """
    sourced = getattr(args, "source", None)
    if sourced:
        tokens = sourced if isinstance(sourced, (list, tuple)) else [sourced]
        return [s.strip() for token in tokens for s in str(token).split(",") if s.strip()]

    batch_file = getattr(args, "batch_file", None)
    if batch_file:
        text = Path(batch_file).read_text(encoding="utf-8")
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    url = getattr(args, "url", None)
    if url:
        return [str(url).strip()]
    return []


def _cmd_batch(
    args: argparse.Namespace,
    settings: Settings,
    sources: list[str],
) -> int:
    """Run the pipeline once per explicit source and report per-source outcomes."""
    from shorts_clipper.pipeline.runner import BatchRunError, run_batch

    upload = getattr(args, "upload", False)
    count = getattr(args, "count", 1)
    niche = getattr(args, "niche", None)
    continue_on_error = getattr(args, "continue_on_error", False)

    print(f"🚀 BATCH START: {len(sources)} source(s) (fail-fast={not continue_on_error})")
    try:
        result = run_batch(
            sources,
            settings=settings,
            count=count,
            upload=upload,
            niche=niche,
            continue_on_error=continue_on_error,
        )
    except BatchRunError as exc:
        print(f"❌ BATCH ABORTED — {exc}")
        return 1

    for r in result.results:
        if r.ok:
            n = 0
            if r.outputs:
                n = len(r.outputs) if isinstance(r.outputs, list) else 1
            print(f"✅ [{r.index}/{result.total}] {r.source} — {n} clip(s)")
        else:
            print(f"❌ [{r.index}/{result.total}] {r.source} FAILED: {r.error}")

    if result.ok:
        print(f"\n🔥 BATCH SUCCESS — {result.succeeded}/{result.total} source(s) clipped.")
        return 0
    print(f"\n❌ BATCH FAILED — {result.failed}/{result.total} source(s) failed.")
    return 1


def _cmd_clip(args: argparse.Namespace, settings: Settings) -> int:
    sources = _resolve_sources(args)
    if not sources:
        print(
            "❌ No source given: pass a URL, --source URL1,URL2,..., or --batch-file file.txt",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "source", None) or getattr(args, "batch_file", None):
        if args.output:
            print("❌ --output is not supported with --source / --batch-file.", file=sys.stderr)
            return 2
        return _cmd_batch(args, settings, sources)

    from shorts_clipper.pipeline.runner import run

    url = sources[0]
    out = Path(args.output) if args.output else None
    count = getattr(args, "count", 1)
    upload = getattr(args, "upload", False)

    source_title = None
    source_channel = None
    try:
        import subprocess

        from shorts_clipper.downloader.yt_dlp import get_base_yt_dlp_cmd

        cmd = get_base_yt_dlp_cmd()
        cmd.extend(["--skip-download", "--print", "%(title)s\n%(uploader)s", "--", url])
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        lines = res.stdout.strip().split("\n")
        source_title = lines[0] if len(lines) > 0 else "YouTube Video"
        source_channel = lines[1] if len(lines) > 1 else ""
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to fetch video details via yt-dlp: %s. Using placeholders.", e
        )
        source_title = "Unknown Video"
        source_channel = "Unknown Channel"

    try:
        path_or_paths = run(
            url,
            settings=settings,
            output_path=out,
            count=count,
            upload=upload,
            source_title=source_title,
            source_channel=source_channel,
        )
        if isinstance(path_or_paths, list):
            print("\n🔥 Clips ready:")
            for p in path_or_paths:
                print(f"  - {p}")
        else:
            print(f"\n🔥 Clip ready: {path_or_paths}")
        return 0
    except Exception as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc)
        return 1


def _cmd_autopilot(args: argparse.Namespace, settings: Settings) -> int:
    sources = _resolve_sources(args)
    if sources:
        return _cmd_batch(args, settings, sources)

    from shorts_clipper.pipeline.runner import run_autopilot

    count = getattr(args, "count", 1)
    upload = getattr(args, "upload", False)
    path_or_paths = run_autopilot(
        settings=settings,
        channel=getattr(args, "channel", None),
        niche=getattr(args, "niche", None),
        keyword=getattr(args, "keyword", None),
        count=count,
        upload=upload,
    )
    if path_or_paths:
        if isinstance(path_or_paths, list):
            print("\n🔥 Clips ready:")
            for p in path_or_paths:
                print(f"  - {p}")
        else:
            print(f"\n🔥 Clip ready: {path_or_paths}")
        return 0
    print("❌ Autopilot could not find a suitable video.")
    return 1


def _cmd_scout(args: argparse.Namespace, settings: Settings) -> int:  # noqa: ARG001
    from shorts_clipper.scout.trending import get_trending_link

    count = getattr(args, "count", 1)
    found = 0
    while found < count:
        url = get_trending_link(
            channel=getattr(args, "channel", None),
            niche=getattr(args, "niche", None),
            keyword=getattr(args, "keyword", None),
            max_age_days=settings.scout_max_age_days,
        )
        if url:
            print(url)
            found += 1
        else:
            print("❌ No suitable video found.", file=sys.stderr)
            break
    return 0 if found == count else 1


def _cmd_web(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from shorts_clipper.api.server import app

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)

    print(f"\n🚀 Launching Vanguard Clipper Web Console on http://{host}:{port}...")
    try:
        uvicorn.run(app, host=host, port=port)
        return 0
    except Exception as exc:
        print(f"❌ Failed to run Vanguard server: {exc}")
        return 1


def _cmd_repair_metadata(args: argparse.Namespace, settings: Settings) -> int:
    from shorts_clipper.cli.repair_metadata import run_repair

    return run_repair()


def _cmd_inspect(args: argparse.Namespace, settings: Settings) -> int:
    from shorts_clipper.inspect import run_inspect

    return run_inspect(args, settings)


def _cmd_cleanup(args: argparse.Namespace, settings: Settings) -> int:  # noqa: ARG001
    from shorts_clipper.core.cleanup import run_cleanup

    counts = run_cleanup(settings)
    print(
        f"Cleanup done: archived={counts['archived']} deleted_old={counts['deleted_old']} "
        f"deleted_excess={counts['deleted_excess']} deleted_stray={counts['deleted_stray']}"
    )
    return 0


def _cmd_chess(args: argparse.Namespace, settings: Settings) -> int:
    from shorts_clipper.chess import batch

    if getattr(args, "clear_used", False):
        history = Path(args.used_path) if args.used_path else batch.used_path(settings)
        dropped = batch.clear_used(history)
        print(f"🧹 Cleared {dropped} chess moment(s) from {history}")
        if not getattr(args, "path", None):
            return 0

    paths = [str(p) for p in (getattr(args, "path", None) or [])]
    if not paths:
        print(
            "❌ No PGN given: pass a file or a directory "
            "(e.g. shorts-clipper chess data/pgns --count 2)",
            file=sys.stderr,
        )
        return 2

    from shorts_clipper.chess import analysis

    if not analysis.available():
        print(
            "❌ python-chess is not installed. Install the extra: pip install -e \".[chess]\"",
            file=sys.stderr,
        )
        return 2

    count = getattr(args, "count", 1)
    continue_on_error = getattr(args, "continue_on_error", False)
    music = getattr(args, "music", None)
    no_music = getattr(args, "no_music", False)
    if no_music:
        music = None

    if not no_music:
        bed = (
            Path(music)
            if music
            else Path(settings.music_dir) / batch.DEFAULT_CHESS_MUSIC
        )
        if not bed.is_file():
            print(f"⚠️  No music bed at {bed} — rendering silent.", file=sys.stderr)

    print(f"♟️  CHESS BATCH START: {len(paths)} path(s), --count {count} "
          f"(fail-fast={not continue_on_error})")
    try:
        result = batch.run_batch(
            paths,
            settings=settings,
            count=count,
            critical_cp=getattr(args, "critical_cp", None),
            min_ply=getattr(args, "min_ply", None),
            output_dir=getattr(args, "output", None),
            music=music,
            no_music=no_music,
            seed=getattr(args, "seed", batch.DEFAULT_SEED),
            continue_on_error=continue_on_error,
            used_file=getattr(args, "used_path", None),
        )
    except batch.ChessBatchError as exc:
        print(f"❌ CHESS BATCH ABORTED — {exc}")
        return 1

    for r in result.results:
        if r.ok:
            detail = f"{len(r.clips)} clip(s)"
            if r.skipped_used:
                detail += f", {r.skipped_used} already cut"
            if r.games_in_file > 1:
                detail += f" (file holds {r.games_in_file} games, used the first)"
            print(f"✅ [{r.index}/{result.total}] {r.source} — {detail}")
        else:
            print(f"❌ [{r.index}/{result.total}] {r.source} FAILED: {r.error}")

    if not result.clips:
        print(
            "❌ No clip produced: no game yielded a moment above the threshold"
            + (f" ({result.skipped_duplicate} already cut — try --clear-used)"
               if result.skipped_duplicate else ""),
            file=sys.stderr,
        )

    if result.ok and result.clips:
        print("\n🔥 CHESS BATCH SUCCESS — " + ", ".join(str(p) for p in result.clips))
        return 0
    if result.ok and not result.clips:
        print("\n❌ CHESS BATCH FAILED — nothing was clipped.")
        return 1
    print(f"\n❌ CHESS BATCH FAILED — {result.failed}/{result.total} source(s) failed.")
    return 1


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:  # noqa: ARG001
    from shorts_clipper.cli.doctor import run_doctor

    return run_doctor(settings)


def _cmd_retention_report(args: argparse.Namespace, settings: Settings) -> int:
    from shorts_clipper.cli.retention_report import run_retention_report

    return run_retention_report(settings, args)


def _cmd_revenue_report(args: argparse.Namespace, settings: Settings) -> int:
    from shorts_clipper.cli.revenue_report import run_revenue_report

    return run_revenue_report(settings, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shorts_clipper",
        description="AI-powered viral shorts clipping pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--env",
        default=".env",
        metavar="FILE",
        help="Path to .env file (default: .env)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── clip ──────────────────────────────────────────────────────────────
    clip_p = sub.add_parser("clip", help="Clip a specific YouTube video.")
    clip_p.add_argument("url", nargs="?", help="YouTube video URL.")
    clip_p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Output file path (default: outputs/clip_TIMESTAMP.mp4)",
    )
    clip_p.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="Number of viral clips to extract per source (default: 1)",
    )
    clip_p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the resulting clips to YouTube Shorts",
    )
    clip_p.add_argument(
        "--source",
        nargs="+",
        metavar="URL",
        help="Batch: one or more YouTube URLs/IDs to clip sequentially "
             "(comma-separated lists are also accepted).",
    )
    clip_p.add_argument(
        "--batch-file",
        metavar="FILE",
        help="Batch: text file with one YouTube URL/ID per line "
             "(blank lines and lines starting with '#' are ignored).",
    )
    clip_p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Batch: keep going after a failed source instead of failing fast "
             "(the overall run still exits non-zero).",
    )

    # ── autopilot ─────────────────────────────────────────────────────────
    autopilot_p = sub.add_parser(
        "autopilot",
        help="Scout a trending video and clip it automatically.",
    )
    autopilot_p.add_argument(
        "--channel",
        help="Search only this channel's recent videos.",
    )
    autopilot_p.add_argument(
        "--niche",
        help="Build 5 targeted search queries around this niche and rotate between them.",
    )
    autopilot_p.add_argument(
        "--keyword",
        help="Search specifically for this term across multiple platforms.",
    )
    autopilot_p.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="Number of viral clips to extract (default: 1)",
    )
    autopilot_p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the resulting clips to YouTube Shorts",
    )
    autopilot_p.add_argument(
        "--source",
        nargs="+",
        metavar="URL",
        help="Batch: one or more YouTube URLs/IDs to clip sequentially, "
             "skipping discovery (comma-separated lists are also accepted).",
    )
    autopilot_p.add_argument(
        "--batch-file",
        metavar="FILE",
        help="Batch: text file with one YouTube URL/ID per line "
             "(blank lines and lines starting with '#' are ignored).",
    )
    autopilot_p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Batch: keep going after a failed source instead of failing fast "
             "(the overall run still exits non-zero).",
    )

    # ── scout ─────────────────────────────────────────────────────────────
    scout_p = sub.add_parser("scout", help="Print trending video URLs and exit.")
    scout_p.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of URLs to find (default: 1)",
    )
    scout_p.add_argument(
        "--channel",
        help="Search only this channel's recent videos.",
    )
    scout_p.add_argument(
        "--niche",
        help="Build 5 targeted search queries around this niche and rotate between them.",
    )
    scout_p.add_argument(
        "--keyword",
        help="Search specifically for this term across multiple platforms.",
    )

    # ── web ───────────────────────────────────────────────────────────────
    web_p = sub.add_parser("web", help="Start the Vanguard Web Console Dashboard.")
    web_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )

    # ── inspect ─────────────────────────────────────────────────────────────
    inspect_p = sub.add_parser(
        "inspect",
        help="Print a read-only quality report for a rendered short.",
    )
    inspect_p.add_argument("file", help="Path to the rendered .mp4 file.")

    # ── repair-metadata ───────────────────────────────────────────────────────────────
    sub.add_parser("repair-metadata", help="Repair clips missing metadata.")

    # ── cleanup ────────────────────────────────────────────────────────────────────────
    sub.add_parser("cleanup", help="Archive published clips and prune old output files.")

    # ── doctor ─────────────────────────────────────────────────────────────────────────
    sub.add_parser("doctor", help="Check local environment and setup readiness.")

    # ── retention-report ──────────────────────────────────────────────────────────────────
    retention_p = sub.add_parser(
        "retention-report",
        help="Decision-science retention report per platform/niche from the metrics DB.",
    )
    retention_p.add_argument(
        "--platform",
        default=None,
        help="Only report rows for this platform (default: all).",
    )
    retention_p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback window in days for published clips (default: 30).",
    )
    retention_p.add_argument(
        "--niche",
        default=None,
        help="Only report rows for this niche (default: all).",
    )
    retention_p.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Path to the JSON report; the Markdown report is written next to it "
             "(default: outputs/retention_report.json).",
    )

    # ── chess ─────────────────────────────────────────────────────────────────
    chess_p = sub.add_parser(
        "chess",
        help="Cut chess shorts from a PGN, a directory of PGNs, or both.",
    )
    chess_p.add_argument(
        "path",
        nargs="*",
        metavar="PATH",
        help="PGN file and/or directory of .pgn files. With no path, "
             "--clear-used just resets the dedup history.",
    )
    chess_p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory (default: outputs/chess)",
    )
    chess_p.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="Moments to cut across the whole run, best N first (default: 1)",
    )
    chess_p.add_argument(
        "--critical-cp",
        type=int,
        default=None,
        help="Minimum evaluation loss in centipawns for a move to count "
             "(default: 200)",
    )
    chess_p.add_argument(
        "--min-ply",
        type=int,
        default=None,
        help="Ignore the first N plies so 'critical move' is not move 4 (default: 10)",
    )
    chess_p.add_argument(
        "--music",
        metavar="FILE",
        default=None,
        help="Music bed; defaults to dark_industrial_loop.wav under SHORTS_MUSIC_DIR",
    )
    chess_p.add_argument("--no-music", action="store_true", help="Render without a bed")
    chess_p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the bed offset (default: 7)",
    )
    chess_p.add_argument(
        "--used-path",
        metavar="FILE",
        default=None,
        help="Dedup history file (default: data/chess_used.json)",
    )
    chess_p.add_argument(
        "--clear-used",
        action="store_true",
        help="Forget every moment cut so far, so this run may re-cut them",
    )
    chess_p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Batch: keep going after a failed PGN instead of failing fast "
             "(the overall run still exits non-zero).",
    )

    # ── revenue-report ───────────────────────────────────────────────────────────
    revenue_p = sub.add_parser(
        "revenue-report",
        help="Affiliate income report (total revenue, platform/niche and per-partner "
             "breakdown) from the affiliate_events table in the metrics DB.",
    )
    revenue_p.add_argument(
        "--niche",
        default=None,
        help="Only report rows for this niche (default: all).",
    )
    revenue_p.add_argument(
        "--partner",
        default=None,
        help="Only report rows for this partner id (default: all).",
    )
    revenue_p.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Path to the Markdown report (default: outputs/revenue_report.md).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = Settings.from_env(env_path=args.env)

    dispatch = {
        "clip": _cmd_clip,
        "autopilot": _cmd_autopilot,
        "scout": _cmd_scout,
        "web": _cmd_web,
        "inspect": _cmd_inspect,
        "repair-metadata": _cmd_repair_metadata,
        "cleanup": _cmd_cleanup,
        "doctor": _cmd_doctor,
        "retention-report": _cmd_retention_report,
        "revenue-report": _cmd_revenue_report,
        "chess": _cmd_chess,
    }
    return dispatch[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())

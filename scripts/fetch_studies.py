"""Download Lichess studies as PGNs the chess factory can actually analyse.

The Lichess API serves a whole study as one PGN:

    https://lichess.org/api/study/<ID>.pgn

Three things have to be dealt with before python-chess and Stockfish will touch
it, and all three are silent failures otherwise:

1. **Variations.** Study moves carry alternatives in parentheses. They are the
   whole point of a study and the exact opposite of what an analysis feed wants:
   one linear game per file.
2. **Lichess markup.** ``{[%csl Ge5]}`` and ``{[%cal Ge5e6]}`` annotation blocks
   are not PGN comments. Left in, python-chess raises on them.
3. **Chapters that are not games.** Every study opens with a prose "Introduction"
   written as a move sequence -- ``1. e4 {In the start of the 20th century...}``.
   There is no opponent, no result and no board worth analysing. Feeding one to the
   engine produces a Decision from an essay.

Free to use: studies are other people's annotations, so they are read, never
republished. What leaves this module is a PGN per game, which is a private working
copy for analysis.

Usage::

    python scripts/fetch_studies.py eSvXUpbM daOPuS07
    python scripts/fetch_studies.py --out data/games eSvXUpbM
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://lichess.org/api/study/{study_id}.pgn"
TIMEOUT = 60

# [%csl ...], [%cal ...] and the arrows/arrows-left glyphs Lichess writes into
# comments. Removed wholesale rather than unwrapped: they are annotation, not
# narrative, and the Russian narration is written from the position anyway.
_LICHESS_TAG = re.compile(r"\[%(?:csl|cal)[^\]]*\]")
_LICHESS_GLYPH = re.compile(r"[↑↓←→⇒⇔]+")


class StudyFetchError(RuntimeError):
    """A study could not be downloaded or contained nothing usable."""


def fetch_study(study_id: str) -> str:
    """The raw PGN text of one study."""
    url = API.format(study_id=study_id.strip())
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise StudyFetchError(f"{study_id}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise StudyFetchError(f"{study_id}: {exc.reason}") from exc
    return raw.decode("utf-8", errors="replace")


def _strip_markup(text: str) -> str:
    out = _LICHESS_GLYPH.sub("", _LICHESS_TAG.sub("", text))
    return re.sub(r"[ \t]{2,}", " ", out)


def _split_games(text: str) -> list[str]:
    """Split a multi-game PGN, keeping each game's own headers."""
    blocks: list[str] = []
    current: list[str] = []
    in_moves = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            tag = stripped[1:].split(" ", 1)[0]
            if tag in ("Event", "White", "Black") and in_moves:
                blocks.append("\n".join(current))
                current = []
                in_moves = False
            current.append(stripped)
            continue
        if stripped and not stripped.startswith("["):
            in_moves = True
        current.append(stripped)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _header(block: str, name: str) -> str:
    match = re.search(rf'^\[{re.escape(name)}\s+"(.*)"\]\s*$', block, re.M)
    return match.group(1) if match else ""


def _is_real_game(block: str) -> tuple[bool, str]:
    """Whether this block is a playable game rather than a study's prose chapter."""
    white, black = _header(block, "White"), _header(block, "Black")
    result = _header(block, "Result")
    if not white or not black:
        return False, "no players"
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return False, f"result is {result or 'missing'}, not a played game"
    # A study chapter naming its own chapter name is usually a hand-typed
    # demonstration. Real games carry a Site or a Round or an ECO.
    has_provenance = any(
        _header(block, tag) for tag in ("Site", "Round", "ECO", "Opening")
    )
    if not has_provenance:
        return False, "no Site/Round/ECO"
    return True, ""


def strip_variations(text: str) -> str:
    """Drop ``(alternative)`` move variations, keeping the main line.

    A stack, not a regex: parentheses nest inside comments, and a comment can
    itself contain a bracket. Scanning char by char is the only way to know
    whether a ``)`` closes a move variation or is just prose.

    The comment state ends at the first ``}`` and nothing else. An earlier
    version tracked "have I seen a closing brace" as a one-character lookback
    that was always false, so the comment never closed and every move after the
    first comment was discarded -- a 60-ply game came out as 1 ply.
    """
    out: list[str] = []
    depth = 0
    in_comment = False
    for ch in text:
        if in_comment:
            if ch == "}":
                in_comment = False
            continue
        if ch == "{":
            in_comment = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth:
                depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def split_games(text: str) -> list[str]:
    """Playable, analysis-ready games from a study PGN."""
    cleaned = _strip_markup(text)
    kept: list[str] = []
    for block in _split_games(cleaned):
        ok, _why = _is_real_game(block)
        if not ok:
            continue
        head, _, moves = block.partition("\n\n")
        if not moves.strip():
            continue
        lines = [ln for ln in head.splitlines() if ln.strip()]
        lines += [ln for ln in moves.splitlines() if ln.strip()]
        game = strip_variations("\n".join(lines))
        # Strip the study-only tags: they confuse nothing but they are not PGN.
        game = "\n".join(
            ln for ln in game.splitlines()
            if not re.match(r"^\[(StudyName|ChapterName|ChapterURL|Annotator|UTCDate|UTCTime)\s", ln)
        )
        if game.strip():
            kept.append(game.strip() + "\n")
    return kept


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return out[:60] or "game"


def download_study(study_id: str, out_dir: Path) -> list[Path]:
    """Download one study and write one .pgn per playable game."""
    games = split_games(fetch_study(study_id))
    if not games:
        raise StudyFetchError(f"{study_id}: no playable games after cleaning")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, game in enumerate(games):
        white = _slug(_header(game, "White"))
        black = _slug(_header(game, "Black"))
        year = re.sub(r"[^0-9]", "", _header(game, "Date")[:4]) or "undated"
        path = out_dir / f"{year}_{white}-{black}_{index:03d}.pgn"
        path.write_text(game, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study_ids", nargs="+", help="Lichess study IDs, e.g. eSvXUpbM")
    ap.add_argument(
        "--out",
        default="data/games",
        help="Directory to write the per-game PGNs into (default: data/games)",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    total = 0
    failed = 0
    for study_id in args.study_ids:
        try:
            paths = download_study(study_id, out_dir)
        except StudyFetchError as exc:
            print(f"FAILED {exc}")
            failed += 1
            continue
        total += len(paths)
        print(f"{study_id}: {len(paths)} game(s) -> {out_dir}")

    print(f"\n{total} game(s) written, {failed} study/studies failed")
    return 1 if failed and not total else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
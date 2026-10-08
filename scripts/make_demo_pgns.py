"""Write a directory of demo PGNs for exercising the chess factory.

Not part of the published pipeline: these are fixtures. The player names are
deliberately fake ("Demo White", "Alpha", ...) because a rendered clip carries
them on screen, and a synthetic game filed under a real player's name is
misinformation about a real game.

Each game opens from one of the main lines so the opening format has something
to recognise, then plays out with a bias toward captures so the position actually
develops instead of shuffling.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None  # noqa: F811
    raise SystemExit(
        "python-chess is required: pip install 'shorts-clipper[chess]'"
    ) from None


OPENINGS = (
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5"),
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"),
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5"),
    ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nf6"),
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6"),
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"),
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O"),
)

# Fake names only. See the module docstring.
NAMES = (
    ("Demo White", "Demo Black"),
    ("Alpha", "Bravo"),
    ("Gamma", "Delta"),
    ("Echo", "Foxtrot"),
    ("Golf", "Hotel"),
    ("India", "Juliett"),
    ("Kilo", "Lima"),
    ("Mike", "November"),
)

LENGTHS = (40, 60, 90, 130, 170)


def write_demo_pgns(
    out_dir: str | Path,
    *,
    games: int = 14,
    seed: int = 20261008,
    capture_bias: float = 0.42,
) -> list[Path]:
    """Write *games* playable PGNs to *out_dir* and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    written: list[Path] = []
    for index in range(games):
        board = chess.Board()
        game = chess.pgn.Game()
        node: chess.pgn.GameNode = game
        plies = LENGTHS[index % len(LENGTHS)]
        played = 0

        for san in OPENINGS[index % len(OPENINGS)]:
            move = board.parse_san(san)
            node = node.add_variation(move)
            board.push(move)

        # counted by hand: a GameNode's mainline_moves() is a view with no len()
        while played < plies:
            if board.is_game_over(claim_draw=False):
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            captures = [m for m in moves if board.is_capture(m)]
            pool = captures if captures and rng.random() < capture_bias else moves
            move = rng.choice(pool)
            node = node.add_variation(move)
            board.push(move)
            played += 1

        white, black = NAMES[index % len(NAMES)]
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Event"] = "Factory demo"
        game.headers["Result"] = "*"

        path = out / f"game_{index:02d}.pgn"
        path.write_text(str(game), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="Directory to write the demo PGNs into.")
    ap.add_argument("--games", type=int, default=14, help="How many games (default: 14).")
    ap.add_argument("--seed", type=int, default=20261008, help="RNG seed (default: 20261008).")
    args = ap.parse_args(argv)

    paths = write_demo_pgns(args.out_dir, games=args.games, seed=args.seed)
    print(f"wrote {len(paths)} PGNs to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
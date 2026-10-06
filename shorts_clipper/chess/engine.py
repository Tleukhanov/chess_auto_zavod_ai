"""Real Stockfish evaluation for chess analysis.

The engine path used to be a hand-rolled UCI-over-stdin probe: one long session,
every position of the game pushed down the same pipe, ``score cp`` picked out with
a regex, a fixed depth and no notion of time. That could not tell ``score mate``
from ``score cp`` -- both matched the regex, and a mate distance in moves was
silently read as centipawns -- and one hung search lost the whole batch.

This module does it properly.

* **python-chess drives the engine.** ``chess.engine`` speaks UCI, keeps one
  process for the whole batch, and guarantees ``quit()`` on the way out, so a
  search that dies cannot leak an engine process.
* **One perspective, stated once.** Stockfish reports relative to the side to
  move and reports mates as a distance in moves, not centipawns. Every score that
  leaves this module (:class:`Eval`) is **white's point of view**, with mate
  saturated to :data:`MATE_CP` and the mate distance kept alongside in ``mate``.
  That is the same convention :func:`analysis.material_eval` already uses, which
  is what lets one piece of arithmetic cover both evaluation paths.
* **Resolution and caching.** ``SHORTS_CHESS_ENGINE`` wins, then ``stockfish`` on
  ``PATH``, then a binary downloaded into ``SHORTS_MODELS_DIR/stockfish`` -- the
  gitignored models directory, because a ~100 MB engine must never be committed.
  :func:`download_stockfish` fetches an official GitHub release, verifies the
  asset sha256 when the API publishes one, and installs only the binary.
* **Depth first, time second.** The default is a fixed ply depth with no
  wall-clock cap, which keeps two runs on the same game identical.
  ``SHORTS_CHESS_MOVETIME_MS`` exists as a ceiling for pathological positions,
  but a time limit is *weakly non-deterministic*: where the search stops depends
  on how loaded the machine is that day, so it is off by default.

Nothing here raises for a missing or broken engine: :func:`analyse` returns
``None`` and the caller falls back to the material estimate.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

ENGINE_ENV = "SHORTS_CHESS_ENGINE"
DEPTH_ENV = "SHORTS_CHESS_DEPTH"
MOVETIME_ENV = "SHORTS_CHESS_MOVETIME_MS"
THREADS_ENV = "SHORTS_CHESS_THREADS"
HASH_ENV = "SHORTS_CHESS_HASH_MB"
MODELS_DIR_ENV = "SHORTS_MODELS_DIR"

# Depth is the default limit because it is deterministic. 12 plies is roughly
# 30-100 ms per position on a laptop, so a 100-ply game costs a few seconds --
# the right order of magnitude for "a short needs an answer in seconds, not a
# minute". Raise SHORTS_CHESS_DEPTH for a final render, lower it for a draft.
DEFAULT_DEPTH = 12
# 0 means "no wall-clock cap": the search runs to the requested depth whatever it
# costs. Anything above 0 makes results weakly non-deterministic (see module
# docstring), so it is a safety valve, not the default.
DEFAULT_MOVETIME_MS = 0
# One thread for the same reason: multi-threaded search is faster but makes the
# result depend on scheduling, and a clip that re-renders differently is a bug.
DEFAULT_THREADS = 1
# A small, fixed hash keeps memory bounded and keeps results reproducible: the
# transposition table is part of the search, so a varying hash size changes them.
DEFAULT_HASH_MB = 16

# Centipawn stand-in for "mate is on the board". Large enough that no real centi-
# pawn score can reach it, small enough that captions do not print "минус 100.0"
# when a move walks into mate.
MATE_CP = 10_000

ENGINE_SUBDIR = "stockfish"
STOCKFISH_RELEASES_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
DOWNLOAD_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 15
USER_AGENT = "shorts-clipper/stockfish-fetch"
# Ordered best-first: a tuned build for this exact ISA, then the fat universal one
# that runs anywhere, then whatever plain build the release happened to ship.
_ASSET_FLAVORS = ("avx2", "universal", "")


@dataclasses.dataclass(frozen=True)
class Eval:
    """One engine score, always from white's point of view.

    ``cp`` is centipawns with mate saturated to +/-:data:`MATE_CP`, which makes it
    directly comparable with :func:`analysis.material_eval`. ``mate`` keeps the
    real distance -- moves, positive when white is mating -- because "mate in 3"
    and "mate in 12" are different blunders and the number is worth keeping.
    """

    cp: int
    mate: int | None = None
    depth: int = 0
    best_move: str | None = None

    @property
    def is_mate(self) -> bool:
        return self.mate is not None

    def for_mover(self, white_to_move: bool) -> tuple[int, int | None]:
        """``(cp, mate)`` from the side-to-move's point of view."""
        sign = 1 if white_to_move else -1
        return sign * self.cp, (None if self.mate is None else sign * self.mate)


def _chess_module():
    """python-chess with ``chess.engine`` bound, or ``None``.

    ``chess.engine`` has to be imported explicitly, the same way ``chess.pgn``
    does in :func:`analysis._chess_module`: importing the package alone leaves the
    submodule unbound.
    """
    try:
        import chess
        import chess.engine  # noqa: F401  (binds the submodule)
    except ImportError:
        return None
    return chess


def available() -> bool:
    """True when python-chess *and* its engine submodule can be imported."""
    return _chess_module() is not None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer env var, falling back on nonsense."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not a whole number, using %d", name, raw, default)
        return default
    if value < minimum:
        log.warning("%s=%d is below %d, using %d", name, value, minimum, default)
        return default
    return value


def depth(override: int | None = None) -> int:
    """Search depth in plies: the argument wins, then the env var, then the default."""
    if override is not None:
        return max(1, int(override))
    return _env_int(DEPTH_ENV, DEFAULT_DEPTH)


def movetime_ms(override: int | None = None) -> int:
    """Wall-clock cap per position in ms; ``0`` means no cap (see DEFAULT_MOVETIME_MS)."""
    if override is not None:
        return max(0, int(override))
    raw = os.getenv(MOVETIME_ENV, "").strip()
    if not raw:
        return DEFAULT_MOVETIME_MS
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not a whole number, using 0 (no cap)", MOVETIME_ENV, raw)
        return 0
    if value < 0:
        log.warning("%s=%d is negative, using 0 (no cap)", MOVETIME_ENV, value)
        return 0
    return value


def threads(override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    return _env_int(THREADS_ENV, DEFAULT_THREADS)


def hash_mb(override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    return _env_int(HASH_ENV, DEFAULT_HASH_MB)


def models_dir() -> Path:
    """Where cached engines live: ``SHORTS_MODELS_DIR``, else ``Settings.models_dir``.

    Falls back to ``models/`` if the settings module cannot be imported, because a
    chess clip should still work in a stripped-down install.
    """
    raw = os.getenv(MODELS_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        from shorts_clipper.core.settings import Settings

        return Path(Settings.from_env().models_dir).expanduser()
    except Exception:
        log.debug("Settings unavailable, defaulting the models dir", exc_info=True)
        return Path("models")


def engine_dir() -> Path:
    """``<models>/stockfish`` -- inside the gitignored models directory on purpose."""
    return models_dir() / ENGINE_SUBDIR


def installed_name(tag: str) -> str:
    """File name this module installs *tag* under."""
    suffix = ".exe" if os.name == "nt" else ""
    return f"stockfish-{tag}{suffix}"


def _looks_like_binary(path: Path) -> bool:
    """Whether *path* is a stockfish executable rather than an archive or source file.

    The official archives ship the whole source tree next to the binary, so the
    filter has to reject ``stockfish-19.zip`` and ``stockfish/src/search.cpp``.
    """
    if path.suffix.lower() in {".zip", ".gz", ".tar", ".bz2", ".xz", ".json", ".txt", ".md"}:
        return False
    if not path.name.lower().startswith("stockfish"):
        return False
    if os.name == "nt":
        return path.suffix.lower() == ".exe"
    # The POSIX release binary is a single extension-less file.
    return path.suffix == ""


def cached_binary() -> Path | None:
    """Newest engine already downloaded into the models dir, if any."""
    root = engine_dir()
    if not root.is_dir():
        return None
    found = [p for p in root.iterdir() if p.is_file() and _looks_like_binary(p)]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def binary() -> Path | None:
    """Resolve the engine: explicit path, then ``PATH``, then the models cache.

    Returns ``None`` when nothing is found. This never touches the network: a
    missing engine must not turn into an 80 MB download behind the caller's back.
    Use :func:`ensure_stockfish` for that.
    """
    configured = os.getenv(ENGINE_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        log.warning("%s=%s is not a file, looking further", ENGINE_ENV, configured)
    found = shutil.which("stockfish")
    if found:
        return Path(found)
    return cached_binary()


_PROBE_CACHE: dict[str, bool] = {}


def probe(path: Path | None = None) -> bool:
    """Whether *path* is an engine that answers UCI, probed once per path.

    Cheap on purpose: this runs on every :func:`analysis.has_engine` call, so it
    is a short subprocess rather than :func:`open_engine` (which costs ~1.4s of
    process start plus NNUE load).
    """
    target = path if path is not None else binary()
    if target is None:
        return False
    key = str(target)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = _probe_uncached(target)
    return _PROBE_CACHE[key]


def _probe_uncached(path: Path) -> bool:
    if os.name != "nt" and not os.access(path, os.X_OK):
        log.debug("%s is not executable", path)
        return False
    try:
        proc = subprocess.run(
            [str(path)],
            input="uci\nisready\nquit\n",
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except Exception:
        log.debug("engine probe failed for %s", path, exc_info=True)
        return False
    out = proc.stdout.lower()
    # "id name" is the identifying line; "uciok" proves the handshake finished.
    return proc.returncode == 0 and ("id name" in out or "uciok" in out)


def has_engine() -> bool:
    """Whether a usable engine is reachable. Probed once, then cached."""
    return probe()


def reset_cache() -> None:
    """Forget cached probe results (tests that swap binaries need this)."""
    _PROBE_CACHE.clear()


def open_engine(
    path: Path | None = None,
    threads_override: int | None = None,
    hash_override: int | None = None,
):
    """Start an engine process, configured for reproducible searches.

    Threads and hash size are pinned here rather than left to Stockfish's
    defaults: both feed the search, so leaving them to the machine is how the same
    game ends up scored differently on two runs.

    Raises if it cannot be started -- the caller decides whether that is fatal.
    :func:`analyse` treats it as "fall back to material".
    """
    chess = _chess_module()
    if chess is None:
        raise RuntimeError("python-chess is not installed")
    target = path if path is not None else binary()
    if target is None:
        raise FileNotFoundError("no stockfish binary found")
    proc = chess.engine.SimpleEngine.popen_uci(str(target), timeout=DOWNLOAD_TIMEOUT_S)
    try:
        proc.configure(
            {
                "Threads": threads(threads_override),
                "Hash": hash_mb(hash_override),
            }
        )
    except Exception:
        with contextlib.suppress(Exception):
            proc.quit()
        raise
    log.debug(
        "engine %s: threads=%d hash=%dMB",
        target,
        threads(threads_override),
        hash_mb(hash_override),
    )
    return proc


@contextlib.contextmanager
def session(**kwargs) -> Iterator[object]:
    """Context-managed engine: closed even if the body raises.

    Yields ``None`` when no engine could be started, so callers can write
    ``with session() as eng: if eng is None: ...`` without a try/except.
    """
    proc = None
    try:
        proc = open_engine(**kwargs)
    except Exception:
        log.debug("could not start an engine", exc_info=True)
        yield None
        return
    try:
        yield proc
    finally:
        with contextlib.suppress(Exception):
            proc.quit()


def limits(depth_override: int | None = None, movetime_override: int | None = None):
    """Build the ``chess.engine.Limit`` for one search.

    Depth always applies. ``time`` is only sent when a cap is configured, because
    python-chess omits unset fields and Stockfish then runs to the requested depth
    regardless of how long that takes.
    """
    chess_engine = _chess_module()
    if chess_engine is None:
        raise RuntimeError("python-chess is not installed")
    kwargs = {"depth": depth(depth_override)}
    cap_ms = movetime_ms(movetime_override)
    if cap_ms > 0:
        kwargs["time"] = cap_ms / 1000.0
    return chess_engine.engine.Limit(**kwargs)


def _eval_from_info(info: dict, chess) -> Eval:
    """Convert one python-chess analysis dict into a white-perspective :class:`Eval`.

    ``info["score"]`` is a :class:`chess.engine.PovScore`: relative to the side to
    move, with a separate ``Cp``/``Mate`` type. ``.pov(WHITE)`` fixes the
    perspective and ``score(mate_score=...)`` folds the mate distance into the same
    centipawn scale, keeping the distance itself in ``mate``.

    ``best_move`` comes from the principal variation's first move: python-chess puts
    ``bestmove`` in the result of :meth:`play`, not of :meth:`analyse`, but the head
    of the PV is the same move and costs nothing extra.
    """
    score = info["score"].pov(chess.WHITE)
    mate = score.mate()
    pv = info.get("pv") or ()
    return Eval(
        cp=int(score.score(mate_score=MATE_CP)),
        mate=None if mate is None else int(mate),
        depth=int(info.get("depth") or 0),
        best_move=pv[0].uci() if pv else None,
    )


def analyse(
    fens: Sequence[str],
    depth_override: int | None = None,
    movetime_override: int | None = None,
    path: Path | None = None,
) -> list[Eval] | None:
    """Score *fens* from white's point of view, or ``None`` if that is impossible.

    One engine process for the whole batch and one search per position. Returns
    ``None`` -- never a partial list, never an exception -- so the caller can fall
    back to the material estimate and keep :attr:`Decision.material` honest.
    """
    if not fens:
        return []
    chess = _chess_module()
    if chess is None:
        return None
    log.debug(
        "analysing %d positions at depth %d (movetime cap %dms)",
        len(fens),
        depth(depth_override),
        movetime_ms(movetime_override),
    )
    with session(path=path) as proc:
        if proc is None:
            return None
        limit = limits(depth_override, movetime_override)
        out: list[Eval] = []
        try:
            for fen in fens:
                out.append(_eval_from_info(proc.analyse(chess.Board(fen), limit), chess))
        except Exception:
            log.debug("engine analysis failed after %d positions", len(out), exc_info=True)
            return None
    return out


def mover_loss_cp(before: Eval, after: Eval, white_to_move: bool) -> int:
    """Centipawns the mover gave up by playing the move; positive means a mistake.

    Both evals are white-perspective, so they are flipped into the mover's view
    once, here, rather than being mixed up by the caller -- the bug that made the
    engine path mis-measure every loss (see ``analysis.find_decisions``).

    Mate is the part a plain subtraction gets wrong. ``score mate 0`` would read
    as zero centipawns and "threw the mate away" would read as a small loss, so:

    * still mating after the move -> never a blunder (a slower mate is a tempo,
      not a piece), so the result is clamped to at most 0;
    * already mated before and still mated -> likewise, the game was gone already;
    * walked into a mate that was not there before -> the whole value of the game,
      reported as :data:`MATE_CP` so the caption can say "mate" instead of
      printing a five-figure pawn count;
    * had a mate and no longer does -> same saturation: the win was thrown away.
    """
    before_cp, before_mate = before.for_mover(white_to_move)
    after_cp, after_mate = after.for_mover(white_to_move)
    raw = before_cp - after_cp
    if after_mate is not None and after_mate >= 0:
        return min(raw, 0)
    if after_mate is not None and before_mate is not None:
        return min(raw, 0)
    if after_mate is not None:
        return MATE_CP
    if before_mate is not None and before_mate > 0:
        return MATE_CP
    return int(raw)


# --- download and install -------------------------------------------------
#
# Explicit, never automatic: `has_engine` stays a cheap offline check, so the test
# suite and short renders cannot trigger an 80 MB fetch.


def _platform_keys() -> tuple[str, str] | None:
    """``("windows", "x86-64")``-style keys for this machine, or ``None``."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        key = "windows"
    elif system == "linux":
        key = "linux"
    elif system == "darwin":
        key = "macos"
    else:
        return None
    if machine in {"x86_64", "amd64"}:
        return key, "x86-64"
    if machine in {"arm64", "aarch64"}:
        return key, "arm64"
    return None


def pick_asset(assets: Sequence[dict]) -> dict | None:
    """Choose the release asset for this platform, best flavor first.

    Release naming has changed over the years -- ``stockfish-windows-x86-64.zip``
    back when there was one build, ``...-avx2.zip`` and now
    ``...-universal.zip`` -- and the macOS assets carry no architecture at all, so
    both forms are accepted instead of only the current one.
    """
    keys = _platform_keys()
    if keys is None:
        return None
    system, machine = keys
    candidates: list[tuple[str, dict]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        if not name.endswith((".zip", ".gz", ".tar.xz", ".tar.bz2")):
            continue
        stem = name.split(".")[0]
        for base in (f"stockfish-{system}-{machine}", f"stockfish-{system}"):
            if stem == base:
                candidates.append(("", asset))
            elif stem.startswith(base + "-"):
                candidates.append((stem[len(base) + 1 :], asset))
    for flavor in _ASSET_FLAVORS:
        for suffix, asset in candidates:
            if suffix == flavor:
                return asset
    return None


def latest_release(timeout: int = 60) -> dict:
    """The newest official Stockfish release, as the GitHub API reports it."""
    with urllib.request.urlopen(
        urllib.request.Request(STOCKFISH_RELEASES_API, headers={"User-Agent": USER_AGENT}),
        timeout=timeout,
    ) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> Iterator[zipfile.ZipInfo]:
    for info in archive.infolist():
        parts = Path(info.filename.replace("\\", "/")).parts
        if info.filename.startswith("/") or ".." in parts or Path(info.filename).is_absolute():
            log.warning("skipping suspicious archive member %s", info.filename)
            continue
        yield info


def _safe_tar_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    for info in archive.getmembers():
        name = info.name.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            log.warning("skipping suspicious archive member %s", info.name)
            continue
        yield info


def _archive_kind(archive_path: Path) -> str:
    """``"zip"`` or ``"tar"`` from the file's magic bytes.

    Sniffing rather than trusting the suffix matters: the download lands on a
    ``.part`` temporary name, and a release that ships ``.tar.xz`` would otherwise
    be handed to ``zipfile`` on the strength of its name.
    """
    with archive_path.open("rb") as handle:
        head = handle.read(6)
    if head[:2] == b"PK":
        return "zip"
    if head[:3] == b"\x1f\x8b\x08":  # gzip
        return "tar"
    if head[:6] in (b"\xfd7zXZ\x00", b"BZh"):
        return "tar"
    if archive_path.suffix.lower() == ".zip":
        return "zip"
    return "tar"


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract *archive_path* into *dest*, refusing to write outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    if _archive_kind(archive_path) == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dest, members=list(_safe_members(archive)))  # noqa: S202
        return
    with tarfile.open(archive_path, "r:*") as archive:
        try:
            archive.extractall(dest, members=list(_safe_tar_members(archive)), filter="data")  # type: ignore[call-arg]  # noqa: S202
        except TypeError:  # Python 3.11 has no extraction filters
            archive.extractall(dest, members=list(_safe_tar_members(archive)))  # noqa: S202


def _find_extracted_binary(root: Path) -> Path | None:
    """The executable inside an extracted release, shallowest first."""
    found = [p for p in root.rglob("*") if p.is_file() and _looks_like_binary(p)]
    if not found:
        return None
    return min(found, key=lambda p: (len(p.parts), p.name))


def install_archive(archive_path: Path, tag: str, dest_dir: Path | None = None) -> Path:
    """Extract *archive_path* and keep only the engine binary, under *dest_dir*.

    The release archive also carries the full C++ source tree, which would roughly
    double what the models directory holds for no benefit at runtime.
    """
    root = dest_dir if dest_dir is not None else engine_dir()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".unpack-") as tmp:
        staging = Path(tmp)
        extract_archive(archive_path, staging)
        found = _find_extracted_binary(staging)
        if found is None:
            raise FileNotFoundError(f"no stockfish binary inside {archive_path.name}")
        target = root / installed_name(tag)
        if target.exists():
            target.unlink()
        shutil.move(str(found), str(target))
        if os.name != "nt":
            target.chmod(0o755)
    return target


def download_stockfish(
    tag: str | None = None,
    url: str | None = None,
    timeout: int = DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Download an official Stockfish release into the models dir and return it.

    Either resolves the newest release through the GitHub API (*tag*/*url* both
    ``None``) or installs a specific asset (*url* given). The asset's sha256 is
    checked when the API publishes a ``digest`` (it does for current releases),
    and the download lands on a temporary name first so an interrupted transfer
    cannot be mistaken for a usable engine.

    Raises on any network or verification failure -- nothing here is faked.
    """
    root = engine_dir()
    root.mkdir(parents=True, exist_ok=True)
    digest = None
    if url is None:
        release = latest_release()
        tag = tag or str(release.get("tag_name") or "")
        asset = pick_asset(release.get("assets") or [])
        if not tag:
            raise RuntimeError("the Stockfish release has no tag name")
        if asset is None:
            raise RuntimeError(f"no Stockfish build for {platform.system()} in release {tag}")
        url = str(asset["browser_download_url"])
        digest = asset.get("digest")
    elif not tag:
        raise ValueError("installing an explicit url needs a tag to name it after")

    log.info("downloading Stockfish %s from %s", tag, url)
    handle, tmp_name = tempfile.mkstemp(dir=root, prefix=".download-", suffix=".part")
    archive = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
            timeout=timeout,
        ) as response:
            shutil.copyfileobj(response, out, length=1 << 20)
        if isinstance(digest, str) and digest.startswith("sha256:"):
            expected = digest.split(":", 1)[1].strip().lower()
            actual = _sha256(archive)
            if actual != expected:
                raise ValueError(
                    f"checksum mismatch for {url}: expected {expected}, got {actual}"
                )
        installed = install_archive(archive, str(tag), root)
    finally:
        with contextlib.suppress(OSError):
            archive.unlink()

    (root / f"stockfish-{tag}.json").write_text(
        json.dumps(
            {"tag": tag, "url": url, "sha256": digest, "installed": installed.name},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    reset_cache()
    log.info("installed %s", installed)
    return installed


def ensure_stockfish(download: bool = True) -> Path | None:
    """Return a usable engine path, downloading one if allowed and needed."""
    resolved = binary()
    if resolved is not None:
        return resolved
    if not download:
        return None
    try:
        return download_stockfish()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError, TimeoutError):
        log.warning("could not download Stockfish; continuing without an engine", exc_info=True)
        return None
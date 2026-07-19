"""Resumable acquisition of public replay JSON into an immutable raw cache."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from p0.replays.schema import FetchIndexEntry, FetchMetadata


class ReplayFetchError(RuntimeError):
    """Raised after a replay request exhausts its bounded retry budget."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes


Transport = Callable[[str, float], HttpResponse]


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    """Stable acquisition settings; changing them creates a new run plan."""

    format_id: str
    cache_dir: Path
    search_url: str = "https://replay.pokemonshowdown.com/search.json"
    replay_url_template: str = "https://replay.pokemonshowdown.com/{replay_id}.json"
    page_size: int = 50
    max_pages: int = 1
    cutoff: str | None = None
    concurrency: int = 4
    retries: int = 3
    backoff_seconds: float = 0.5
    rate_limit_per_second: float = 2.0
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not self.format_id:
            raise ValueError("ScrapeConfig.format_id must be non-empty")
        for name, value in (
            ("page_size", self.page_size),
            ("max_pages", self.max_pages),
            ("concurrency", self.concurrency),
            ("retries", self.retries),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"ScrapeConfig.{name} must be positive")
        if self.backoff_seconds < 0 or self.rate_limit_per_second < 0:
            raise ValueError("ScrapeConfig backoff and rate limit must be nonnegative")
        if self.timeout_seconds <= 0:
            raise ValueError("ScrapeConfig.timeout_seconds must be positive")
        if self.cutoff is not None:
            datetime.fromisoformat(self.cutoff.replace("Z", "+00:00"))


class _RateLimiter:
    def __init__(self, rate: float):
        self._interval = 0.0 if rate == 0 else 1.0 / rate
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)


def _default_transport(url: str, timeout: float) -> HttpResponse:
    request = Request(url, headers={"User-Agent": "p0-replay-acquirer/1"})
    with urlopen(request, timeout=timeout) as response:
        return HttpResponse(int(response.status), response.read())


def _request_with_retry(
    url: str,
    *,
    config: ScrapeConfig,
    transport: Transport,
    limiter: _RateLimiter,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[HttpResponse, int]:
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        limiter.wait()
        try:
            response = transport(url, config.timeout_seconds)
            if response.status == 429 or response.status >= 500:
                raise ReplayFetchError(f"retryable HTTP status {response.status} for {url}")
            if not 200 <= response.status < 300:
                raise ReplayFetchError(f"HTTP status {response.status} for {url}")
            return response, attempt
        except (OSError, ReplayFetchError) as exc:
            last_error = exc
            if attempt < config.retries:
                sleeper(config.backoff_seconds * (2 ** (attempt - 1)))
    raise ReplayFetchError(
        f"Replay request failed after {config.retries} attempts: {url}"
    ) from last_error


def _json_object(body: bytes, url: str) -> Mapping[str, object] | list[object]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayFetchError(f"Endpoint returned malformed JSON: {url}") from exc
    if not isinstance(value, (Mapping, list)):
        raise ReplayFetchError(f"Endpoint returned a non-container JSON value: {url}")
    return value


def _replay_id(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        value = item.get("id", item.get("replay_id"))
        return value if isinstance(value, str) and value else None
    return None


def _format_id(item: object) -> str | None:
    if not isinstance(item, Mapping):
        return None
    value = item.get("format", item.get("format_id"))
    return value if isinstance(value, str) else None


def _cutoff_value(cutoff: str | None) -> datetime | None:
    return None if cutoff is None else datetime.fromisoformat(cutoff.replace("Z", "+00:00"))


def _upload_time(item: object) -> datetime | None:
    if not isinstance(item, Mapping):
        return None
    value = item.get("uploadtime", item.get("upload_time"))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class ReplayFetcher:
    """Fetch pages and replay bodies while keeping filesystem writes resumable."""

    def __init__(self, config: ScrapeConfig, *, transport: Transport | None = None):
        self.config = config
        self.transport = transport or _default_transport
        self._limiter = _RateLimiter(config.rate_limit_per_second)

    @property
    def index_path(self) -> Path:
        return self.config.cache_dir / self.config.format_id / "index.jsonl"

    def _page_url(self, page: int) -> str:
        query = urlencode({"format": self.config.format_id, "page": page})
        return f"{self.config.search_url}?{query}"

    def discover_ids(self) -> tuple[str, ...]:
        cutoff = _cutoff_value(self.config.cutoff)
        discovered: set[str] = set()
        for page in range(1, self.config.max_pages + 1):
            response, _ = _request_with_retry(
                self._page_url(page),
                config=self.config,
                transport=self.transport,
                limiter=self._limiter,
            )
            value = _json_object(response.body, self._page_url(page))
            items = (
                value if isinstance(value, list) else value.get("replays", value.get("results", []))
            )
            if not isinstance(items, list):
                raise ReplayFetchError("Search endpoint did not return a replay list")
            ids = []
            for item in items:
                replay_id = _replay_id(item)
                if replay_id is None:
                    raise ReplayFetchError("Search endpoint returned a replay without an id")
                listed_format = _format_id(item)
                if listed_format is not None and listed_format != self.config.format_id:
                    continue
                if cutoff is not None:
                    timestamp = _upload_time(item)
                    if timestamp is not None and timestamp < cutoff:
                        continue
                ids.append(replay_id)
            discovered.update(ids)
            if len(items) < self.config.page_size:
                break
        return tuple(sorted(discovered))

    def _write_immutable(self, replay_id: str, body: bytes) -> tuple[str, int]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", replay_id):
            raise ReplayFetchError(f"Replay id contains unsafe path characters: {replay_id!r}")
        directory = self.config.cache_dir / self.config.format_id / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(body).hexdigest()
        path = directory / f"{replay_id}.json.gz"
        if path.exists():
            try:
                with gzip.open(path, "rb") as stream:
                    existing = stream.read()
            except (OSError, EOFError) as exc:
                raise ReplayFetchError(f"Existing raw replay is unreadable: {path}") from exc
            if hashlib.sha256(existing).hexdigest() != digest:
                raise ReplayFetchError(f"Immutable raw replay changed on disk: {path}")
            return digest, len(existing)
        canonical_directory = directory / ".sha256"
        canonical_directory.mkdir(exist_ok=True)
        canonical_path = canonical_directory / f"{digest}.json.gz"
        if not canonical_path.exists():
            compressed = gzip.compress(body, mtime=0)
            with tempfile.NamedTemporaryFile(
                dir=canonical_directory, prefix=f".{digest}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(compressed)
            os.replace(temporary, canonical_path)
        else:
            with gzip.open(canonical_path, "rb") as stream:
                existing = stream.read()
            if existing != body:
                raise ReplayFetchError(
                    f"Duplicate content digest changed on disk: {canonical_path}"
                )
        try:
            os.link(canonical_path, path)
        except FileExistsError:
            with gzip.open(path, "rb") as stream:
                existing = stream.read()
            if existing != body:
                raise ReplayFetchError(f"Immutable raw replay changed during acquisition: {path}")
        return digest, len(body)

    def _fetch_one(self, replay_id: str) -> FetchIndexEntry:
        url = self.config.replay_url_template.format(replay_id=replay_id)
        started = time.monotonic()
        response, attempt = _request_with_retry(
            url,
            config=self.config,
            transport=self.transport,
            limiter=self._limiter,
        )
        digest, size = self._write_immutable(replay_id, response.body)
        metadata = FetchMetadata(
            source_url=url,
            fetched_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            http_status=response.status,
            attempt=attempt,
            retry_count=attempt - 1,
            elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
        )
        self._write_fetch_metadata(replay_id, metadata)
        return FetchIndexEntry(
            replay_id=replay_id,
            format_id=self.config.format_id,
            source_url=url,
            fetched_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            http_status=response.status,
            content_sha256=digest,
            byte_size=size,
        )

    def _write_fetch_metadata(self, replay_id: str, metadata: FetchMetadata) -> None:
        directory = self.config.cache_dir / self.config.format_id / "metadata"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{replay_id}.json"
        encoded = json.dumps(metadata.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ReplayFetchError(f"Fetch metadata changed on disk: {path}")
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)

    def acquire(self, replay_ids: Iterable[str] | None = None) -> tuple[FetchIndexEntry, ...]:
        ids = tuple(sorted(set(replay_ids if replay_ids is not None else self.discover_ids())))
        existing = read_fetch_index(self.index_path)
        known = {entry.replay_id: entry for entry in existing}
        pending = tuple(replay_id for replay_id in ids if replay_id not in known)
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            fetched = tuple(executor.map(self._fetch_one, pending))
        merged = tuple(sorted((*existing, *fetched), key=lambda entry: entry.replay_id))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in merged),
            encoding="utf-8",
        )
        os.replace(temporary, self.index_path)
        return merged


def read_fetch_index(path: str | Path) -> tuple[FetchIndexEntry, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    entries: list[FetchIndexEntry] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            entry = FetchIndexEntry.from_dict(json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed fetch index line {line_number}: {path}") from exc
        if entry.replay_id in seen:
            raise ValueError(f"Duplicate replay id in fetch index: {entry.replay_id}")
        seen.add(entry.replay_id)
        entries.append(entry)
    return tuple(sorted(entries, key=lambda entry: entry.replay_id))


def load_raw_replay(path: str | Path) -> bytes:
    """Load exact response bytes from a compressed immutable cache entry."""
    try:
        with gzip.open(path, "rb") as stream:
            return stream.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"Malformed compressed raw replay: {path}") from exc


__all__ = [
    "FetchMetadata",
    "HttpResponse",
    "ReplayFetchError",
    "ReplayFetcher",
    "ScrapeConfig",
    "load_raw_replay",
    "read_fetch_index",
]

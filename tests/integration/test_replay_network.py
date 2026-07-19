"""Small explicit public-endpoint smoke test for replay acquisition and parsing."""

from __future__ import annotations

import pytest

from p0.replays.protocol import parse_replay_payload
from p0.replays.scrape import ReplayFetcher, ScrapeConfig, load_raw_replay

PINNED_PUBLIC_BO3_REPLAY = "gen9vgc2026regfbo3-2487376539-h4bvc5xjkxsu87xjhtys1iykhu27nw2pw"


@pytest.mark.integration
@pytest.mark.network
def test_pinned_public_bo3_replay_round_trips(tmp_path) -> None:
    config = ScrapeConfig(
        format_id="gen9vgc2026regfbo3",
        cache_dir=tmp_path,
        concurrency=1,
        retries=2,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config).acquire((PINNED_PUBLIC_BO3_REPLAY,))
    assert len(entries) == 1
    document = parse_replay_payload(
        load_raw_replay(
            tmp_path / config.format_id / "raw" / f"{PINNED_PUBLIC_BO3_REPLAY}.json.gz"
        ),
        replay_id=PINNED_PUBLIC_BO3_REPLAY,
        format_id=config.format_id,
    )
    assert document.metadata.format_id == config.format_id
    assert document.protocol_lines

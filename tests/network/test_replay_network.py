"""Small explicit public-endpoint smoke test for replay acquisition and parsing."""

from __future__ import annotations

import pytest

from p0.format_config import FORMAT
from p0.replays.group import validated_bo3_series
from p0.replays.protocol import parse_replay_payload
from p0.replays.scrape import ReplayFetcher, ScrapeConfig, load_raw_replay

PINNED_PUBLIC_CHAMPIONS_BO3_REPLAY = "gen9championsvgc2026regmbbo3-2653729595"


@pytest.mark.integration
@pytest.mark.network
def test_pinned_public_champions_bo3_series_round_trips(tmp_path) -> None:
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        concurrency=1,
        retries=2,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config).acquire((PINNED_PUBLIC_CHAMPIONS_BO3_REPLAY,))
    documents = tuple(
        parse_replay_payload(
            load_raw_replay(tmp_path / config.format_id / "raw" / f"{entry.replay_id}.json.gz"),
            replay_id=entry.replay_id,
            format_id=config.format_id,
        )
        for entry in entries
    )
    series = validated_bo3_series(documents)

    assert len(entries) == 2
    assert all(document.metadata.format_id == config.format_id for document in documents)
    assert [membership.game_number for membership in series[0].memberships] == [1, 2]
    assert series[0].record.is_complete

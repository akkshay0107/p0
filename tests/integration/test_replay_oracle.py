from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from poke_env.battle import DoubleBattle

from p0.model.resources import default_runtime_resources
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.contract import SHOWDOWN_COMMIT, run_pinned_showdown_oracle
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state


class TestReplayOracleIntegration:
    @pytest.mark.integration
    def test_oracle_accepts_relative_repository_root(self, monkeypatch) -> None:
        root = Path.cwd()
        monkeypatch.chdir(root.parent)

        assert run_pinned_showdown_oracle(repository_root=root.name)

    @pytest.mark.integration
    def test_battlestream_oracle_emits_real_pinned_target_protocol(self) -> None:
        protocol_values: list[str] = []
        for line in run_pinned_showdown_oracle(repository_root="."):
            if line.startswith(("|split|", "|uhtml")):
                continue
            if line.startswith("|switch|") and protocol_values:
                previous = protocol_values[-1]
                if (
                    previous.startswith("|switch|")
                    and previous.split("|", 3)[2] == line.split("|", 3)[2]
                ):
                    protocol_values[-1] = line
                    continue
            if not protocol_values or line != protocol_values[-1]:
                protocol_values.append(line)
        protocol = tuple(protocol_values)
        assert protocol
        assert protocol[0].startswith("|t:|")
        assert "|gametype|doubles" in protocol
        assert any(line.startswith("|move|") for line in protocol)
        assert any(line == "|win|Alice" for line in protocol)
        assert SHOWDOWN_COMMIT == "8282e63102fa824fd2f7472778ec09793ceb7cac"

    @pytest.mark.integration
    def test_reconstruction_matches_independent_public_oracle_after_each_line(self) -> None:
        protocol_values: list[str] = []
        for line in run_pinned_showdown_oracle(repository_root="."):
            if line.startswith(("|split|", "|uhtml")):
                continue
            if line.startswith("|switch|") and protocol_values:
                previous = protocol_values[-1]
                if (
                    previous.startswith("|switch|")
                    and previous.split("|", 3)[2] == line.split("|", 3)[2]
                ):
                    protocol_values[-1] = line
                    continue
            if not protocol_values or line != protocol_values[-1]:
                protocol_values.append(line)
        protocol = tuple(protocol_values)
        names = ("Pikachu", "Eevee", "Raichu", "Jolteon", "Vaporeon", "Flareon")
        abilities = ("Static", "Run Away", "Static", "Volt Absorb", "Water Absorb", "Flash Fire")

        def ots() -> list[dict[str, object]]:
            return [
                {"name": name, "species": name, "ability": ability, "moves": ["Protect", "Tackle"]}
                for name, ability in zip(names, abilities, strict=True)
            ]

        payload = {
            "id": "phase5-oracle",
            "formatid": "gen9championsvgc2026regmb",
            "p1": "Alice",
            "p2": "Bob",
            "uploadtime": 1_750_000_000,
            "roomid": "phase5-oracle",
            "log": "\n".join(
                [
                    "|showteam|p1|" + json.dumps(ots()),
                    "|showteam|p2|" + json.dumps(ots()),
                    *protocol,
                ]
            ),
        }
        document = parse_replay_payload(payload)
        resolved = resolve_replay_events(document, dex=None)
        assert not resolved.diagnostics
        state = reduce_replay_state(
            document.metadata.replay_id,
            document.ots,
            resolved.require_accepted(),
            dex=default_runtime_resources().dex,
        )
        snapshots = {snapshot.line_index: snapshot for snapshot in state.snapshots}
        oracle = DoubleBattle("phase5-oracle", "Alice", logging.getLogger("phase5"), gen=9)
        for line in document.protocol_lines:
            if line.parts[1] in {"", "t:", "showteam", "win", "tie", "upkeep"}:
                continue
            oracle.parse_message(list(line.parts))
            snapshot = snapshots.get(line.index)
            if snapshot is None or snapshot.turn == 0:
                continue
            for slot, live in enumerate(oracle.active_pokemon):
                projected_id = snapshot.sides[0].active[slot]
                projected = snapshot.member(projected_id) if projected_id is not None else None
                assert (live is None) == (projected is None)
                if live is None or projected is None:
                    continue
                assert live.species.casefold() == projected.displayed_species.casefold()
                assert live.status == projected.status
                assert live.current_hp_fraction == pytest.approx(projected.hp_fraction, abs=0.02)
                assert dict(live.boosts) == dict(projected.boosts)

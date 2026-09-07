"""Generate the checked-in replay protocol contract from classifiers."""

from __future__ import annotations

import json
from pathlib import Path

from p0.replays.identity import normalize_showdown_id
from p0.replays.reconstruction.classification import (
    CLASSIFICATION_REGISTRY,
    UNSUPPORTED_PREDICATES,
    UNSUPPORTED_TAGS,
)

SHOWDOWN_COMMIT = "8282e63102fa824fd2f7472778ec09793ceb7cac"

# This is deliberately explicit.  Test-name substring matching silently
# dropped most stateful protocol tags from the old contract.  Each entry names
# a real public reducer test; the validator checks that the function exists
# and exercises the reducer through its public API.
STATEFUL_TRANSITION_TESTS = {
    "-ability": "tests/unit/test_reconstruction_state.py:test_trace_source_reference_does_not_overwrite_the_target_ability",
    "-activate": "tests/unit/test_reconstruction_state.py:test_unnamed_skill_swap_uses_current_ability_components",
    "-block": "tests/unit/test_reconstruction_state.py:test_source_shaped_state_neutral_effect_events",
    "-boost": "tests/unit/test_reconstruction_state.py:test_baton_pass_transfers_boosts_but_shed_tail_only_substitute",
    "-clearallboost": "tests/unit/test_reconstruction_state.py:test_clearallboost_clears_every_active_boost",
    "-clearboost": "tests/unit/test_reconstruction_state.py:test_clearboost_clears_the_named_member_boosts",
    "-clearnegativeboost": "tests/unit/test_reconstruction_state.py:test_clearnegativeboost_preserves_positive_boosts",
    "-clearpositiveboost": "tests/unit/test_reconstruction_state.py:test_clearpositiveboost_preserves_negative_boosts",
    "-copyboost": "tests/unit/test_reconstruction_state.py:test_copyboost_copies_donor_to_argument_zero",
    "-curestatus": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-damage": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-end": "tests/unit/test_reconstruction_state.py:test_dynamic_effect_start_and_end_clear_metadata",
    "-endability": "tests/unit/test_reconstruction_state.py:test_endability_preserves_ability_and_records_gastro_acid",
    "-enditem": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-fieldactivate": "tests/unit/test_reconstruction_state.py:test_source_shaped_weather_and_field_transitions",
    "-fieldend": "tests/unit/test_reconstruction_state.py:test_source_shaped_weather_and_field_transitions",
    "-fieldstart": "tests/unit/test_reconstruction_state.py:test_source_shaped_weather_and_field_transitions",
    "-formechange": "tests/unit/test_reconstruction_state.py:test_formechange_updates_the_transient_form",
    "-heal": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-hitcount": "tests/unit/test_reconstruction_state.py:test_source_shaped_state_neutral_effect_events",
    "-immune": "tests/unit/test_reconstruction_state.py:test_source_shaped_state_neutral_effect_events",
    "-invertboost": "tests/unit/test_reconstruction_state.py:test_invertboost_inverts_each_member_boost",
    "-item": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-mega": "tests/unit/test_reconstruction_state.py:test_mega_marks_the_side_as_having_mega_evolved",
    "-mustrecharge": "tests/unit/test_reconstruction_state.py:test_recharge_cant_consumes_mustrecharge_state",
    "-setboost": "tests/unit/test_reconstruction_state.py:test_setboost_sets_the_named_stat_absolute_value",
    "-sethp": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-sideend": "tests/unit/test_reconstruction_state.py:test_sidestart_and_sideend_update_side_conditions",
    "-sidestart": "tests/unit/test_reconstruction_state.py:test_sidestart_and_sideend_update_side_conditions",
    "-singlemove": "tests/unit/test_reconstruction_state.py:test_singlemove_records_a_member_scoped_effect",
    "-singleturn": "tests/unit/test_reconstruction_state.py:test_member_singleturn_effect_expires_at_upkeep",
    "-start": "tests/unit/test_reconstruction_state.py:test_dynamic_effect_start_and_end_clear_metadata",
    "-status": "tests/unit/test_reconstruction_state.py:test_source_shaped_item_status_hp_transitions",
    "-swapboost": "tests/unit/test_reconstruction_state.py:test_swapboost_swaps_only_the_named_stat",
    "-transform": "tests/unit/test_reconstruction_state.py:test_transform_is_an_immutable_overlay_with_its_own_pp",
    "-unboost": "tests/unit/test_reconstruction_state.py:test_unboost_decreases_the_named_stat",
    "-weather": "tests/unit/test_reconstruction_state.py:test_source_shaped_weather_and_field_transitions",
    "-zbroken": "tests/unit/test_reconstruction_state.py:test_source_shaped_state_neutral_effect_events",
    "clearpoke": "tests/unit/test_reconstruction_state.py:test_initialization_events_are_state_neutral",
    "detailschange": "tests/unit/test_reconstruction_state.py:test_switch_cleanup_preserves_persistent_state_and_old_snapshots",
    "faint": "tests/unit/test_reconstruction_state.py:test_faint_cleanup_clears_status_counter",
    "player": "tests/unit/test_reconstruction_state.py:test_initialization_events_are_state_neutral",
    "poke": "tests/unit/test_reconstruction_state.py:test_initialization_events_are_state_neutral",
    "showteam": "tests/unit/test_reconstruction_state.py:test_initialization_events_are_state_neutral",
    "start": "tests/unit/test_reconstruction_state.py:test_initialization_events_are_state_neutral",
    "swap": "tests/unit/test_reconstruction_state.py:test_delayed_move_stays_with_physical_slot_after_swap",
    "teamsize": "tests/unit/test_reconstruction_state.py:test_declared_team_size_excludes_unselected_open_sheet_reserves",
}


def main() -> None:
    raw_inventory_path = (
        Path(__file__).parents[1]
        / "src/p0/replays/reconstruction/showdown_raw_emission_inventory.json"
    )
    raw_entries = json.loads(raw_inventory_path.read_text(encoding="utf-8"))["entries"]
    source_by_tag = {}
    raw_witnesses = []
    for raw in raw_entries:
        tag = raw.get("tag")
        if tag and raw.get("reachability") != "excluded":
            source_by_tag.setdefault(tag, raw)
        if raw.get("reachability") in {"reachable-potential", "reachable-resolved"}:
            raw_witnesses.append(
                {
                    "id": f"{raw['path']}:{raw['line']}:{raw['call']}",
                    "path": raw["path"],
                    "line": raw["line"],
                    "call": raw["call"],
                    "tag": raw.get("tag"),
                    "resolved_tags": raw.get("resolved_tags", []),
                    "arguments": raw.get("arguments", []),
                    "reachability": raw["reachability"],
                    "disposition": (
                        CLASSIFICATION_REGISTRY[raw["tag"]].classification.value
                        if raw.get("tag") in CLASSIFICATION_REGISTRY
                        else CLASSIFICATION_REGISTRY[raw["resolved_tags"][0]].classification.value
                        if raw.get("resolved_tags")
                        else "no_state_change"
                    ),
                    "normalized_effects": sorted(
                        {
                            normalize_showdown_id(argument.strip("'\"").split(":", 1)[-1])
                            for argument in raw.get("arguments", [])
                            if argument.startswith(
                                ("'move:", '"move:', "'ability:", '"ability:', "'item:", '"item:')
                            )
                        }
                    ),
                }
            )
    test_files = [
        Path(__file__).parents[1] / "tests/unit/test_reconstruction_events.py",
        Path(__file__).parents[1] / "tests/unit/test_reconstruction_contract.py",
        Path(__file__).parents[1] / "tests/unit/test_reconstruction_state.py",
    ]
    test_nodes = []
    for test_file in test_files:
        for line in test_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("def test_"):
                test_nodes.append(
                    f"{test_file.relative_to(Path(__file__).parents[1])}:{stripped[4 : stripped.index('(')]}"
                )
    test_node_by_name = {node.rsplit(":", 1)[-1]: node for node in test_nodes}
    missing_witness_tests = {
        test_id.rsplit(":", 1)[-1]
        for test_id in STATEFUL_TRANSITION_TESTS.values()
        if test_id.rsplit(":", 1)[-1] not in test_node_by_name
    }
    if missing_witness_tests:
        raise ValueError(f"stateful transition tests are missing: {sorted(missing_witness_tests)}")
    entries = []
    for tag, rule in sorted(CLASSIFICATION_REGISTRY.items()):
        source = source_by_tag.get(tag)
        entries.append(
            {
                "tag": tag,
                "shape": {
                    "minimum_arguments": rule.minimum_arguments,
                    "maximum_arguments": rule.maximum_arguments,
                    "required_nonempty": list(rule.required_nonempty),
                },
                "disposition": rule.classification.value,
                "source": {
                    "path": source["path"] if source else "pokemon-showdown/sim/SIM-PROTOCOL.md",
                    "line": source["line"] if source else None,
                    "review": (
                        "reachable AST emission witness"
                        if source
                        else "protocol shape documented in SIM-PROTOCOL; no direct AST emission witness"
                    ),
                },
            }
        )
    value = {
        "schema": 1,
        "showdown_commit": SHOWDOWN_COMMIT,
        "formats": ["gen9championsvgc2026regmb", "gen9championsvgc2026regmbbo3"],
        "unsupported_tags": sorted(UNSUPPORTED_TAGS),
        "unsupported_predicates": [list(item) for item in sorted(UNSUPPORTED_PREDICATES)],
        "review_status": "source_inventory_reachable_sites_resolved",
        "obligations": ["pressure_additional_target_pp", "source_shaped_stateful_transition_tests"],
        "raw_witnesses": raw_witnesses,
        "test_nodes": test_nodes,
        "stateful_witnesses": {
            tag: {
                "classifier": tag,
                "transition_tests": [test_node_by_name[test_id.rsplit(":", 1)[-1]]],
            }
            for tag, test_id in sorted(STATEFUL_TRANSITION_TESTS.items())
        },
        "entries": entries,
    }
    path = Path(__file__).parents[1] / "src/p0/replays/reconstruction/replay_protocol_contract.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

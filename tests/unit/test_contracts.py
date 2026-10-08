"""Model compatibility checks for vocabulary and dex changes."""

from p0.contracts import RuntimeContract, compare_runtime_contracts


class TestRuntimeContract:
    def test_same_size_vocabulary_permutation_is_breaking(self) -> None:
        original = RuntimeContract.from_resources({"species": {"pikachu": 1, "raichu": 2}}, {})
        remapped = RuntimeContract.from_resources({"species": {"pikachu": 2, "raichu": 1}}, {})
        assert compare_runtime_contracts(original, remapped) == "incompatible"

    def test_dex_balance_change_only_warns(self) -> None:
        original = RuntimeContract.from_resources({"moves": {"tackle": 1}}, {"power": 40})
        buffed = RuntimeContract.from_resources({"moves": {"tackle": 1}}, {"power": 50})
        assert compare_runtime_contracts(original, buffed) == "warning"

    def test_resource_object_order_does_not_break_compatibility(self) -> None:
        original = RuntimeContract.from_resources({"species": {"a": 1, "b": 2}}, {"a": 1, "b": 2})
        reordered = RuntimeContract.from_resources({"species": {"b": 2, "a": 1}}, {"b": 2, "a": 1})
        assert compare_runtime_contracts(original, reordered) == "compatible"

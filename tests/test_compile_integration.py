from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.resources import default_runtime_resources


def test_compile_policy_device_guard() -> None:
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources).to("cpu")

    # On CPU, compile_policy should return policy uncompiled (bypass CUDA guard)
    compiled_cpu = compile_policy(policy, enable=True)
    assert compiled_cpu is policy
    assert not hasattr(policy.encoder, "_orig_mod")


def test_compile_policy_state_dict_integrity() -> None:
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources)
    keys_before = set(policy.state_dict().keys())

    compiled = compile_policy(policy, enable=False)
    keys_after = set(compiled.state_dict().keys())

    assert keys_before == keys_after

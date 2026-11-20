# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.paged_batch import gather_token_steering_scale
from ignition.components.sampling import (
    AdmissionBuffers,
    DeviceRequestState,
    RequestAdmission,
)
from ignition.preparation import runtime_core_pb2 as wire
from ignition.recipes.deepseek41.steering import (
    SteeringSidecar,
    steer_attention,
    steer_shared_expert,
)


def _write_synthetic_sidecar(path: Path) -> bytes:
    generator = torch.Generator().manual_seed(1017)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(4, 37):
        raw = torch.randn(5120, generator=generator, dtype=torch.float32)
        direction = (raw / raw.norm()).to(torch.bfloat16).contiguous()
        gain = (1.0 + 0.02 * torch.randn(5120, generator=generator)).to(torch.bfloat16).contiguous()
        alpha = torch.tensor(3.0 + 0.01 * (layer - 4), dtype=torch.float32)
        prefix = f"layers.{layer}"
        tensors[f"{prefix}.alpha"] = alpha
        tensors[f"{prefix}.attn_row_gain"] = gain
        tensors[f"{prefix}.direction"] = direction
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).digest()


def test_sidecar_inventory_identity_and_validation(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "steering.safetensors"
    expected_sha256 = _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    assert steering.sha256 == expected_sha256
    assert tuple(steering.layers) == tuple(range(4, 37))
    for layer in steering.layers.values():
        assert layer.direction.shape == (5120,)
        assert layer.direction.dtype == torch.bfloat16
        assert layer.attention_gain_delta.shape == (5120,)
        assert layer.alpha > 0.0

    bad_path = tmp_path / "bad_alpha.safetensors"
    _write_synthetic_sidecar(bad_path)
    from safetensors.torch import load_file
    bad_tensors = load_file(str(bad_path))
    bad_tensors["layers.10.alpha"] = torch.tensor(0.0, dtype=torch.float32)
    save_file(bad_tensors, str(bad_path))
    with pytest.raises(ValueError, match="non-positive-alpha"):
        SteeringSidecar.load(bad_path, torch.device("cpu"))


def test_runtime_core_schema_round_trips_steering_metadata() -> None:
    expected_hex = "68" * 32
    defaults = wire.DefaultSamplingParameters(
        steering_scale=0.0,
        ranges=wire.SamplingRanges(
            steering_scale=wire.SamplingFloatRange(min=0.0, max=1.0)
        ),
    )
    decoded = wire.DefaultSamplingParameters.FromString(defaults.SerializeToString())
    assert decoded.HasField("steering_scale")
    assert decoded.ranges.HasField("steering_scale")
    contract = wire.ServingContract(steering_sidecar_sha256=expected_hex)
    decoded_contract = wire.ServingContract.FromString(contract.SerializeToString())
    assert decoded_contract.steering_sidecar_sha256 == expected_hex


def test_mixed_rows_preserve_unsteered_bits_and_steer_independently(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    layer = steering.layers[18]
    parameters = torch.zeros((8, 7), dtype=torch.float32)
    parameters[:4, 6] = torch.tensor([0.0, 1.0, 0.5, 0.0])
    row_slots = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32)
    requests = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.int64)
    positions = torch.tensor([10, 0, 1, 5, 22, -1], dtype=torch.int64)
    scale = gather_token_steering_scale(parameters, row_slots, requests, positions)
    assert scale[:, 0].tolist() == [0.0, 1.0, 1.0, 0.5, 0.0, 0.0]

    generator = torch.Generator().manual_seed(42)
    attention = torch.randn(6, 5120, dtype=torch.bfloat16, generator=generator)
    shared = torch.randn(6, 5120, dtype=torch.bfloat16, generator=generator)
    attention_out = steer_attention(attention, layer, scale)
    shared_out = steer_shared_expert(shared, layer, scale)
    for row in (0, 4, 5):
        assert torch.equal(attention_out[row], attention[row])
        assert torch.equal(shared_out[row], shared[row])
    for row in (1, 2, 3):
        assert not torch.equal(shared_out[row], shared[row])


def test_shared_expert_projection_commutes_with_tp4_sum(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    scale = torch.tensor([[1.0], [0.0], [0.75], [1.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    for layer_id in (4, 15, 24, 36):
        layer = steering.layers[layer_id]
        partials = [torch.randn(4, 5120, generator=generator) for _ in range(4)]
        sum_of_steered = sum(steer_shared_expert(partial, layer, scale) for partial in partials)
        steered_sum = steer_shared_expert(sum(partials), layer, scale)
        relative = (sum_of_steered - steered_sum).norm() / steered_sum.norm()
        assert float(relative) < 1e-6


def test_request_state_keeps_legacy_width_without_sidecar() -> None:
    unsteered = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4)
    steered = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4, steering=True)
    unsteered_staging = AdmissionBuffers.allocate("cpu", vocab_size=32, capacity=4)
    steered_staging = AdmissionBuffers.allocate(
        "cpu", vocab_size=32, capacity=4, steering=True
    )
    assert unsteered.parameters.shape == (4, 6)
    assert unsteered_staging.admission_parameters.shape == (4, 6)
    assert steered.parameters.shape == (4, 7)
    assert steered_staging.admission_parameters.shape == (4, 7)

    admission = RequestAdmission(generation=1, seed=2, steering_scale=1.0)
    steered.admit(0, admission, [1, 2])
    steered_staging.stage(0, 0, admission, [1, 2])
    assert steered.parameters[0, 6].item() == 1.0
    assert steered_staging.admission_parameters[0, 6].item() == 1.0
    with pytest.raises(ValueError, match="steering-enabled"):
        unsteered.admit(0, admission, [1, 2])


def test_metadata_pack_and_set_propagate_steering_scale_under_fullgraph_compile(
    tmp_path: Path,
) -> None:
    from ignition.components.paged_batch import Metadata

    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    layer = steering.layers[24]

    parameters = torch.zeros((4, 7), dtype=torch.float32)
    parameters[1, 6] = 0.0
    parameters[2, 6] = 1.0
    row_slot_ids = torch.tensor([1, 2], dtype=torch.int32)
    starts = torch.tensor([10, 20], dtype=torch.int64)
    lengths = torch.tensor([2, 3], dtype=torch.int32)
    states = torch.tensor([1, 2], dtype=torch.int64)
    pages = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    candidates = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int64)
    live_rows = torch.tensor([2], dtype=torch.int32)

    def packed_step(hidden: torch.Tensor) -> torch.Tensor:
        packed = Metadata.pack(
            starts,
            lengths,
            states,
            pages,
            candidates,
            live_rows,
            token_capacity=6,
            sample_count=8,
            max_query_tokens=4,
            parameters=parameters,
            row_slot_ids=row_slot_ids,
        )
        assert packed.steering_scale is not None
        return steer_shared_expert(hidden, layer, packed.steering_scale)

    compiled = torch.compile(packed_step, backend="eager", fullgraph=True)
    hidden = torch.randn(6, 5120, dtype=torch.bfloat16)
    out = compiled(hidden)
    assert torch.equal(out[0], hidden[0])
    assert torch.equal(out[1], hidden[1])
    assert not torch.equal(out[2], hidden[2])
    assert not torch.equal(out[3], hidden[3])
    assert not torch.equal(out[4], hidden[4])
    assert torch.equal(out[5], hidden[5])

    allocated = Metadata.allocate(6, 2, 2, torch.device("cpu"), samples=8, steering=True)
    allocated.set(
        starts,
        lengths,
        states,
        pages,
        candidates,
        live_rows,
        parameters=parameters,
        row_slot_ids=row_slot_ids,
    )
    assert allocated.steering_scale is not None
    assert allocated.steering_scale.squeeze(1).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared activation-steering constants for DeepSeek V4.1 decoder layers."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from safetensors.torch import load_file

from ignition.preparation.persistent import persistent_copy


@dataclass(frozen=True)
class LayerSteering:
    alpha: float
    direction: torch.Tensor
    attention_gain_delta: torch.Tensor


@dataclass(frozen=True)
class SteeringSidecar:
    layers: Mapping[int, LayerSteering]
    sha256: bytes

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "SteeringSidecar":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"steering sidecar is not a file: {path}")
        digest = hashlib.sha256(path.read_bytes()).digest()
        tensors = load_file(str(path), device="cpu")
        expected = {
            f"layers.{layer}.{name}"
            for layer in range(4, 37)
            for name in ("alpha", "attn_row_gain", "direction")
        }
        if set(tensors) != expected:
            missing = sorted(expected - set(tensors))
            extra = sorted(set(tensors) - expected)
            raise ValueError(
                f"steering sidecar tensor inventory differs from layers 4 through 36: "
                f"missing={missing}, extra={extra}"
            )
        layers = {}
        for layer in range(4, 37):
            prefix = f"layers.{layer}"
            alpha_tensor = tensors[f"{prefix}.alpha"]
            direction = tensors[f"{prefix}.direction"]
            gain = tensors[f"{prefix}.attn_row_gain"]
            if alpha_tensor.dtype != torch.float32 or alpha_tensor.shape != torch.Size([]):
                raise ValueError(f"{prefix}.alpha must be a scalar FP32 tensor")
            if direction.dtype != torch.bfloat16 or direction.shape != (5120,):
                raise ValueError(f"{prefix}.direction must be BF16[5120]")
            if gain.dtype != torch.bfloat16 or gain.shape != (5120,):
                raise ValueError(f"{prefix}.attn_row_gain must be BF16[5120]")
            alpha = float(alpha_tensor.item())
            norm = float(direction.float().norm().item())
            if (not math.isfinite(alpha) or alpha <= 0.0
                    or not bool(torch.isfinite(direction.float()).all())
                    or not bool(torch.isfinite(gain.float()).all())
                    or abs(norm - 1.0) >= 1e-3):
                raise ValueError(f"{prefix} contains non-finite, non-positive-alpha, or non-unit-norm steering values")
            layers[layer] = LayerSteering(
                alpha=alpha,
                direction=persistent_copy(direction.to(device=device, non_blocking=False).contiguous()),
                attention_gain_delta=persistent_copy(
                    (gain.float() - 1.0).to(torch.bfloat16).to(device=device, non_blocking=False).contiguous()
                ),
            )
        return cls(MappingProxyType(layers), digest)


def steer_attention(output: torch.Tensor, steering: LayerSteering,
                    token_scale: torch.Tensor) -> torch.Tensor:
    """Project and row-rescale a replicated attention output."""
    direction = steering.direction.float()
    projected = output.float()
    dot = (projected @ direction).unsqueeze(1)
    projected = projected - token_scale * steering.alpha * dot * direction.unsqueeze(0)
    projected = projected * (
        1.0 + token_scale * steering.attention_gain_delta.float().unsqueeze(0)
    )
    steered = projected.to(output.dtype)
    return torch.where(token_scale == 0.0, output, steered)


def steer_shared_expert(output: torch.Tensor, steering: LayerSteering,
                        token_scale: torch.Tensor) -> torch.Tensor:
    """Project one rank's shared-expert partial before the existing reduction."""
    direction = steering.direction.float()
    projected = output.float()
    dot = (projected @ direction).unsqueeze(1)
    steered = (
        projected - token_scale * steering.alpha * dot * direction.unsqueeze(0)
    ).to(output.dtype)
    return torch.where(token_scale == 0.0, output, steered)

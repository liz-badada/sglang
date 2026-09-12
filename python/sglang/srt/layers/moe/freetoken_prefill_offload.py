"""Pinned-host expert streaming for a disaggregated SM120 prefill worker.

The two-buffer choreography follows FreeToken's Apache-2.0 host-bank design
(commit 711325d17edcdb75a803aef4e384dd3d76ab911d).
"""

from __future__ import annotations

import torch
from torch.nn import Module, Parameter

from sglang.srt.mem_cache.pool_host.common import _cuda_host_register
from sglang.srt.mem_cache.storage.mmap import alloc_mmap

_WEIGHT_NAMES = (
    "w13_weight",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_scale_inv",
)


class FreeTokenPrefillOffloadManager:
    """Stream full expert layers through two alternating GPU buffers."""

    def __init__(self) -> None:
        self._layers: dict[int, dict[str, torch.Tensor]] = {}
        self._layer_ids: list[int] = []
        self._layer_positions: dict[int, int] = {}
        self._staging: list[dict[str, torch.Tensor]] | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._ready: list[torch.cuda.Event] = []
        self._released: list[torch.cuda.Event] = []
        self._scheduled: list[int | None] = [None, None]

    def create_weights(
        self,
        fp8_method,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Preserve loader metadata without first allocating the full layer.
        with torch.device("meta"):
            fp8_method.create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                fp4_scale_dtype=torch.float8_e8m0fnu,
                **extra_weight_attrs,
            )

        layer_id = int(layer.layer_id)
        if layer_id in self._layers:
            raise RuntimeError(f"duplicate FreeToken layer id {layer_id}")

        tensors: dict[str, torch.Tensor] = {}
        for name in _WEIGHT_NAMES:
            source = getattr(layer, name)
            target = Parameter(
                alloc_mmap(tuple(source.shape), source.dtype), requires_grad=False
            )
            target.__dict__.update(source.__dict__)
            _cuda_host_register(target)
            setattr(layer, name, target)
            tensors[name] = target

        self._layers[layer_id] = tensors
        self._layer_ids = sorted(self._layers)
        self._layer_positions = {
            current: index for index, current in enumerate(self._layer_ids)
        }

    def initialize_staging(self, layer_id: int) -> None:
        if self._staging is not None:
            return
        device = torch.device("cuda", torch.cuda.current_device())
        example = self._layers[layer_id]
        self._staging = [
            {
                name: torch.empty_like(tensor, device=device)
                for name, tensor in example.items()
            }
            for _ in range(2)
        ]
        self._copy_stream = torch.cuda.Stream(device=device)
        self._ready = [torch.cuda.Event() for _ in range(2)]
        self._released = [torch.cuda.Event() for _ in range(2)]

    def _slot(self, layer_id: int) -> int:
        return self._layer_positions[layer_id] & 1

    def _prefetch(self, layer_id: int) -> None:
        assert self._staging is not None and self._copy_stream is not None
        slot = self._slot(layer_id)
        with torch.cuda.stream(self._copy_stream):
            if self._scheduled[slot] is not None:
                self._copy_stream.wait_event(self._released[slot])
            for name in _WEIGHT_NAMES:
                self._staging[slot][name].copy_(
                    self._layers[layer_id][name], non_blocking=True
                )
            self._ready[slot].record(self._copy_stream)
        self._scheduled[slot] = layer_id

    def acquire(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        position = self._layer_positions.get(layer_id)
        if position is None:
            raise RuntimeError(f"unregistered FreeToken layer {layer_id}")
        if self._staging is None:
            raise RuntimeError("FreeToken staging buffers were not initialized")
        if position == 0:
            self._prefetch(layer_id)
        elif self._scheduled[self._slot(layer_id)] != layer_id:
            raise RuntimeError(
                "FreeToken requires sequential MoE layers without batch overlap"
            )

        slot = self._slot(layer_id)
        torch.cuda.current_stream().wait_event(self._ready[slot])
        if position + 1 < len(self._layer_ids):
            self._prefetch(self._layer_ids[position + 1])
        assert self._staging is not None
        return tuple(self._staging[slot][name] for name in _WEIGHT_NAMES)

    def release(self, layer_id: int) -> None:
        slot = self._slot(layer_id)
        self._released[slot].record(torch.cuda.current_stream())


_MANAGER = FreeTokenPrefillOffloadManager()


def get_freetoken_prefill_offload_manager() -> FreeTokenPrefillOffloadManager:
    return _MANAGER

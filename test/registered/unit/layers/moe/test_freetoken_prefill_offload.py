import pytest
import torch

import sglang.srt.layers.moe.freetoken_prefill_offload as offload
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class FakeFp8Method:
    def create_weights(self, layer, *args, **kwargs):
        assert torch.empty(1).device.type == "meta"
        for name in offload._WEIGHT_NAMES:
            parameter = torch.nn.Parameter(
                torch.empty((2, 4), dtype=torch.int8), requires_grad=False
            )
            parameter.weight_loader = name
            layer.register_parameter(name, parameter)


def create_layer(manager, layer_id):
    layer = torch.nn.Module()
    layer.layer_id = layer_id
    manager.create_weights(FakeFp8Method(), layer, 2, 128, 128, torch.bfloat16)
    return layer


def test_create_weights_preserves_loader_metadata(monkeypatch):
    monkeypatch.setattr(
        offload, "alloc_mmap", lambda shape, dtype: torch.empty(shape, dtype=dtype)
    )
    monkeypatch.setattr(offload, "_cuda_host_register", lambda _: None)
    manager = offload.FreeTokenPrefillOffloadManager()
    layer = create_layer(manager, 7)

    assert manager._layer_ids == [7]
    for name in offload._WEIGHT_NAMES:
        parameter = getattr(layer, name)
        assert parameter.device.type == "cpu"
        assert parameter.weight_loader == name


def test_double_buffer_streams_layers_in_order(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    monkeypatch.setattr(offload, "_cuda_host_register", lambda _: None)
    manager = offload.FreeTokenPrefillOffloadManager()
    layers = []
    for layer_id, value in ((3, 17), (7, 29)):
        layer = create_layer(manager, layer_id)
        for name in offload._WEIGHT_NAMES:
            getattr(layer, name).data.fill_(value)
        layers.append(layer)

    manager.initialize_staging(layers[0].layer_id)
    for layer, expected in zip(layers, (17, 29), strict=True):
        tensors = manager.acquire(layer.layer_id)
        torch.cuda.current_stream().synchronize()
        assert all(torch.all(tensor == expected) for tensor in tensors)
        manager.release(layer.layer_id)

    torch.cuda.synchronize()

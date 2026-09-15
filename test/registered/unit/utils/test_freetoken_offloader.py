import torch

from sglang.srt.utils.offloader import (
    _hook_module_forward_for_offloader_before_forward,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Module(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward(self, value):
        self.events.append("forward")
        return value * self.weight


class _Offloader:
    def __init__(self, index, events):
        self.index = index
        self.events = events

    def wait_and_get_device_tensors(self):
        self.events.append(f"wait:{self.index}")
        return {"weight": torch.tensor(float(self.index + 2))}

    def start_onload(self):
        self.events.append(f"start:{self.index}")

    def offload(self):
        self.events.append(f"offload:{self.index}")


def test_freetoken_prefetches_before_forward_and_wraps_after_last_layer():
    events = []
    offloaders = [_Offloader(index, events) for index in range(3)]

    first = _Module(events)
    _hook_module_forward_for_offloader_before_forward(
        0, first, offloaders, prefetch_step=1
    )
    assert first(torch.tensor(2.0)) == 4
    assert events == ["wait:0", "start:1", "forward", "offload:0"]

    events.clear()
    last = _Module(events)
    _hook_module_forward_for_offloader_before_forward(
        2, last, offloaders, prefetch_step=1
    )
    assert last(torch.tensor(2.0)) == 8
    assert events == ["wait:2", "forward", "start:0", "offload:2"]

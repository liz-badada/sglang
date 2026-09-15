from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _request(tokens):
    return SimpleNamespace(
        origin_input_ids=[0] * tokens,
        output_ids=[],
        prefix_indices=[],
        pending_bootstrap=False,
        finished_reason=None,
        to_finish=None,
    )


def _scheduler(requests):
    return SimpleNamespace(
        waiting_queue=requests,
        chunked_req=None,
        page_size=256,
        max_prefill_tokens=35072,
        max_running_requests=128,
    )


@pytest.mark.parametrize(
    ("tokens", "should_delay", "order"),
    [
        ([16384], True, None),
        ([4096, 14336, 16384], False, [0, 2, 1]),
        ([4096, 20480, 16384, 14336], True, None),
        ([62000], False, None),
    ],
)
def test_token_aware_admission(monkeypatch, tokens, should_delay, order):
    requests = [_request(value) for value in tokens]
    scheduler = _scheduler(requests)
    monkeypatch.setattr("sglang.srt.disaggregation.prefill.time.monotonic", lambda: 1.0)
    with (
        envs.SGLANG_FREETOKEN_PREFILL_OFFLOAD.override(True),
        envs.SGLANG_FREETOKEN_PREFILL_TARGET_TOKENS.override(32768),
        envs.SGLANG_FREETOKEN_PREFILL_MAX_WAIT_MS.override(1000),
    ):
        assert (
            SchedulerDisaggregationPrefillMixin.maybe_delay_freetoken_prefill(scheduler)
            is should_delay
        )
    if order is not None:
        assert scheduler.waiting_queue == [requests[index] for index in order]


def test_admission_releases_at_deadline(monkeypatch):
    request = _request(16384)
    scheduler = _scheduler([request])
    times = iter((1.0, 2.1))
    monkeypatch.setattr(
        "sglang.srt.disaggregation.prefill.time.monotonic", lambda: next(times)
    )
    with (
        envs.SGLANG_FREETOKEN_PREFILL_OFFLOAD.override(True),
        envs.SGLANG_FREETOKEN_PREFILL_TARGET_TOKENS.override(32768),
        envs.SGLANG_FREETOKEN_PREFILL_MAX_WAIT_MS.override(1000),
    ):
        assert SchedulerDisaggregationPrefillMixin.maybe_delay_freetoken_prefill(
            scheduler
        )
        assert not SchedulerDisaggregationPrefillMixin.maybe_delay_freetoken_prefill(
            scheduler
        )

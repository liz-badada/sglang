"""SM120 core-attention routing contract for the two DSv4 Triton switches.

``SGLANG_DSV4_TRITON_DECODE`` must not move prefill: on SM120 an extend batch
that fails the sparse-prefill guard has to reach FlashInfer, not the Triton
paged-fp8 entry. Every route returns the same values, so the contract is
checked at dispatch level -- each route raises a sentinel and the test asserts
which one a given (switch, forward mode) pair reaches.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    _populate_swa_kv_cache,
    build_dsv4_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-large")

SPARSE_PREFILL = "sparse_prefill"
TRITON_PAGED = "triton_paged"
FLASHINFER = "flashinfer"


class _RouteTaken(Exception):
    """Raised in place of each kernel so the route is observable."""

    def __init__(self, route: str):
        super().__init__(route)
        self.route = route


_EXTEND_CASE = DSV4AttentionCase(
    name="route_extend",
    backend="dsv4",
    forward_mode=ForwardMode.EXTEND,
    num_heads=64,
    page_size=DSV4_PAGE_SIZE,
    prefix_lens=(64,),
    extend_lens=(16,),
)
_DECODE_CASE = DSV4AttentionCase(
    name="route_decode",
    backend="dsv4",
    forward_mode=ForwardMode.DECODE,
    num_heads=64,
    page_size=DSV4_PAGE_SIZE,
    prefix_lens=(64,),
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4Sm120CoreAttnRouting(CustomTestCase):
    """Which kernel each (SPARSE_PREFILL, TRITON_DECODE, mode) triple reaches.

    `_is_sm120` is forced on so the contract is checked on any CUDA device;
    every route is mocked, so no SM120 kernel is actually launched.
    """

    def _route_for(self, case, *, sparse_prefill: bool, triton_decode: bool) -> str:
        fixture = build_dsv4_attention_fixture(self, case)
        _populate_swa_kv_cache(
            fixture,
            max_context_len=fixture.runner.req_to_token_pool.req_to_token.shape[1],
            device="cuda",
        )
        q_input, _ = fixture.actual_module.project(fixture.input_hidden)

        def _raise(route):
            def _fn(*args, **kwargs):
                raise _RouteTaken(route)

            return _fn

        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(False),
            mock.patch.object(dsv4_backend, "_is_sm120", True),
            mock.patch.object(
                dsv4_backend, "_dsv4_triton_sparse_prefill", sparse_prefill
            ),
            mock.patch.object(dsv4_backend, "_dsv4_triton_decode", triton_decode),
            mock.patch.object(
                DeepseekV4AttnBackend,
                "_forward_prefill_sparse",
                _raise(SPARSE_PREFILL),
            ),
            mock.patch(
                "sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill"
                ".sparse_mla_prefill_paged_fp8_native",
                _raise(TRITON_PAGED),
            ),
            mock.patch(
                "sglang.kernels.ops.attention.flash_mla_sm120"
                ".flash_mla_with_kvcache_sm120",
                _raise(FLASHINFER),
            ),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            with self.assertRaises(_RouteTaken) as caught:
                fixture.backend.forward(
                    q=q_input,
                    k=q_input,
                    v=q_input,
                    layer=fixture.actual_module.attn,
                    forward_batch=fixture.forward_batch,
                    compress_ratio=0,
                    save_kv_cache=False,
                    attn_sink=fixture.actual_module.attn_sink,
                )
        return caught.exception.route

    def test_prefill_routes(self):
        # (sparse_prefill, triton_decode) -> route an EXTEND batch must take.
        #
        # The (False, True) row is the regression: before the fix it reached
        # TRITON_PAGED, which is how the decode switch came to move prefill.
        expected = {
            (False, False): FLASHINFER,
            (True, False): SPARSE_PREFILL,
            (False, True): FLASHINFER,
            (True, True): SPARSE_PREFILL,
        }
        for (sparse_prefill, triton_decode), route in expected.items():
            with self.subTest(sparse_prefill=sparse_prefill, decode=triton_decode):
                self.assertEqual(
                    self._route_for(
                        _EXTEND_CASE,
                        sparse_prefill=sparse_prefill,
                        triton_decode=triton_decode,
                    ),
                    route,
                )

    def test_decode_routes(self):
        # Decode follows TRITON_DECODE alone and never reaches the sparse
        # prefill route, whatever the prefill switch says.
        expected = {
            (False, False): FLASHINFER,
            (True, False): FLASHINFER,
            (False, True): TRITON_PAGED,
            (True, True): TRITON_PAGED,
        }
        for (sparse_prefill, triton_decode), route in expected.items():
            with self.subTest(sparse_prefill=sparse_prefill, decode=triton_decode):
                self.assertEqual(
                    self._route_for(
                        _DECODE_CASE,
                        sparse_prefill=sparse_prefill,
                        triton_decode=triton_decode,
                    ),
                    route,
                )


if __name__ == "__main__":
    unittest.main()

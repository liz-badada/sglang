import torch

from vllm.distributed import (
            tensor_model_parallel_all_reduce,
            )

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.models.deepseek_v2 import DeepseekV2Model


def forward_qwen_model_layer_print(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch, input_embeds: torch.Tensor = None,) -> torch.Tensor:
    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    residual = None
    for i in range(len(self.layers)):
        print(f"[Qwen]: Layer_{i}")
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions, hidden_states, forward_batch, residual
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def forward_deepseek_model_layer_print(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,) -> torch.Tensor:
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    for i in range(len(self.layers)):
        print(f"[DeepSeek]: Layer_{i}")
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
    if not forward_batch.forward_mode.is_idle():
        hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def moe_select_experts_tracker(func):
    def wrapper(*args, **kwargs):
        topk_weights, topk_ids = func(*args, **kwargs)
        print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}")
        print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}")
        return topk_weights, topk_ids
    return wrapper
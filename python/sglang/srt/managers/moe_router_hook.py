import torch
from vllm.distributed import (
            tensor_model_parallel_all_reduce,
            )
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2Model, DeepseekV2MoE


def forward_deepseek_model_layer_print(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,) -> torch.Tensor:
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    for i in range(len(self.layers)):
        print(f"[DeepSeek MoE Router Analysis]: Layer_{i}")
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
    if not forward_batch.forward_mode.is_idle():
        hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states

def forward_deepseek_moe_router_analysis(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    print(f"[DeepSeek MoE Router Analysis]: Hidden States Shape {num_tokens, hidden_dim}")
    hidden_states = hidden_states.view(-1, hidden_dim)
    if self.n_shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        print(f"[DeepSeek MoE Router Analysis]: Router Logits {router_logits}")
        final_hidden_states = (self.experts(hidden_states=hidden_states, router_logits=router_logits) * self.routed_scaling_factor)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_dim)

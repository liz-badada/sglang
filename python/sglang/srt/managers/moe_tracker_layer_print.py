import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.models.deepseek_v2 import DeepseekV2Model
from sglang.srt.managers import moe_tracker_router_hook


def forward_qwen_model_layer_print(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch, input_embeds: torch.Tensor = None,) -> torch.Tensor:
    moe_tracker_router_hook.moe_tracker_log = 'qwen_moe_tracker_log.txt'

    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    residual = None
    for i in range(len(self.layers)):
        with open(moe_tracker_router_hook.moe_tracker_log, 'a') as file:
            print(f"[Qwen]: Layer_{i}", file=file)
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions, hidden_states, forward_batch, residual
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def forward_deepseek_model_layer_print(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,) -> torch.Tensor:
    moe_tracker_router_hook.moe_tracker_log = 'deepseek_moe_tracker_log.txt'
    
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    for i in range(len(self.layers)):
        with open(moe_tracker_router_hook.moe_tracker_log, 'a') as file:
            print(f"[DeepSeek]: Layer_{i}", file=file)
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
    if not forward_batch.forward_mode.is_idle():
        hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers import moe_tracker_router_hook


# https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json
def forward_qwen_model_layer_hook(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch, input_embeds: torch.Tensor = None,) -> torch.Tensor:
    moe_tracker_router_hook.moe_tracker_model = 'qwen1.5-moe'
    moe_tracker_router_hook.moe_tracker_log = 'qwen_moe_tracker_log.txt'
    moe_tracker_router_hook.moe_tracker_num_experts = 60 # Currently manually specified

    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    residual = None
    for i in range(len(self.layers)):
        # print(f"[Qwen]: Layer_{i}")
        moe_tracker_router_hook.moe_tracker_layer_id = i
        if i not in moe_tracker_router_hook.moe_tracker_dict:
            moe_tracker_router_hook.moe_tracker_dict[i] = [0] * moe_tracker_router_hook.moe_tracker_num_experts
        # with open(moe_tracker_router_hook.moe_tracker_log, 'a') as file:
        #     print(f"[Qwen]: Layer_{i}", file=file)
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions, hidden_states, forward_batch, residual
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
def forward_deepseek_model_layer_hook(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,) -> torch.Tensor:
    moe_tracker_router_hook.moe_tracker_model = 'deepseek-v3'
    moe_tracker_router_hook.moe_tracker_log = 'deepseek_moe_tracker_log.txt'
    moe_tracker_router_hook.moe_tracker_num_experts = 256 # Currently manually specified

    hidden_states = self.embed_tokens(input_ids)
    residual = None
    for i in range(len(self.layers)):
        # print(f"[DeepSeek]: Layer_{i}")
        moe_tracker_router_hook.moe_tracker_layer_id = i
        if i not in moe_tracker_router_hook.moe_tracker_dict:
            moe_tracker_router_hook.moe_tracker_dict[i] = [0] * moe_tracker_router_hook.moe_tracker_num_experts
        # with open(moe_tracker_router_hook.moe_tracker_log, 'a') as file:
        #     print(f"[DeepSeek]: Layer_{i}", file=file)
        layer = self.layers[i]
        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
    if not forward_batch.forward_mode.is_idle():
        hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
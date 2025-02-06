from sglang.srt.managers import moe_tracker_layer_print


def moe_select_experts_tracker(func):
    def wrapper(*args, **kwargs):
        topk_weights, topk_ids = func(*args, **kwargs)

        moe_tracker_log = moe_tracker_layer_print.moe_tracker_log
        with open(moe_tracker_log, 'a') as file:
            print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}")
            print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}")
        return topk_weights, topk_ids
    return wrapper
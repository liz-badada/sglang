moe_tracker_log = 'moe_tracker_log.txt'
moe_tracker_layer_id = 0
moe_tracker_dict = {}


def moe_select_experts_tracker(func):
    global moe_tracker_layer_id
    global moe_tracker_dict
    def wrapper(*args, **kwargs):
        topk_weights, topk_ids = func(*args, **kwargs)

        print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}")
        # print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}")

        if moe_tracker_layer_id not in moe_tracker_dict:
            raise ValueError(f"Layer ID {moe_tracker_layer_id} not initialized in layer_dict.")
    
        # 遍历 topk_ids 对应的元素累加
        for idx in topk_ids:
            if 0 <= idx < len(moe_tracker_dict[moe_tracker_layer_id]):
                moe_tracker_dict[moe_tracker_layer_id][idx] += 1
            else:
                raise IndexError(f"TopK ID {idx} is out of the valid range for given num_experts.")

        # global moe_tracker_log
        # with open(moe_tracker_log, 'a') as file:
        #     print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}", file=file)
        #     print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}", file=file)
        return topk_weights, topk_ids
    return wrapper


def moe_tracker_analysis():
    global moe_tracker_dict
    for layer_id, stats in moe_tracker_dict.items():
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(stats))),
            y=stats,
            marker_color='indigo'
        ))

        fig.update_layout(
            title=f"Layer {layer_id} Expert Selection",
            xaxis_title="Expert ID",
            yaxis_title="Selection Count",
            xaxis=dict(dtick=1)
        )

        file_name = f"layer_{layer_id}_expert_selection.png"
        fig.write_image(file_name)
        print(f"Saved {file_name}")
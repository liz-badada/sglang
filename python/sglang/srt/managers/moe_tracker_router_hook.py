import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


moe_tracker_model = ''
moe_tracker_log = 'moe_tracker_log.txt'
moe_tracker_num_experts = 0
moe_tracker_layer_id = 0
moe_tracker_dict = {}


def moe_select_experts_tracker(func):
    global moe_tracker_num_experts
    global moe_tracker_layer_id
    global moe_tracker_dict
    def wrapper(*args, **kwargs):
        topk_weights, topk_ids = func(*args, **kwargs)
        flattened_topk_ids = topk_ids.flatten()

        # print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}")
        # print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}")

        if moe_tracker_layer_id not in moe_tracker_dict:
            raise ValueError(f"Layer ID {moe_tracker_layer_id} not initialized in layer_dict.")
    
        # 遍历 topk_ids 对应的元素累加
        for _, expert_idx in enumerate(flattened_topk_ids):
            # print(expert_idx)
            assert expert_idx < moe_tracker_num_experts, f"TopK ID {expert_idx} is out of the valid range for given num_experts."
            moe_tracker_dict[moe_tracker_layer_id][expert_idx] += 1
            # print(f"experts: {moe_tracker_num_experts}, layer_id{moe_tracker_layer_id}, expert_id{expert_idx}, count{moe_tracker_dict[moe_tracker_layer_id][expert_idx]}")

        # global moe_tracker_log
        # with open(moe_tracker_log, 'a') as file:
        #     print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}", file=file)
        #     print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}", file=file)
        return topk_weights, topk_ids
    return wrapper


def moe_tracker_analysis():
    global moe_tracker_model
    global moe_tracker_num_experts
    global moe_tracker_dict

    num_layers = len(moe_tracker_dict.keys())
    if num_layers != 0:
        fig = make_subplots(rows=num_layers, cols=1, subplot_titles=[f"Layer {layer_id} Expert Selection" for layer_id in moe_tracker_dict.keys()])

        for layer_idx, (layer_id, stats) in enumerate(moe_tracker_dict.items(), start=1):
            print(f"experts: {moe_tracker_num_experts}, layer_id{layer_id}, expert_stats{stats}")

            fig.add_trace(
                go.Bar(
                    x=list(range(len(stats))),
                    y=stats,
                    marker_color='indigo',
                    name=f"Layer {layer_id}"
                ),
                row=layer_idx,
                col=1
            )

        fig.update_layout(
            height=300 * num_layers,
            title=f"MoE Tracker Analysis by Layer (Model: {moe_tracker_model})",
            showlegend=False,
        )

        fig.update_xaxes(title_text="Expert ID", dtick=1)
        fig.update_yaxes(title_text="Selection Count")

        file_dir = f"./moe_tracker_stats/{moe_tracker_model}"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = f"{file_dir}/all_layers_expert_selection.html"
        
        fig.write_html(file_name)
        print(f"Saved {file_name}")


'''
def moe_tracker_analysis():
    global moe_tracker_model
    global moe_tracker_num_experts
    global moe_tracker_dict
    for layer_id, stats in moe_tracker_dict.items():
        print(f"experts: {moe_tracker_num_experts}, layer_id{layer_id}, expert_stats{stats}")

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

        file_dir = f"./moe_tracker_stats/{moe_tracker_model}"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = f"{file_dir}/layer_{layer_id}_expert_selection.html"
        # fig.write_image(file_name)
        fig.write_html(file_name)
        print(f"Saved {file_name}")
'''
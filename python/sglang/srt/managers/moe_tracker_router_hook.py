import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


moe_tracker_model = ''
moe_tracker_log = 'moe_tracker_log.txt'
moe_tracker_num_experts = 0
moe_tracker_layer_id = 0
moe_tracker_dict = {'prefill': {}, 'decode': {}}
moe_tracker_stage = ''


def moe_select_experts_tracker(func):
    global moe_tracker_num_experts
    global moe_tracker_layer_id
    global moe_tracker_dict
    global moe_tracker_stage
    def wrapper(*args, **kwargs):
        topk_weights, topk_ids = func(*args, **kwargs)
        flattened_topk_ids = topk_ids.flatten()

        # print(f"[MoE Router Topk]: weights shape {topk_weights.shape}, ids shape {topk_ids.shape}")
        # print(f"[MoE Router TopK]: weights {topk_weights}, ids {topk_ids}")

        if moe_tracker_stage == '':
            raise ValueError(f"Running stage {moe_tracker_stage} unknown.")
        if moe_tracker_layer_id not in moe_tracker_dict[moe_tracker_stage]:
            raise ValueError(f"Layer ID {moe_tracker_layer_id} not initialized in layer_dict.")

        for _, expert_idx in enumerate(flattened_topk_ids):
            # print(expert_idx)
            assert expert_idx < moe_tracker_num_experts, f"TopK ID {expert_idx} is out of the valid range for given num_experts."
            moe_tracker_dict[moe_tracker_stage][moe_tracker_layer_id][expert_idx] += 1
            # print(f"experts: {moe_tracker_num_experts}, layer_id{moe_tracker_layer_id}, expert_id{expert_idx}, count{moe_tracker_dict[moe_tracker_stage][moe_tracker_layer_id][expert_idx]}")

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
    global moe_tracker_stage

    file_dir = f"./moe_tracker_stats/{moe_tracker_model}"

    num_layers = len(moe_tracker_dict[moe_tracker_stage].keys())
    if num_layers != 0:
        # visualize
        fig = make_subplots(rows=num_layers, cols=1, subplot_titles=[f"Layer {layer_id} Expert Selection" for layer_id in moe_tracker_dict[moe_tracker_stage].keys()])

        for layer_idx, (layer_id, stats) in enumerate(moe_tracker_dict[moe_tracker_stage].items(), start=1):
            # print(f"experts: {moe_tracker_num_experts}, layer_id{layer_id}, expert_stats{stats}")

            total_count = sum(stats)
            # relative_stats = [count / total_count * 100 for count in stats] if total_count > 0 else [0] * len(stats)
            fig['layout']['annotations'][layer_id]['text'] = f"Layer {layer_id} Expert Selection, Total Tokens to be processed {total_count}"

            fig.add_trace(
                go.Bar(
                    x=list(range(len(stats))),
                    y=stats,
                    marker=dict(
                        color=stats,
                        colorscale='viridis',
                        colorbar=dict(title="Selection Count")
                    ),
                    text=[f"{count/(total_count+1e-9):.2f}%" for count in stats],
                    textposition='inside',
                    insidetextanchor='end',
                    textangle=0,
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

        fig.update_xaxes(title_text="Expert Index", dtick=1)
        fig.update_yaxes(title_text="Selection Count")

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = f"{file_dir}/{moe_tracker_model}_all_layers_expert_selection_{moe_tracker_stage}.html"

        fig.write_html(file_name)
        print(f"figure saved {file_name}")

        # table
        analysis_results = {'prefill': [], 'decode': []}
        for layer_id, stats in moe_tracker_dict[moe_tracker_stage].items():
            expert_data = pd.DataFrame({
                'expert_id': range(len(stats)),
                'tokens': stats
            })

            sorted_experts = expert_data.sort_values(by='tokens', ascending=False)

            top_3 = sorted_experts.head(3)
            bottom_3 = sorted_experts.tail(3)

            layer_result = {
                'layer_id': layer_id,
                'top-1 exp': top_3.iloc[0]['expert_id'],
                'top-1 tokens': top_3.iloc[0]['tokens'],
                'top-2 exp': top_3.iloc[1]['expert_id'],
                'top-2 tokens': top_3.iloc[1]['tokens'],
                'top-3 exp': top_3.iloc[2]['expert_id'],
                'top-3 tokens': top_3.iloc[2]['tokens'],
                'bottom-3 exp': bottom_3.iloc[0]['expert_id'],
                'bottom-3 tokens': bottom_3.iloc[0]['tokens'],
                'bottom-2 exp': bottom_3.iloc[1]['expert_id'],
                'bottom-2 tokens': bottom_3.iloc[1]['tokens'],
                'bottom-1 exp': bottom_3.iloc[2]['expert_id'],
                'bottom-1 tokens': bottom_3.iloc[2]['tokens'],
            }

            analysis_results[moe_tracker_stage].append(layer_result)

        df = pd.DataFrame(analysis_results[moe_tracker_stage])    
        file_name = f"{file_dir}/{moe_tracker_model}_all_layers_expert_selection_{moe_tracker_stage}.csv"
        df.to_csv(file_name, index=False)
        print(f"data saved to {file_name}")
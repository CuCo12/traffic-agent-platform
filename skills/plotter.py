import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import tool
from typing import Optional, Union, List

@tool
def plot_metrics_comparison(run1_path: Union[str, List[str]], run2_path: Optional[Union[str, List[str]]] = None) -> str:
    """
    读取一个或两个仿真日志路径列表中的 training_history.json，绘制并保存关键训练曲线对比图。
    支持输入单个路径字符串，或包含多个种子路径的列表。
    当提供多个种子路径时，将计算并绘制均值曲线以及标准差范围填充阴影区。
    绘制指标包括：平均旅行时间（ATT）、车辆吞吐量（Throughput）、平均速度或延时（Speed/Delay）、排队长度（Queue）。
    图表会被保存为 traffic_comparison.png，并返回该图片的绝对路径。
    """
    
    def get_paths_list(path_input):
        if not path_input:
            return []
        if isinstance(path_input, list):
            return [os.path.abspath(p) for p in path_input if p]
        if isinstance(path_input, str):
            # 兼容 JSON 格式的字符串列表
            if path_input.startswith("[") and path_input.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(path_input)
                    if isinstance(parsed, list):
                        return [os.path.abspath(p) for p in parsed if p]
                except Exception:
                    pass
            return [os.path.abspath(path_input)]
        return []

    paths1 = get_paths_list(run1_path)
    paths2 = get_paths_list(run2_path)

    # 读取并聚合多模型种子数据
    def load_averaged_data(paths):
        valid_data = []
        for p in paths:
            p_json = os.path.join(p, "training_history.json") if os.path.isdir(p) else p
            if os.path.exists(p_json):
                try:
                    with open(p_json, 'r', encoding='utf-8') as f:
                        valid_data.append(json.load(f))
                except Exception:
                    pass
        if not valid_data:
            return None
            
        # 寻找对齐的最小轮数
        lengths = [len(d.get("rounds", [])) for d in valid_data]
        if not lengths or min(lengths) == 0:
            return None
        min_len = min(lengths)
        
        rounds = valid_data[0].get("rounds", [])[:min_len]
        
        # 自适应字段判断
        # 1. 吞吐量判断：如果有 completed_history，则使用 completed_history，否则使用 round_debug 里的 throughput
        use_completed = any("completed_history" in d and d["completed_history"] for d in valid_data)
        
        # 2. 速度/延时判断：如果有 speed_history，则使用 speed_history；否则使用 delay_history
        use_speed = any("speed_history" in d and d["speed_history"] for d in valid_data)
        use_delay = not use_speed and any("delay_history" in d and d["delay_history"] for d in valid_data)
        
        # 提取指标均值与标准差
        def extract_field_stats(field_name, extract_from_round_debug=False, force_completed=False, force_delay=False):
            series_list = []
            for d in valid_data:
                if force_completed and "completed_history" in d:
                    series = d.get("completed_history", [])[:min_len]
                elif force_delay and "delay_history" in d:
                    series = d.get("delay_history", [])[:min_len]
                elif extract_from_round_debug:
                    series = [r.get(field_name, 0.0) for r in d.get("round_debug", [])[:min_len]]
                else:
                    series = d.get(field_name, [])[:min_len]
                # 长度裁剪或零填充
                if len(series) < min_len:
                    series = series + [0.0] * (min_len - len(series))
                else:
                    series = series[:min_len]
                series_list.append(series)
            
            arr = np.array(series_list)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            return mean.tolist(), std.tolist()
            
        att_mean, att_std = extract_field_stats("att_history")
        queue_mean, queue_std = extract_field_stats("queue_history")
        
        throughput_mean, throughput_std = extract_field_stats(
            "throughput", 
            extract_from_round_debug=not use_completed, 
            force_completed=use_completed
        )
        
        if use_speed:
            speed_mean, speed_std = extract_field_stats("speed_history")
            speed_key = "speed"
            speed_title = "Mean Speed"
            speed_ylabel = "Velocity (m/s)"
        elif use_delay:
            speed_mean, speed_std = extract_field_stats("delay_history", force_delay=True)
            speed_key = "speed"
            speed_title = "Average Delay"
            speed_ylabel = "Delay (Seconds)"
        else:
            speed_mean, speed_std = extract_field_stats("stops_history")
            speed_key = "speed"
            speed_title = "Average Stops"
            speed_ylabel = "Stops Count"
        
        return {
            "rounds": rounds,
            "att": {"mean": att_mean, "std": att_std},
            "speed": {"mean": speed_mean, "std": speed_std, "title": speed_title, "ylabel": speed_ylabel},
            "queue": {"mean": queue_mean, "std": queue_std},
            "throughput": {"mean": throughput_mean, "std": throughput_std},
            "num_seeds": len(valid_data)
        }

    data1 = load_averaged_data(paths1)
    data2 = load_averaged_data(paths2) if paths2 else None

    if not data1:
        return f"错误：未能从路径中提取任何合规的训练历史数据。"

    # 创建 2x2 子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Traffic Signal Control Training Metrics Comparison", fontsize=16, fontweight='bold', color='#1e293b')
    
    # 字体与配置样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 兼容中英文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    c_blue = '#3b82f6'
    c_orange = '#f97316'
    
    def get_label(data, paths):
        # 提取上一级目录作为显示标签（如 benchmark_ec）
        parent = os.path.basename(os.path.dirname(paths[0]))
        if data["num_seeds"] > 1:
            return f"{parent} (Mean of {data['num_seeds']} seeds)"
        else:
            return f"{parent}➔{os.path.basename(paths[0])}"
            
    label1 = get_label(data1, paths1)
    label2 = get_label(data2, paths2) if data2 else None

    # 定义绘图辅助器
    def plot_field(ax, key, title, ylabel):
        # 绘制 Run 1
        x1 = list(range(1, len(data1["rounds"]) + 1))
        m1 = data1[key]["mean"]
        s1 = data1[key]["std"]
        ax.plot(x1, m1, label=label1, color=c_blue, linewidth=2)
        if data1["num_seeds"] > 1:
            ax.fill_between(x1, [m - s for m, s in zip(m1, s1)], [m + s for m, s in zip(m1, s1)], color=c_blue, alpha=0.15)
        
        # 绘制 Run 2 (可选)
        if data2:
            x2 = list(range(1, len(data2["rounds"]) + 1))
            m2 = data2[key]["mean"]
            s2 = data2[key]["std"]
            ax.plot(x2, m2, label=label2, color=c_orange, linewidth=2, linestyle='--')
            if data2["num_seeds"] > 1:
                ax.fill_between(x2, [m - s for m, s in zip(m2, s2)], [m + s for m, s in zip(m2, s2)], color=c_orange, alpha=0.15)
            
        ax.set_title(title, fontsize=12, fontweight='bold', color='#334155')
        ax.set_xlabel("Rounds (轮次)", fontsize=10, color='#64748b')
        ax.set_ylabel(ylabel, fontsize=10, color='#64748b')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

    # 1. 绘制 ATT 曲线
    plot_field(axs[0, 0], "att", "Average Travel Time (ATT)", "Time (Seconds)")
    
    # 2. 绘制 吞吐量 曲线
    plot_field(axs[0, 1], "throughput", "Vehicle Throughput", "Vehicles Count")
    
    # 3. 绘制 平均速度 / 延时 / 停车 曲线
    plot_field(axs[1, 0], "speed", data1["speed"]["title"], data1["speed"]["ylabel"])
    
    # 4. 绘制 平均排队长度 曲线
    plot_field(axs[1, 1], "queue", "Average Queue Length", "Queue Vehicles")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存路径到 traffic_agent 目录下
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(os.path.join(output_dir, "../traffic_comparison.png"))
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

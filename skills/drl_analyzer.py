import math

def analyze_drl_convergence(rounds, att_history):
    """
    数学诊断强化学习训练曲线的收敛性与稳态稳定性。
    """
    if not att_history or len(att_history) < 10:
        return {
            "converged": False, 
            "converged_round": 0,
            "max_improvement": 0.0,
            "min_att": 0.0,
            "best_round": 0,
            "steady_mean": 0.0,
            "steady_std": 0.0,
            "steady_cv": 0.0,
            "summary": "ℹ️ 数据量不足（少于 10 轮），无法进行收敛性数学分析。"
        }
    
    init_att = att_history[0]
    min_att = min(att_history)
    best_round_idx = att_history.index(min_att)
    best_round = rounds[best_round_idx]
    max_improvement = ((init_att - min_att) / init_att) * 100 if init_att else 0.0
    
    window_size = 5
    converged_idx = None
    
    # 动态滑动窗口一阶导数稳定性判定
    for i in range(window_size, len(att_history) - 3):
        w_prev = sum(att_history[i-window_size:i]) / window_size
        w_curr = sum(att_history[i:i+window_size]) / window_size if i + window_size <= len(att_history) else sum(att_history[i:]) / (len(att_history) - i)
        
        diff = abs(w_curr - w_prev) / w_prev if w_prev else 0.0
        # 波动率阈值设为 1.5%
        if diff < 0.015:
            # 连续 3 个轮次均保持低波动以过滤短暂的虚假收敛
            is_stable = True
            for j in range(1, 4):
                if i + j + window_size > len(att_history):
                    break
                w_next_prev = sum(att_history[i+j-window_size:i+j]) / window_size
                w_next_curr = sum(att_history[i+j:i+j+window_size]) / window_size
                if abs(w_next_curr - w_next_prev) / w_next_prev >= 0.015:
                    is_stable = False
                    break
            if is_stable:
                converged_idx = i
                break
                
    if converged_idx is None:
        # 保底机制：若未找到完全收敛点，取最后 20% 区间作为稳态分析段
        converged_idx = int(len(att_history) * 0.8)
        converged = False
    else:
        converged = True
        
    converged_round = rounds[converged_idx]
    
    # 计算收敛后的稳态指标（均值、标准差、变异系数 CV）
    steady_atts = att_history[converged_idx:]
    mean_steady = sum(steady_atts) / len(steady_atts)
    variance = sum((x - mean_steady) ** 2 for x in steady_atts) / len(steady_atts)
    std_dev = math.sqrt(variance)
    # 变异系数 CV = (标准差 / 均值) * 100
    cv = (std_dev / mean_steady) * 100 if mean_steady else 0.0
    
    # 构造 Markdown 分析报告段落
    summary = (
        f"📊 【学习曲线收敛性与稳定性数理诊断】:\n"
        f"  - 稳态收敛判定: {'已进入稳态（已收敛）' if converged else '未表现出明显稳态（使用后 20% 轮次估算）'}\n"
        f"  - 稳态起始轮次: 第 {converged_round} 轮\n"
        f"  - 最佳旅行时间 (ATT): {min_att:.2f} 秒 (发生在第 {best_round} 轮)\n"
        f"  - 最大性能提升幅度: {max_improvement:+.1f}%\n"
        f"  - 收敛稳态阶段均值: {mean_steady:.2f} 秒\n"
        f"  - 收敛稳态阶段标准差 (波动率): {std_dev:.3f} 秒 (体现训练的稳定性阻尼)\n"
        f"  - 稳态变异系数 (CV): {cv:.2f}% (变异系数越小，表示训练收敛后的控制抗震荡、抗随机噪声抖动性越强)\n"
    )
    
    return {
        "converged": converged,
        "converged_round": converged_round,
        "max_improvement": max_improvement,
        "min_att": min_att,
        "best_round": best_round,
        "steady_mean": mean_steady,
        "steady_std": std_dev,
        "steady_cv": cv,
        "summary": summary
    }

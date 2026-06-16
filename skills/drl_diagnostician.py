import os
import json
import logging
from tools.drl_analyzer import analyze_drl_convergence

logger = logging.getLogger(__name__)

def diagnose_drl_history(log_path: str) -> dict:
    """
    认知技能：控制理论数理诊断分析。
    自动检测指定仿真日志路径下的 training_history.json，加载并运行一阶导数稳定性判定算法，
    最终生成包含收敛轮次、稳态均值、标准差、变异系数（CV）的数理评估结果。
    """
    error_res = {
        "converged": False,
        "converged_round": 0,
        "max_improvement": 0.0,
        "min_att": 0.0,
        "best_round": 0,
        "steady_mean": 0.0,
        "steady_std": 0.0,
        "steady_cv": 0.0,
        "summary": "⚠️ 未能获取到有效的 DRL 训练曲线历史记录（缺少 training_history.json 或数据格式错误）。"
    }

    if not log_path:
        return error_res

    json_path = log_path
    if os.path.isdir(log_path):
        json_path = os.path.join(log_path, "training_history.json")

    if not os.path.exists(json_path):
        return error_res

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rounds = data.get("rounds", [])
        att_history = data.get("att_history", [])
        
        if rounds and att_history:
            # 调用底层 math tool 进行物理计算
            return analyze_drl_convergence(rounds, att_history)
            
    except Exception as e:
        logger.warning(f"Error executing DRL diagnostics on {json_path}: {e}")
        error_res["summary"] = f"⚠️ 解析 DRL 收敛数据失败: {e}"

    return error_res

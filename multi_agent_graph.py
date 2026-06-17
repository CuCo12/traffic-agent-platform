import os
import sys
import re
import operator
import io
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# 解决 Windows 控制台 GBK 编码无法打印表情/特殊字符导致的 UnicodeEncodeError 问题
if sys.platform.startswith('win') and 'streamlit' not in sys.modules:
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

# 确保导入同目录下的 agent.py 和 skills
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from agent import llm, analyze_cityflow_log, search_traffic_literature
from tools.arxiv_searcher import search_latest_arxiv_papers
from tools.plotter import plot_metrics_comparison
from tools.drl_analyzer import analyze_drl_convergence

BASE_DIR = CURRENT_DIR

# 1. 定义全局状态 (State) —— 这是多个 Agent 共享的“大脑内存”
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    baseline_data: dict       # 存放基准模型的分析数据
    optimized_data: dict      # 存放对比优化模型的分析数据（可选）
    comparison_img: str       # 对比图表图片的绝对路径
    review_feedback: str      # 存放评审员的修改意见
    final_report: str         # 存放最终报告
    retry_count: int          # 重试计数器，防止评审打回导致死循环

# 辅助函数：实时输出进度到 Streamlit 页面与控制台，方便排查卡顿
def log_progress(message: str):
    try:
        print(message)
    except Exception:
        pass
    import sys
    if 'streamlit' in sys.modules:
        import streamlit as st
        if "logs" in st.session_state and "logs_placeholder" in st.session_state:
            if st.session_state.logs_placeholder is not None:
                st.session_state.logs.append(message)
                st.session_state.logs_placeholder.markdown("\n".join(st.session_state.logs))

# 2. 定义节点 (Nodes) —— 具体的 Agent 角色
def data_analyst_node(state: AgentState):
    """分析师：提取并对比仿真指标，调用工具生成趋势图"""
    log_progress("⏳ **[分析师 Agent]** 开始处理仿真指标分析流程...")
    
    # 提取输入消息中的最新一条 HumanMessage 作为解析对象，避免会话历史干扰
    paths = []
    user_content = ""
    latest_human = None
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            latest_human = message
            break
            
    if latest_human:
        user_content = latest_human.content
        # 正则匹配 Windows/Linux 绝对路径，使用 a-zA-Z0-9 排除中文字符干扰
        found = re.findall(r'[a-zA-Z]:\\[a-zA-Z0-9\\\.\-\_]+|[a-zA-Z]:/[a-zA-Z0-9//\.\-\_]+', latest_human.content)
        if found:
            for p in found:
                p_abs = os.path.abspath(p)
                if os.path.exists(p_abs) and p_abs not in paths:
                    paths.append(p_abs)
                        
    # 使用 LLM 智能分类用户意图，判断是需要进行仿真日志诊断与绘图（DIAGNOSTIC）还是纯学术文献问答（QUERY_ONLY）
    intent = "QUERY_ONLY"
    if paths:
        intent = "DIAGNOSTIC"
    else:
        intent_prompt = (
            "您是一个交通信号控制系统的意图分类器。\n"
            "评估用户的提问，判断该问题是需要【读取本地运行日志、进行仿真指标提取并绘制对比曲线图】（DIAGNOSTIC），还是【纯学术论文检索、文献问答、算法原理对比、论文写作建议】（QUERY_ONLY）。\n"
            "重要注意点：\n"
            "- 如果问题包含具体的日志对比、画收敛图、提取运行吞吐量/延迟等指标（哪怕没有写路径，如“对比benchmark_ec的三个种子”），请分类为 DIAGNOSTIC。\n"
            "- 如果是查询文献、写论文摘要总结、对比学术论文设计异同、评价论文水平等纯学术文字问答，请分类为 QUERY_ONLY。\n\n"
            f"【用户提问】：\"{user_content}\"\n\n"
            "请仅回答 DIAGNOSTIC 或 QUERY_ONLY，不要有任何其他多余字符。"
        )
        try:
            res = llm.invoke(intent_prompt).content.strip().upper()
            if "DIAGNOSTIC" in res:
                intent = "DIAGNOSTIC"
        except Exception:
            # 容错：如果 LLM 调用失败，使用关键字保底
            has_diagnostic_kw = any(kw in user_content.lower() for kw in ["诊断", "评估", "仿真", "日志", "history", "log", "training", "training_history", "图", "曲线", "收敛", "吞吐", "延迟"])
            has_academic_kw = any(kw in user_content.lower() for kw in ["文献", "论文", "paper", "arxiv", "pdf", "摘要"])
            if has_diagnostic_kw and not (has_academic_kw and "画" not in user_content and "绘" not in user_content):
                intent = "DIAGNOSTIC"
                
    if intent == "QUERY_ONLY":
        log_progress("➔ ℹ️ 检测为纯学术提问请求，跳过日志解析与图表绘制。")
        query_only_data = {
            "log_path": "",
            "raw_analysis": "无仿真数据，本次运行为学术问题解答。",
            "throughput": 0.0,
            "avg_delay": 0.0,
            "status": "query_only"
        }
        return {
            "baseline_data": query_only_data,
            "optimized_data": {},
            "comparison_img": "",
            "messages": [AIMessage(content="分析师：学术问答请求。", name="analyst")]
        }
    
    baseline_path = paths[0] if len(paths) > 0 else None
    optimized_path = paths[1] if len(paths) > 1 else None
    
    # 智能扫描和关键字查找模型库（如果消息中提及了模型类别如 benchmark_ec 等且包含城市/种子）
    if not baseline_path:
        base_model_dir = os.path.abspath(os.path.join(BASE_DIR, "../TransformerLight-main/model"))
        log_progress(f"➔ 📁 未在输入中发现绝对路径，启动智能模型库关键词匹配...")
        
        # 匹配模型类别目录
        model_keywords = []
        for cat in ["benchmark_adco", "benchmark_amp", "benchmark_dqn", "benchmark_ec", "benchmark_ep", "ablation"]:
            if cat in user_content.lower() or cat.replace("_", "") in user_content.lower():
                model_keywords.append(cat)
                
        # 匹配城市
        city_keyword = ""
        if "hangzhou" in user_content.lower() or "杭州" in user_content.lower() or "5816" in user_content.lower():
            city_keyword = "hangzhou"
        elif "jinan" in user_content.lower() or "济南" in user_content.lower():
            city_keyword = "jinan"
        elif "newyork" in user_content.lower() or "new york" in user_content.lower() or "纽约" in user_content.lower():
            city_keyword = "newyork"
            
        # 匹配种子/过滤词
        seed_keyword = ""
        for seed in ["5816", "2000", "2500"]:
            if seed in user_content.lower():
                seed_keyword = seed
                
        matched_paths = []
        if model_keywords:
            for cat in model_keywords:
                cat_dir = os.path.join(base_model_dir, cat)
                if os.path.exists(cat_dir):
                    subdirs = []
                    for root, dirs, files in os.walk(cat_dir):
                        if "training_history.json" in files:
                            folder_name = os.path.basename(root).lower()
                            # 判断是否匹配城市与种子
                            match_city = (not city_keyword) or (city_keyword in folder_name)
                            match_seed = (not seed_keyword) or (seed_keyword in folder_name)
                            if match_city and match_seed:
                                subdirs.append(root)
                    if subdirs:
                        # 按字母名称排序，确保多个类别种子对齐（seed0 vs seed0, seed1 vs seed1 ...）
                        subdirs.sort()
                        if len(subdirs) > 1:
                            matched_paths.append(subdirs)
                        else:
                            matched_paths.append(subdirs[0])
                        
        if len(matched_paths) >= 1:
            baseline_path = matched_paths[0]
            if isinstance(baseline_path, list):
                log_progress(f"➔ 🔍 智能匹配基准路径 (共 {len(baseline_path)} 个种子): `{model_keywords[0]}` 的匹配子目录")
            else:
                log_progress(f"➔ 🔍 智能匹配基准路径: `{os.path.relpath(baseline_path, base_model_dir)}`")
        if len(matched_paths) >= 2:
            optimized_path = matched_paths[1]
            if isinstance(optimized_path, list):
                log_progress(f"➔ 🔍 智能匹配对比路径 (共 {len(optimized_path)} 个种子): `{model_keywords[1]}` 的匹配子目录")
            else:
                log_progress(f"➔ 🔍 智能匹配对比路径: `{os.path.relpath(optimized_path, base_model_dir)}`")
            
    # 如果智能匹配和正则匹配均未找到，则自动扫描最新生成的模型日志目录作为基准路径（保底逻辑）
    if not baseline_path:
        base_model_dir = os.path.abspath(os.path.join(BASE_DIR, "../TransformerLight-main/model"))
        log_progress(f"➔ 📁 未匹配到特定关键词，正在扫描并加载最新仿真日志...")
        candidate_dirs = []
        if os.path.exists(base_model_dir):
            for root, dirs, files in os.walk(base_model_dir):
                if "training_history.json" in files:
                    history_file_path = os.path.join(root, "training_history.json")
                    mtime = os.path.getmtime(history_file_path)
                    candidate_dirs.append((mtime, root))
        if candidate_dirs:
            candidate_dirs.sort(key=lambda x: x[0], reverse=True)
            baseline_path = candidate_dirs[0][1]
            log_progress(f"➔ 🔍 自动选取最新仿真日志目录: `{baseline_path}`")
            
    def paths_exist(p_input):
        if not p_input:
            return False
        if isinstance(p_input, list):
            return all(os.path.exists(p) for p in p_input)
        return os.path.exists(p_input)
            
    if not paths_exist(baseline_path):
        log_progress("❌ 错误：未能找到任何有效的仿真日志目录，流程终止。")
        error_data = {
            "log_path": "",
            "raw_analysis": "未找到仿真日志数据。",
            "throughput": -1.0,
            "avg_delay": -1.0,
            "status": "error"
        }
        return {
            "baseline_data": error_data,
            "optimized_data": {},
            "comparison_img": "",
            "messages": [AIMessage(content="分析师：未找到仿真日志数据。", name="analyst")]
        }

    def get_drl_analysis(p):
        from tools.drl_diagnostician import diagnose_drl_history
        res = diagnose_drl_history(p)
        if res.get("summary", "").startswith("⚠️"):
            return None
        return res

    # 定义多目录指标计算与解析提取函数
    def analyze_paths(paths_input):
        if isinstance(paths_input, list):
            analyses = []
            att_list = []
            tp_list = []
            for p in paths_input:
                res = analyze_cityflow_log.invoke({"log_path": p})
                analyses.append(res)
                att_match = re.search(r"平均旅行时间\s*\(ATT\):\s*([\d\.]+)\s*秒", res)
                tp_match = re.search(r"车辆吞吐量\s*\(Throughput\):\s*([\d\.]+)\s*辆", res)
                
                if att_match: att_list.append(float(att_match.group(1)))
                else:
                    best_att_match = re.search(r"最佳平均旅行时间\s*\(ATT\):\s*([\d\.]+)\s*秒", res)
                    if best_att_match: att_list.append(float(best_att_match.group(1)))
                    
                if tp_match: tp_list.append(float(tp_match.group(1)))
                else:
                    best_tp_match = re.search(r"最佳吞吐量:\s*([\d\.]+)\s*辆", res)
                    if best_tp_match: tp_list.append(float(best_tp_match.group(1)))
                    
            avg_att = sum(att_list) / len(att_list) if att_list else 0.0
            avg_tp = sum(tp_list) / len(tp_list) if tp_list else 0.0
            
            summary = (
                f"🎯 【多种子仿真联合分析成功】\n"
                f"模型包含种子数: {len(paths_input)} 个\n"
                f"平均旅行时间 (ATT) 均值: {avg_att:.2f} 秒\n"
                f"车辆吞吐量 (Throughput) 均值: {avg_tp:.1f} 辆\n\n"
                f"详细单种子日志参考（首个种子）:\n{analyses[0]}"
            )
            if paths_input:
                drl_res = get_drl_analysis(paths_input[0])
                if drl_res:
                    summary += f"\n\n{drl_res['summary']}"
            return summary, avg_tp, avg_att
        else:
            res = analyze_cityflow_log.invoke({"log_path": paths_input})
            att_match = re.search(r"平均旅行时间\s*\(ATT\):\s*([\d\.]+)\s*秒", res)
            tp_match = re.search(r"车辆吞吐量\s*\(Throughput\):\s*([\d\.]+)\s*辆", res)
            att = float(att_match.group(1)) if att_match else 0.0
            tp = float(tp_match.group(1)) if tp_match else 0.0
            if att == 0.0:
                best_att_match = re.search(r"最佳平均旅行时间\s*\(ATT\):\s*([\d\.]+)\s*秒", res)
                att = float(best_att_match.group(1)) if best_att_match else 0.0
            if tp == 0.0:
                best_tp_match = re.search(r"最佳吞吐量:\s*([\d\.]+)\s*辆", res)
                tp = float(best_tp_match.group(1)) if best_tp_match else 0.0
            
            drl_res = get_drl_analysis(paths_input)
            if drl_res:
                res += f"\n\n{drl_res['summary']}"
            return res, tp, att

    # ================= 运行工具解析基准数据 =================
    log_progress(f"➔ ⚙️ 正在解析基准模型日志...")
    raw_analysis1, tp1, att1 = analyze_paths(baseline_path)
    baseline_data = {
        "log_path": baseline_path,
        "raw_analysis": raw_analysis1,
        "throughput": tp1,
        "avg_delay": att1,
        "status": "computed"
    }
    log_progress(f"  * [基准指标汇总] 平均吞吐量: {tp1:.1f} 辆 | 平均旅行时间(ATT): {att1:.2f} 秒")
    
    # ================= 运行工具解析对比数据 =================
    optimized_data = {}
    if paths_exist(optimized_path):
        log_progress(f"➔ ⚙️ 正在解析对比模型日志...")
        raw_analysis2, tp2, att2 = analyze_paths(optimized_path)
        optimized_data = {
            "log_path": optimized_path,
            "raw_analysis": raw_analysis2,
            "throughput": tp2,
            "avg_delay": att2,
            "status": "computed"
        }
        log_progress(f"  * [对比指标汇总] 平均吞吐量: {tp2:.1f} 辆 | 平均旅行时间(ATT): {att2:.2f} 秒")
        
    # ================= 调用绘图工具绘制数据图表 =================
    log_progress("➔ 📊 正在调用绘图工具 `plot_metrics_comparison` 绘制对比折线图...")
    tool_input = {"run1_path": baseline_path}
    if optimized_data and optimized_path:
        tool_input["run2_path"] = optimized_path
    img_path = plot_metrics_comparison.invoke(tool_input)
    log_progress(f"➔ 🎨 对比图表绘制成功，保存至: `{os.path.basename(img_path)}`")
    
    log_msg = f"分析师：指标分析及趋势图绘制完成。"
    return {
        "baseline_data": baseline_data,
        "optimized_data": optimized_data,
        "comparison_img": img_path,
        "messages": [AIMessage(content=log_msg, name="analyst")]
    }

def reviewer_node(state: AgentState):
    """评审员：审查数据逻辑，执行反思机制"""
    b_data = state.get("baseline_data", {})
    o_data = state.get("optimized_data", {})
    log_progress("⏳ **[评审员 Agent]** 启动指标合理性与物理规则审查...")
    
    # 如果是纯学术问答请求，免检直接通过
    if b_data.get("status") == "query_only":
        log_progress("➔ ✅ 检测为学术问答请求，自动审核通过。")
        return {
            "review_feedback": "PASS",
            "messages": [AIMessage(content="评审员：学术问答免检通过。", name="reviewer")]
        }
        
    b_att = b_data.get("avg_delay", 0.0)
    b_tp = b_data.get("throughput", 0.0)
    
    o_att = o_data.get("avg_delay", 0.0) if o_data else 0.0
    o_tp = o_data.get("throughput", 0.0) if o_data else 0.0
    
    retry_count = state.get("retry_count", 0)
    
    b_invalid = b_tp <= 0.0 or b_att > 3000.0 or b_att <= 0.0
    o_invalid = o_data and (o_tp <= 0.0 or o_att > 3000.0 or o_att <= 0.0)
    
    if b_invalid or o_invalid:
        log_progress(f"❌ 数据校验异常！基准延迟: {b_att}s, 对比延迟: {o_att}s")
        if retry_count >= 2:
            log_progress("⚠️ 达到最大打回次数限制，安全放行，强制通过。")
            return {
                "review_feedback": "PASS",
                "messages": [AIMessage(content="评审员：重试上限放行。", name="reviewer")]
            }
        
        feedback = "数据异常：延迟指标过高（超出3000秒）或吞吐量异常，打回分析师。"
        return {
            "review_feedback": feedback,
            "retry_count": retry_count + 1,
            "messages": [AIMessage(content=f"评审员打回: {feedback}", name="reviewer")]
        }
    else:
        log_progress("➔ ✅ 所有提取指标均通过物理合理性约束审查。")
        return {
            "review_feedback": "PASS",
            "messages": [AIMessage(content="评审员：校验通过。", name="reviewer")]
        }

def editor_node(state: AgentState):
    """主编：调用文献检索并生成评估报告"""
    log_progress("⏳ **[主编 Agent]** 启动背景文献检索与诊断报告编撰流程...")
    b_data = state.get("baseline_data", {})
    o_data = state.get("optimized_data", {})
    log_path = b_data.get("log_path", "")
    
    # 过滤出干净的对话历史（只包含用户和助手的对话，过滤掉内部节点消息）
    chat_messages = []
    for msg in state.get("messages", []):
        if hasattr(msg, "name") and msg.name in ["analyst", "reviewer"]:
            continue
        chat_messages.append(msg)
        
    # 提取最新的用户提问
    user_question = "诊断最新的仿真日志。"
    for message in reversed(chat_messages):
        if isinstance(message, HumanMessage):
            user_question = message.content
            break
            
    is_query_only = b_data.get("status") == "query_only"
    
    # 解析场景与算法，确定检索核心词
    first_path = log_path[0] if isinstance(log_path, list) else log_path
    path_lower = first_path.lower() if first_path else ""
    algo = "DQN"
    if "adco" in path_lower: algo = "ADCO"
    elif "colight" in path_lower: algo = "CoLight"
    elif "transformerlight" in path_lower: algo = "TransformerLight"
    else:
        # 在学术提问中模糊匹配算法名称
        for possible in ["colight", "transformerlight", "mplight", "presslight", "maxpressure", "dqn", "alignlight", "adco", "frap", "attendlight", "intellilight", "efficientlight", "sotl"]:
            if possible in user_question.lower():
                algo = possible.upper()
                break
        
    city = "Hangzhou"
    if "jinan" in path_lower or "jinan" in user_question.lower():
        city = "Jinan"
    elif "newyork" in path_lower or "new york" in user_question.lower():
        city = "New York"
        
    # ================= 运行本地与联网检索（智能按需触发） =================
    need_literature = any(kw in user_question.lower() for kw in [
        "文献", "论文", "research", "paper", "arxiv", "学术", "理论", 
        "机制", "支撑", "原理", "优势", "pdf", "文件", "阅读", "全文",
        "alignlight", "colight", "transformerlight", "adco", "附录", "推理", "公式", "表格"
    ])
    
    if not need_literature:
        # LLM 兜底分类路由器，识别更隐晦的学术意图 (如“对比注意力机制”)，同时包含历史上下文
        history_str = ""
        for msg in chat_messages[:-1]:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content_snippet = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_str += f"{role}: {content_snippet}\n"
            
        router_prompt = (
            "您是一个智能交通信号控制系统的意图分类器。\n"
            "请评估用户的提问，判断该问题是否涉及或需要阅读本地学术论文、文献库（PDF/TXT）或者需要联网检索（arXiv）学术理论、原理机制或算法对比。\n"
            "重要注意点：\n"
            "1. 如果只是询问系统怎么用、或是纯粹针对城市仿真运行产生的日志文件诊断（例如：“延迟是多少”，“吞吐量怎么是0”），请分类为 NO。\n"
            "2. 如果用户是在讨论学术论文或文献（例如前文讨论的 AlignLight 等），而当前问题询问的是关于这些文献的附录内容、公式、或者文献中介绍的算法指标（如推理时间等），请务必分类为 YES。\n\n"
            "【历史对话记录】:\n"
            f"{history_str}\n"
            f"【当前用户提问】：\"{user_question}\"\n\n"
            "请结合历史对话，判断当前提问是否需要学术文献/原理机制的支撑。请仅回答 YES 或 NO，不要有任何其他多余字符。"
        )
        try:
            res = llm.invoke(router_prompt).content.strip().upper()
            if "YES" in res:
                need_literature = True
                log_progress("➔ 🧠 [意图路由器] 结合历史对话语义，判定为学术文献/原理查询意图，激活检索流。")
        except Exception:
            pass
            
    # 如果确定需要检索文献，则进行检索词重写
    search_query = user_question
    if need_literature:
        from tools.query_rewriter import rewrite_query_for_rag
        try:
            rewritten = rewrite_query_for_rag(chat_messages, user_question, llm)
            if rewritten and rewritten != user_question:
                search_query = rewritten
                log_progress(f"➔ 🧠 [检索词扩展] 基于历史将检索词重写为: `{search_query}`")
        except Exception as e:
            log_progress(f"⚠️ [检索词扩展] 重写检索词时发生异常: {e}")

    if need_literature:
        log_progress(f"➔ 📖 正在检索本地知识库文献 (RAG): `{search_query}`...")
        literature_context = search_traffic_literature.invoke({"query": search_query})
        
        # 提取问题中出现的所有算法，在 arXiv 中用 OR 连接查询
        algos_to_search = []
        for a in ["colight", "transformerlight", "mplight", "presslight", "maxpressure", "dqn", "alignlight", "adco", "frap", "attendlight", "intellilight", "efficientlight", "sotl"]:
            if a in user_question.lower() or a in search_query.lower():
                algos_to_search.append(f'all:"{a}"')
        if algos_to_search:
            arxiv_query = f'all:"traffic signal control" AND ({" OR ".join(algos_to_search)})'
        else:
            arxiv_query = f'all:"traffic signal control" AND all:"{algo.lower()}"'
            
        log_progress(f"➔ 🌐 正在联网向 arXiv 学术服务器获取最新论文: `{arxiv_query}`...")
        try:
            arxiv_context = search_latest_arxiv_papers.invoke({"query": arxiv_query})
        except Exception as e:
            log_progress(f"⚠️ arXiv 联网检索发生网络延迟或异常: {e}")
            arxiv_context = "联网检索超时。"
    else:
        log_progress("➔ ℹ️ 检测到当前提问侧重于仿真指标对比与图表生成，已自动跳过本地文献 RAG 与 arXiv 联网检索。")
        literature_context = "用户未显式要求文献支撑。"
        arxiv_context = "用户未要求检索前沿文献。"
        
    # C. 组装比较数据
    if is_query_only:
        comp_info = "学术问答场景，无仿真运行日志。"
    else:
        if o_data:
            comp_info = (
                f"对比评估场景。包含以下两个模型路径数据：\n"
                f"1. 基准模型 (Baseline Run): 路径: {b_data.get('log_path')}\n"
                f"   运行指标: {b_data.get('raw_analysis')}\n"
                f"2. 对比模型 (Optimized Run): 路径: {o_data.get('log_path')}\n"
                f"   运行指标: {o_data.get('raw_analysis')}\n"
            )
        else:
            comp_info = (
                f"单模型评估场景。模型路径数据：\n"
                f"基准模型 (Baseline Run): 路径: {b_data.get('log_path')}\n"
                f"运行指标: {b_data.get('raw_analysis')}\n"
            )
            
    log_progress("🧠 **[主编 Agent]** 正在调用大模型流式撰写最终文档...")
    
    from tools.report_editor import compile_academic_report, compile_academic_report_stream
    
    import sys
    streamlit_active = 'streamlit' in sys.modules
    response = ""
    
    if streamlit_active:
        import streamlit as st
        if "report_placeholder" in st.session_state and st.session_state.report_placeholder is not None:
            placeholder = st.session_state.report_placeholder
            for chunk_content in compile_academic_report_stream(
                user_question, comp_info, literature_context, arxiv_context, is_query_only, chat_messages, llm
            ):
                response += chunk_content
                placeholder.markdown(response)
        else:
            response = compile_academic_report(
                user_question, comp_info, literature_context, arxiv_context, is_query_only, chat_messages, llm
            )
    else:
        response = compile_academic_report(
            user_question, comp_info, literature_context, arxiv_context, is_query_only, chat_messages, llm
        )
        
    report_file_path = os.path.abspath(os.path.join(BASE_DIR, "evaluation_report.md"))
    try:
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(response)
        log_progress(f"➔ ✨ 报告/解答生成完毕，已保存到本地: `{os.path.basename(report_file_path)}`")
    except Exception as e:
        log_progress(f"⚠️ 保存报告文件失败: {e}")
        
    return {
        "final_report": response,
        "messages": [AIMessage(content="报告生成并保存成功。", name="editor")]
    }

# 3. 定义条件路由 (Conditional Edges)
def should_continue(state: AgentState):
    feedback = state.get("review_feedback", "")
    if feedback == "PASS":
        return "editor"
    else:
        return "analyst"

# 4. 组装多智能体工作流 (Build the Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyst", data_analyst_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("editor", editor_node)

# 定义边
workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "reviewer")

# 评审员看完后，根据反馈决定去哪里
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "editor": "editor",
        "analyst": "analyst"
    }
)
workflow.add_edge("editor", END)

# 5. 编译图，并引入持久化记忆 checkpointer 和 interrupt_before（人机协同断点）
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["editor"])

# --- 运行测试 ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "cli_run"}}
    initial_state = {
        "messages": [HumanMessage(content="请帮我诊断最新的强化学习控制日志。")],
        "retry_count": 0
    }
    
    print("--- [开始运行工作流（断点前）] ---")
    for event in app.stream(initial_state, config):
        print(event)
        
    state = app.get_state(config)
    if state.next:
        print(f"\n🛑 [断点触发] 工作流已在节点 {state.next} 前挂起！")
        val = input("按回车 [Enter] 恢复运行，继续生成报告：")
        print("\n--- [恢复运行工作流（断点后）] ---")
        for event in app.stream(None, config):
            print(event)
            
    final_state = app.get_state(config)
    print("\n==============================================")
    print("🎉 最终生成的报告预览：")
    print(final_state.values.get("final_report", "生成失败"))
    print("==============================================")
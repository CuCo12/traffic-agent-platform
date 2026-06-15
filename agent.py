import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from skills.arxiv_searcher import search_latest_arxiv_papers

try:
    from langchain.agents import initialize_agent, AgentType
except ImportError:
    from langchain_classic.agents import initialize_agent, AgentType

import logging
logging.getLogger('pdfminer').setLevel(logging.ERROR)

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

# 从本地 .env 文件中加载环境变量
def load_local_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_local_env()

# 初始化大模型 (使用 deepseek-chat)
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.1
)

# 定义文件路径
BASE_DIR = "E:/transformerlight/TransformerLight-main/traffic_agent"
PAPERS_DIR = os.path.join(BASE_DIR, "papers")

# 确保文件夹存在
os.makedirs(PAPERS_DIR, exist_ok=True)


# =====================================================================
# 步骤 2：定义工具 1 - 真实仿真与训练历史日志数据解析
# =====================================================================
@tool
def analyze_cityflow_log(log_path: str) -> str:
    """
    读取并解析指定路径下的 CityFlow 交通信号控制仿真日志或训练历史记录 (如 training_history.json)，
    计算并提取出平均旅行时间 (ATT)、停车次数、吞吐量、平均队列长度、平均车速等核心指标在训练前后的变化。
    """
    # 归一化路径
    log_path = os.path.abspath(log_path)
    
    # 如果输入的是目录，尝试在该目录下寻找 training_history.json
    json_path = log_path
    if os.path.isdir(log_path):
        potential_json = os.path.join(log_path, "training_history.json")
        if os.path.exists(potential_json):
            json_path = potential_json
        else:
            return f"❌ 输入的路径是目录，但在其下未找到 'training_history.json'。该目录下的文件有: {os.listdir(log_path)}"
            
    if not os.path.exists(json_path):
        return f"❌ 文件不存在: {json_path}。请输入正确的仿真日志或 json 文件路径。"
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取关键数据字段
        rounds = data.get("rounds", [])
        att_history = data.get("att_history", [])
        stops_history = data.get("stops_history", [])
        queue_history = data.get("queue_history", [])
        speed_history = data.get("speed_history", [])
        round_debug = data.get("round_debug", [])
        completed_history = data.get("completed_history", [])
        delay_history = data.get("delay_history", [])
        
        if not rounds:
            return f"❌ 成功读取 JSON，但未找到有效的 'rounds' 训练数据列表。"
            
        total_rounds = len(rounds)
        last_n = min(5, total_rounds)
        
        # 找出最佳的 Round (以 ATT 最小为准)
        best_idx = 0
        min_att = float('inf')
        for idx, att in enumerate(att_history):
            if att < min_att:
                min_att = att
                best_idx = idx
        best_round_num = rounds[best_idx]
        
        # 1. 吞吐量自适应提取
        has_completed = bool(completed_history)
        if has_completed:
            init_throughput = completed_history[0]
            final_throughput = sum(completed_history[-last_n:]) / last_n
            best_throughput = completed_history[best_idx]
        else:
            initial_round = round_debug[0] if round_debug else {}
            init_throughput = initial_round.get("throughput", 0.0)
            final_throughput = sum([r.get("throughput", 0) for r in round_debug[-last_n:]]) / last_n if round_debug else 0.0
            best_throughput = round_debug[best_idx].get("throughput", 0.0) if round_debug and best_idx < len(round_debug) else 0.0
            
        # 2. 速度/延时自适应提取
        has_speed = bool(speed_history)
        has_delay = not has_speed and bool(delay_history)
        
        if has_speed:
            speed_label = "平均行车速度"
            speed_unit = "m/s"
            init_speed = speed_history[0]
            final_speed = sum(speed_history[-last_n:]) / last_n
            best_speed = speed_history[best_idx]
        elif has_delay:
            speed_label = "平均延时 (Delay)"
            speed_unit = "秒"
            init_speed = delay_history[0]
            final_speed = sum(delay_history[-last_n:]) / last_n
            best_speed = delay_history[best_idx]
        else:
            speed_label = "平均停车次数"
            speed_unit = "次"
            init_speed = stops_history[0] if stops_history else 0.0
            final_speed = sum(stops_history[-last_n:]) / last_n if stops_history else 0.0
            best_speed = stops_history[best_idx] if stops_history else 0.0

        # 其他基础指标
        init_att = att_history[0] if att_history else 0.0
        init_queue = queue_history[0] if queue_history else 0.0
        init_stops = stops_history[0] if stops_history else 0.0
        
        final_att = sum(att_history[-last_n:]) / last_n if att_history else 0.0
        final_queue = sum(queue_history[-last_n:]) / last_n if queue_history else 0.0
        final_stops = sum(stops_history[-last_n:]) / last_n if stops_history else 0.0
        
        # 计算改善比例
        att_diff = ((final_att - init_att) / init_att) * 100 if init_att else 0.0
        speed_diff = ((final_speed - init_speed) / init_speed) * 100 if init_speed else 0.0
        queue_diff = ((final_queue - init_queue) / init_queue) * 100 if init_queue else 0.0
        stops_diff = ((final_stops - init_stops) / init_stops) * 100 if init_stops else 0.0
        throughput_diff = ((final_throughput - init_throughput) / init_throughput) * 100 if init_throughput else 0.0
        
        summary = (
            f"🎯 【真实训练历史分析成功】\n"
            f"路网/场景名称: {os.path.basename(os.path.dirname(json_path))}\n"
            f"总训练轮数 (Rounds): {total_rounds} 轮\n\n"
            f"📈 【初始阶段 (第1轮) 指标】:\n"
            f"  - 平均旅行时间 (ATT): {init_att:.2f} 秒\n"
            f"  - {speed_label}: {init_speed:.2f} {speed_unit}\n"
            f"  - 平均排队长度: {init_queue:.2f}\n"
            f"  - 总停车次数: {init_stops:.0f} 次\n"
            f"  - 车辆吞吐量 (Throughput): {init_throughput:.0f} 辆\n\n"
            f"📉 【收敛稳定阶段 (最后5轮均值) 指标】:\n"
            f"  - 平均旅行时间 (ATT): {final_att:.2f} 秒 (变化: {att_diff:+.1f}%)\n"
            f"  - {speed_label}: {final_speed:.2f} {speed_unit} (变化: {speed_diff:+.1f}%)\n"
            f"  - 平均排队长度: {final_queue:.2f} (变化: {queue_diff:+.1f}%)\n"
            f"  - 总停车次数: {final_stops:.0f} 次 (变化: {stops_diff:+.1f}%)\n"
            f"  - 车辆吞吐量 (Throughput): {final_throughput:.1f} 辆 (变化: {throughput_diff:+.1f}%)\n\n"
            f"🏆 【历史最佳轮次 (第{best_round_num}轮) 指标】:\n"
            f"  - 最佳平均旅行时间 (ATT): {min_att:.2f} 秒\n"
            f"  - 最佳吞吐量: {best_throughput:.0f} 辆\n"
            f"  - 最佳{speed_label}: {best_speed:.2f} {speed_unit}\n"
            f"  - 最佳平均排队: {queue_history[best_idx]:.2f} 辆\n"
            f"  - 探索率 (Epsilon) 最终衰减至: {data.get('epsilon_history', [])[-1] if data.get('epsilon_history') else 'N/A'}\n"
        )
        return summary
        
    except Exception as e:
        return f"❌ 解析 JSON 文件时出错: {e}"


# =====================================================================
# 步骤 3：定义工具 2 - 在线大模型重排/智能语义检索文献 (Online RAG Reader)
# =====================================================================
# 全局文献段落缓存，避免每次工具调用都重新解析所有 PDF (极大提升检索性能与避免磁盘 I/O 延迟)
PAPERS_CACHE = {}

def _parse_pdf_with_tables(file_path: str, filename: str) -> list:
    """
    使用 pdfplumber 提取 PDF 页面中的文本，并自动识别表格，将其转化为 Markdown 格式的表格插入到页面文本下方。
    """
    import pdfplumber
    file_segments = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                tables = page.extract_tables()
                
                # 转换表格为 Markdown 格式
                formatted_tables = []
                if tables:
                    for table in tables:
                        # 过滤掉全为空的行
                        valid_rows = [[str(cell or '').strip() for cell in row] for row in table if any(cell is not None for cell in row)]
                        if not valid_rows or len(valid_rows) < 1:
                            continue
                        
                        headers = valid_rows[0]
                        rows = valid_rows[1:]
                        
                        # 计算各列最大宽度，便于对齐
                        try:
                            col_widths = [max(len(cell) for cell in col) for col in zip(*valid_rows)]
                        except Exception:
                            # 容错：如果行长度不一致，使用平均宽度
                            col_widths = [10] * len(headers)
                            
                        # 构建 Markdown 表格
                        hdr_str = " | ".join(f"{cell:<{w}}" for cell, w in zip(headers, col_widths))
                        sep_str = "-|-".join("-" * w for w in col_widths)
                        
                        md_table = f"| {hdr_str} |\n| {sep_str} |\n"
                        for row in rows:
                            # 长度补齐或截断，防止维度不匹配报错
                            if len(row) < len(headers):
                                row = row + [''] * (len(headers) - len(row))
                            else:
                                row = row[:len(headers)]
                            row_str = " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths))
                            md_table += f"| {row_str} |\n"
                        
                        formatted_tables.append(md_table)
                
                content = page_text
                if formatted_tables:
                    content += "\n\n### [页内提取的表格数据]\n" + "\n\n".join(formatted_tables)
                
                if len(content.strip()) > 30:
                    file_segments.append({
                        "id": f"{filename}#page_{idx}",
                        "source": f"{filename} (第{idx+1}页)",
                        "content": content.strip()
                    })
    except Exception:
        pass
    return file_segments

@tool
def search_traffic_literature(query: str) -> str:
    """
    在交通信号控制学术知识库/文献中检索与查询内容最相关的文献、论文内容、理论定义及核心指标说明。
    该工具通过在线 DeepSeek 大模型强大的上下文注意力语义分析，实现极其高精度的学术段落检索与重排。
    适用于回答如：“什么是 MPLight 的车道压力？”、“CoLight 是如何实现协同控制的？”等学术理论问题。
    """
    import re
    # 如果 papers 文件夹为空
    if not os.path.exists(PAPERS_DIR) or not os.listdir(PAPERS_DIR):
        return "ℹ️ 本地文献库为空。请把相关的 .txt / .md / .pdf 论文放入 papers 文件夹。"

    global PAPERS_CACHE
    current_files = os.listdir(PAPERS_DIR)
    
    # 清理缓存中已被删除的文件
    for cached_file in list(PAPERS_CACHE.keys()):
        if cached_file not in current_files:
            del PAPERS_CACHE[cached_file]

    # 1. 在线读取所有本地文件段落 (优先使用内存缓存，基于修改时间进行失效判断)
    for filename in current_files:
        file_path = os.path.join(PAPERS_DIR, filename)
        try:
            mtime = os.path.getmtime(file_path)
        except Exception:
            mtime = 0.0
            
        if filename in PAPERS_CACHE:
            cached_mtime, cached_segments = PAPERS_CACHE[filename]
            if cached_mtime == mtime:
                continue
            
        file_segments = []
        if filename.endswith(".md") or filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
                for idx, para in enumerate(paragraphs):
                    file_segments.append({
                        "id": f"{filename}#segment_{idx}",
                        "source": filename,
                        "content": para
                    })
            except Exception:
                pass
        # 兼容 PDF 论文解析，使用 pdfplumber 提取文本并智能转出表格
        elif filename.endswith(".pdf"):
            file_segments = _parse_pdf_with_tables(file_path, filename)
        
        if file_segments:
            PAPERS_CACHE[filename] = (mtime, file_segments)

    # 合并所有段落
    all_segments = []
    for cached_mtime, segments in PAPERS_CACHE.values():
        all_segments.extend(segments)

    if not all_segments:
        return "ℹ️ 未在 papers 文件夹中读取到任何有效的文本段落。"

    # 2. 对查询条件进行分词，并评估每个段落的匹配相关度得分 (Relevance Scorer & Ranker)
    query_lower = query.lower()
    
    # 提取英文单词 (长度 >= 2)
    eng_keywords = [w for w in re.findall(r'[a-zA-Z0-9_-]+', query_lower) if len(w) > 1]
    # 提取中文词组 (长度 >= 2)
    zh_keywords = [w for w in re.findall(r'[\u4e00-\u9fa5]+', query) if len(w) > 1]
    keywords = list(set(eng_keywords + zh_keywords))
    
    scored_segments = []
    for seg in all_segments:
        score = 0
        content_lower = seg["content"].lower()
        source_lower = seg["source"].lower()
        
        # A. 关键词匹配得分 (按不同关键词匹配计分，多次匹配有少量额外加分)
        for kw in keywords:
            if kw in content_lower:
                score += 10
                score += min(5, content_lower.count(kw))
                
        # B. 文件名强关联匹配得分 (极其重要：如果用户提到了某篇论文，其段落优先被送入大模型)
        for filename in current_files:
            name_no_ext = os.path.splitext(filename)[0].lower()
            if name_no_ext in query_lower:
                if name_no_ext in source_lower:
                    score += 150  # 极大幅度加分，确保该论文内容不被 truncation 丢弃
                    
        if score > 0:
            scored_segments.append((score, seg))
            
    # 按得分从高到低进行排序
    scored_segments.sort(key=lambda x: x[0], reverse=True)
    
    # 取前 15 个最相关的段落送去给大模型精排
    filtered_segments = [seg for _, seg in scored_segments[:15]]

    if not filtered_segments:
        filtered_segments = all_segments[:15]  # 保底策略

    # 3. 构建大模型重排 Prompt
    rerank_prompt = (
        "您是一个顶尖的交通信号控制学术论文检索与评选专家。\n"
        "请阅读以下从本地文献库中抽取的候选论文段落，并根据用户的学术问题进行【在线语义检索与相关度重排】。\n\n"
        f"【用户的学术问题】:\n\"{query}\"\n\n"
        "【候选论文段落列表】:\n"
    )
    for i, seg in enumerate(filtered_segments):
        rerank_prompt += f"--- [段落 ID: {i}] 来自文献 {seg['source']} ---\n{seg['content']}\n\n"
        
    rerank_prompt += (
        "【请执行以下任务】:\n"
        "1. 从中挑选出与用户问题最相关的 2 到 3 个段落，直接保留其原文内容。\n"
        "2. 在每一个挑选出的段落上方，用一句话点评它为什么与用户的问题相关。\n"
        "3. 不要回答用户的学术问题，只负责提取和评价这些被选中的段落原文。\n"
        "请直接输出结果："
    )

    try:
        # 使用在线大模型进行高精度语义筛选 (Online Reader & Reranker)
        retrieved_response = llm.predict(rerank_prompt)
        return retrieved_response
    except Exception as e:
        # 优雅降级：若大模型调用超时，返回本地基本匹配
        formatted = []
        for seg in filtered_segments[:2]:
            formatted.append(f"【来源文献: {seg['source']} (本地保底匹配)】\n{seg['content']}")
        return "\n\n---\n\n".join(formatted)


@tool
def read_local_paper_content(filename: str) -> str:
    """
    读取并返回本地 papers 目录下指定论文（如 'my_paper.pdf'）的完整文本内容。
    当用户要求你详细评估某篇特定的本地论文、进行全面审稿（Review）、深度润色、提取完整大纲或进行全身段对比时，请调用此工具以获取该论文的完整正文。
    """
    # 归一化文件名，确保带后缀或模糊匹配
    target_file = None
    for f in os.listdir(PAPERS_DIR):
        if filename.lower() in f.lower():
            target_file = f
            break
            
    if not target_file:
        return f"❌ 未在 papers 目录下找到匹配 '{filename}' 的文件。已存在的文献列表: {os.listdir(PAPERS_DIR)}"
        
    file_path = os.path.join(PAPERS_DIR, target_file)
    
    try:
        if target_file.endswith(".md") or target_file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif target_file.endswith(".pdf"):
            # 优先从全局缓存读取以加速
            global PAPERS_CACHE
            if target_file in PAPERS_CACHE:
                cached_mtime, segments = PAPERS_CACHE[target_file]
                return "\n\n".join([seg["content"] for seg in segments])
            else:
                segments = _parse_pdf_with_tables(file_path, target_file)
                return "\n\n".join([seg["content"] for seg in segments])
    except Exception as e:
        return f"❌ 读取文件 '{target_file}' 失败: {e}"


# =====================================================================
# 步骤 4：组装智能体并触发运行
# =====================================================================
if __name__ == "__main__":
    # 构建拥有外挂工具的 Agent
    tools = [analyze_cityflow_log, search_traffic_literature, search_latest_arxiv_papers, read_local_paper_content]

    # 初始化对话记忆，使 Agent 能够记住上下文对话历史
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 默认关闭 verbose 中间过程打印。如需查看大模型思考链条，可改为 True。
    agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=False,
        memory=memory
    )

    # 检查是否通过命令行参数传入了自定义需求
    if len(sys.argv) > 1:
        # 直接使用命令行参数拼接作为问题运行
        user_query = " ".join(sys.argv[1:])
        print(f"🚀 [正在执行命令行指令]: {user_query}\n")
        try:
            response = agent.run(user_query)
            print("\n==================================================")
            print(f"[Report] Agent 结合 RAG 参考文献生成的评估报告:\n{response}")
            print("==================================================")
            report_file_path = os.path.join(BASE_DIR, "evaluation_report.md")
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"\n✨ [报告已自动保存到本地]: {report_file_path}\n")
        except Exception as e:
            print(f"❌ 运行出错: {e}")
    else:
        # 进入智能交通 AI Terminal 对话模式
        print("="*60)
        print("🤖 欢迎使用 Traffic Agentic RAG 智能学术与仿真分析助理！")
        print("您可以直接在此处输入您自定义的任何交通控制或文献检索需求。")
        print("例如：")
        print("  👉 什么是 MPLight 的 pressure 定义？它和 MaxPressure 有什么区别？")
        print("  👉 帮我分析一下这个目录下的仿真指标: E:\\transformerlight\\TransformerLight-main\\TransformerLight-main\\model\\benchmark_dqn\\anon_4_4_hangzhou_real_05_19_07_11_01")
        print("  👉 结合本地文献中的 CoLight 协同控制原理，分析我们刚刚得出的仿真数据。")
        print("提示：输入 'exit' 或 'quit' 即可随时退出助理。")
        print("="*60)
        
        while True:
            try:
                user_query = input("\n👤 请输入您的自定义需求 >> ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ["exit", "quit"]:
                    print("\n👋 感谢使用！再见！")
                    break
                
                print("\n🧠 正在检索本地文献库与解析运行数据，请稍候 (大模型正在在线推理中)...")
                response = agent.run(user_query)
                
                print("\n==================================================")
                print(f"🤖 [Agent 最终生成的专业报告]:\n\n{response}")
                print("==================================================")
                
                # 自动写入 evaluation_report.md
                report_file_path = os.path.join(BASE_DIR, "evaluation_report.md")
                with open(report_file_path, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"\n✨ [报告已自动同步保存至]: {report_file_path}")
                
            except KeyboardInterrupt:
                print("\n👋 退出助理。")
                break
            except Exception as e:
                print(f"❌ 运行出错: {e}")
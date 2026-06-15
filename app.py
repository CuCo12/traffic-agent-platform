import os
import sys
import streamlit as st
import io
import re
from langchain_core.messages import HumanMessage, AIMessage

# 设置 Streamlit 页面配置
st.set_page_config(
    page_title="Traffic Agentic RAG Web Terminal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 确保导入同目录下的 multi_agent_graph.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from multi_agent_graph import app as graph_app

# ==========================================
# 缓存/初始化 Session State 状态
# ==========================================
if "graph_app" not in st.session_state:
    st.session_state.graph_app = graph_app

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "idle" # idle, running_first_half, paused, running_second_half, finished

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "你好！我是 **Traffic Agentic RAG** 智能交通助理。您可以直接与我对话：\n\n- ✍️ **输入学术提问**，如：`什么是 CoLight 协同机制？` 或 `解释 MPLight 的 Pressure 公式`\n- 📁 **输入仿真日志路径进行一键诊断**，例如：`请帮我诊断这个日志：E:/.../model/benchmark_dqn/anon_4_4_hangzhou...`\n- 📊 **对比评估两个日志**：`请帮我比对这两个日志：E:/.../run1 和 E:/.../run2`"}
    ]

if "logs" not in st.session_state:
    st.session_state.logs = []

if "logs_first_half" not in st.session_state:
    st.session_state.logs_first_half = []

if "logs_placeholder" not in st.session_state:
    st.session_state.logs_placeholder = None

if "baseline_data" not in st.session_state:
    st.session_state.baseline_data = {}

if "optimized_data" not in st.session_state:
    st.session_state.optimized_data = {}

if "comparison_img" not in st.session_state:
    st.session_state.comparison_img = ""

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

# ==========================================
# 辅助函数
# ==========================================
def scan_simulation_logs():
    """扫描项目模型文件夹，提取所有包含训练历史的日志文件夹"""
    base_model_dir = os.path.abspath(os.path.join(CURRENT_DIR, "../TransformerLight-main/model"))
    log_dirs = {}
    if os.path.exists(base_model_dir):
        for root, dirs, files in os.walk(base_model_dir):
            if "training_history.json" in files:
                folder_name = os.path.basename(root)
                parent_name = os.path.basename(os.path.dirname(root))
                display_name = f"📊 {parent_name} ➔ {folder_name}"
                log_dirs[display_name] = os.path.abspath(root)
    return log_dirs

# ==========================================
# UI 侧边栏渲染
# ==========================================
st.sidebar.markdown("### 📂 本地仿真日志资源管理器")
st.sidebar.write("您可以从下方复制路径并在对话框中提问，系统会自动抽取并分析该路径下的日志文件（悬停可查看完整路径）：")
log_dict = scan_simulation_logs()

if log_dict:
    # 按照模型前置文件夹分类归纳
    grouped_logs = {}
    for name, path in log_dict.items():
        match = re.search(r"📊\s*(.*?)\s*➔\s*(.*)", name)
        if match:
            category = match.group(1).strip()
            folder_name = match.group(2).strip()
        else:
            category = "其他模型"
            folder_name = name
        if category not in grouped_logs:
            grouped_logs[category] = []
        grouped_logs[category].append((folder_name, path))
        
    for category, items in grouped_logs.items():
        with st.sidebar.expander(f"📁 {category} ({len(items)} 个模型)"):
            for folder_name, path in items:
                st.markdown(f"**{folder_name}**")
                # 使用 HTML input 框配合 onclick="this.select();" 实现点击即可一键全选，解决分词导致双击只能复制一部分的痛点
                st.markdown(
                    f'''
                    <input type="text" value="{path}" readonly onclick="this.select();" 
                           title="点击即可全选，按 Ctrl+C 复制完整路径\n{path}" 
                           style="width: 100%; 
                                  background-color: rgba(59, 130, 246, 0.05); 
                                  padding: 6px 10px; 
                                  border-radius: 6px; 
                                  margin-bottom: 12px; 
                                  border: 1px solid rgba(59, 130, 246, 0.18); 
                                  font-family: Consolas, monospace; 
                                  font-size: 0.75rem; 
                                  color: #1d4ed8; 
                                  cursor: pointer; 
                                  outline: none; 
                                  box-sizing: border-box;"/>
                    ''',
                    unsafe_allow_html=True
                )
else:
    st.sidebar.error("❌ 未找到包含 'training_history.json' 的仿真目录！")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 对话快捷指令模版")

# 模版按钮
if st.sidebar.button("💡 模版 1：学术文献咨询 (CoLight)", use_container_width=True):
    st.session_state.chat_history.append({"role": "user", "content": "请分析近年来 CoLight 算法协同机制的核心原理，它与传统 MaxPressure 算法相比的优势在哪里？中途需要联网检索最新的文献支撑。"})
    st.session_state.current_prompt = "请分析近年来 CoLight 算法协同机制的核心原理，它与传统 MaxPressure 算法相比的优势在哪里？中途需要联网检索最新的文献支撑。"
    st.session_state.workflow_status = "running_first_half"
    st.rerun()

if log_dict:
    first_path = list(log_dict.values())[0]
    second_path = list(log_dict.values())[1] if len(log_dict) > 1 else first_path
    
    if st.sidebar.button("💡 模版 2：单日志仿真诊断", use_container_width=True):
        prompt = f"请帮我诊断这个仿真日志：\n{first_path}"
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.current_prompt = prompt
        st.session_state.workflow_status = "running_first_half"
        st.rerun()
        
    if st.sidebar.button("💡 模版 3：双日志对比诊断", use_container_width=True):
        prompt = f"请帮我比对评估下面两个仿真日志，看看优化后提升了多少，并画图对比：\n1. {first_path}\n2. {second_path}"
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.current_prompt = prompt
        st.session_state.workflow_status = "running_first_half"
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 重置对话与清除缓存", use_container_width=True):
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.workflow_status = "idle"
    st.session_state.logs = []
    st.session_state.logs_first_half = []
    st.session_state.logs_placeholder = None
    st.session_state.chat_history = [
        {"role": "assistant", "content": "系统已重置！请直接在对话框向我发起新的提问或诊断任务。"}
    ]
    st.session_state.baseline_data = {}
    st.session_state.optimized_data = {}
    st.session_state.comparison_img = ""
    st.session_state.current_prompt = ""
    st.rerun()

# ==========================================
# 对话界面与执行逻辑
# ==========================================

# 页面顶部 Banner (高颜值渐变仪表盘风格)
st.markdown("""
<div style="background: linear-gradient(135deg, #3b82f6, #6366f1); padding: 18px 24px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15);">
    <h1 style="color:#ffffff; margin:0; font-family:'Outfit', sans-serif; font-size:1.8rem; font-weight:700;">🤖 Traffic Agentic RAG</h1>
    <p style="color:rgba(255, 255, 255, 0.9); margin:4px 0 0 0; font-size:0.95rem; font-weight:400;">自适应多模型对比评估与学术文献检索 Web 系统</p>
</div>
""", unsafe_allow_html=True)

# 1. 渲染历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("img") and os.path.exists(msg["img"]):
            st.image(msg["img"], caption="收敛指标曲线比对图")

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 2. 状态机流转执行

# A. 运行第一半 (Analyst ➔ Reviewer)
if st.session_state.workflow_status == "running_first_half":
    # 每次新一轮开始，生成全新随机 thread_id，防止 LangGraph checkpoint 记忆带入导致上一次中断状态 carry-over Bug
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        # 注入自动滚动脚本，保持流式输出时页面锚定在最下方，防止布局收缩导致页面滚动条跳回顶部
        st.components.v1.html(
            """
            <script>
                const mainSection = window.parent.document.querySelector('section.main');
                if (mainSection) {
                    mainSection.scrollTo(0, mainSection.scrollHeight);
                    const observer = new MutationObserver(() => {
                        mainSection.scrollTo(0, mainSection.scrollHeight);
                    });
                    observer.observe(window.parent.document.body, { childList: true, subtree: true });
                    setTimeout(() => observer.disconnect(), 45000);
                }
            </script>
            """,
            height=0,
        )
        st.session_state.logs = []  # 清理之前的运行日志
        st.session_state.logs_first_half = []
        st.session_state.baseline_data = {}
        st.session_state.optimized_data = {}
        st.session_state.comparison_img = ""
        
        # 使用 Streamlit 官方自带的高颜值 st.status 动态进度框
        with st.status("🤖 Multi-Agent 工作流执行中...", expanded=True) as status_box:
            log_placeholder = st.empty()
            st.session_state.logs_placeholder = log_placeholder
            
            # 汇整历史消息传给 Graph，以维持多轮对话记忆
            graph_messages = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] == "user":
                    graph_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    graph_messages.append(AIMessage(content=msg["content"]))
            graph_messages.append(HumanMessage(content=st.session_state.current_prompt))
            
            initial_state = {
                "messages": graph_messages,
                "retry_count": 0,
                "baseline_data": {},
                "optimized_data": {},
                "comparison_img": "",
                "review_feedback": "",
                "final_report": ""
            }
            
            # 运行流并拦截事件
            for event in st.session_state.graph_app.stream(initial_state, config):
                if "analyst" in event:
                    st.session_state.comparison_img = event["analyst"].get("comparison_img", "")
            
            status_box.update(label="✅ 数据解析与合规校验完成！", state="complete", expanded=False)
            
        # 在进度框外渲染收敛折线图
        if st.session_state.comparison_img and os.path.exists(st.session_state.comparison_img):
            st.image(st.session_state.comparison_img, caption="仿真收敛趋势对比折线图")
            
        # 备份第一阶段的所有日志快照，以便在后续渲染中独立于第二阶段日志展示
        st.session_state.logs_first_half = list(st.session_state.logs)
            
        # 检查是否因为需要人工干预中断挂起
        state = st.session_state.graph_app.get_state(config)
        if state.next:
            b_data = state.values.get("baseline_data", {})
            # 如果是纯学术问答请求，跳过人工干预断点，直接进入第二阶段生成解答
            if b_data.get("status") == "query_only":
                st.session_state.workflow_status = "running_second_half"
                st.rerun()
            else:
                st.session_state.workflow_status = "paused"
                st.session_state.baseline_data = b_data
                st.session_state.optimized_data = state.values.get("optimized_data", {})
                st.session_state.comparison_img = state.values.get("comparison_img", "")
                st.rerun()
        else:
            # 学术问答直接走完，提取报告并归档到历史
            st.session_state.workflow_status = "finished"
            report = state.values.get("final_report", "执行异常。")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": report,
                "img": st.session_state.comparison_img
            })
            st.rerun()

# B. 挂起状态：等待人类协同确认 (Human-in-the-loop Gate)
elif st.session_state.workflow_status == "paused":
    with st.chat_message("assistant"):
        # 渲染这一轮的第一阶段执行日志（折叠状态框）
        with st.status("🤖 Multi-Agent 工作流执行进度 (第一阶段：数据分析与校验)", state="complete", expanded=False):
            st.markdown("\n".join(st.session_state.logs_first_half))
            
        if st.session_state.comparison_img and os.path.exists(st.session_state.comparison_img):
            st.image(st.session_state.comparison_img, caption="仿真收敛趋势对比折线图")
            
        # 渲染更精美的人工审核表单卡片
        st.markdown("""
        <div style="background-color: rgba(245, 158, 11, 0.05); padding: 16px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #f59e0b; border-right: 1px solid rgba(245, 158, 11, 0.15); border-top: 1px solid rgba(245, 158, 11, 0.15); border-bottom: 1px solid rgba(245, 158, 11, 0.15);">
            <h5 style="color:#b45309; margin-top:0; margin-bottom:6px; font-weight:bold;">⚠️ Human-in-the-loop 人机协同验证关卡</h5>
            <p style="color:#78350f; margin:0; font-size:0.9rem;">大模型主编撰写最终报告前，已触发安全中断。如果分析师提取的数据有偏差，您可以在下方手动微调，确认无误后点击批准继续。</p>
        </div>
        """, unsafe_allow_html=True)
        
        b_current = st.session_state.baseline_data
        o_current = st.session_state.optimized_data
        
        if o_current:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📁 基准模型指标修改")
                b_adj_tp = st.number_input("基准吞吐量 (辆):", value=float(b_current.get("throughput", 0.0)))
                b_adj_att = st.number_input("基准延时 ATT (秒):", value=float(b_current.get("avg_delay", 0.0)))
            with c2:
                st.markdown("##### 📁 对比模型指标修改")
                o_adj_tp = st.number_input("对比吞吐量 (辆):", value=float(o_current.get("throughput", 0.0)))
                o_adj_att = st.number_input("对比延时 ATT (秒):", value=float(o_current.get("avg_delay", 0.0)))
        else:
            st.markdown("##### 📁 仿真指标修改")
            b_adj_tp = st.number_input("基准吞吐量 (辆):", value=float(b_current.get("throughput", 0.0)))
            b_adj_att = st.number_input("基准延时 ATT (秒):", value=float(b_current.get("avg_delay", 0.0)))
            
        user_comment = st.text_input("人工审查审批语:", value="仿真运行状态及数据合理，确认无误，批准主编撰写报告。")
        
        if st.button("🟢 批准并恢复运行 (Resume)", use_container_width=True):
            # 1. 更新基准数据
            b_updated = b_current.copy()
            b_updated["throughput"] = b_adj_tp
            b_updated["avg_delay"] = b_adj_att
            b_updated["raw_analysis"] += f"\n\n[人类审批意见]: {user_comment} (微调基准: 吞吐量 {b_adj_tp}，ATT {b_adj_att})"
            
            # 2. 更新对比数据
            o_updated = {}
            if o_current:
                o_updated = o_current.copy()
                o_updated["throughput"] = o_adj_tp
                o_updated["avg_delay"] = o_adj_att
                o_updated["raw_analysis"] += f"\n\n[人类审批意见]: (微调对比: 吞吐量 {o_adj_tp}，ATT {o_adj_att})"
                
            # 3. 更新 Graph 状态
            st.session_state.graph_app.update_state(
                config,
                {
                    "baseline_data": b_updated,
                    "optimized_data": o_updated if o_current else {},
                    "review_feedback": "PASS"
                },
                as_node="reviewer"
            )
            
            st.session_state.workflow_status = "running_second_half"
            st.rerun()

# C. 运行第二半 (Resume ➔ Editor ➔ End)
elif st.session_state.workflow_status == "running_second_half":
    with st.chat_message("assistant"):
        # 注入自动滚动脚本，保持流式输出时页面锚定在最下方，防止布局收缩导致页面滚动条跳回顶部
        st.components.v1.html(
            """
            <script>
                const mainSection = window.parent.document.querySelector('section.main');
                if (mainSection) {
                    mainSection.scrollTo(0, mainSection.scrollHeight);
                    const observer = new MutationObserver(() => {
                        mainSection.scrollTo(0, mainSection.scrollHeight);
                    });
                    observer.observe(window.parent.document.body, { childList: true, subtree: true });
                    setTimeout(() => observer.disconnect(), 45000);
                }
            </script>
            """,
            height=0,
        )
        # 静态展示第一阶段的日志（折叠状态框）
        if st.session_state.logs_first_half:
            with st.status("🤖 Multi-Agent 工作流执行进度 (第一阶段：数据分析与校验)", state="complete", expanded=False):
                st.markdown("\n".join(st.session_state.logs_first_half))
                
        # 渲染前半段已有的收敛对比趋势图
        if st.session_state.comparison_img and os.path.exists(st.session_state.comparison_img):
            st.image(st.session_state.comparison_img, caption="仿真收敛趋势对比折线图")
            
        # 实时渲染报告打字机流式输出的占位符（报告在最底下展现）
        report_area = st.empty()
        st.session_state.report_placeholder = report_area
        
        # 运行后半段工作流，并在单独的进度状态框中展示和流式追加第二阶段的日志
        with st.status("📝 报告分析编撰中...", expanded=True) as status_box:
            log_placeholder = st.empty()
            st.session_state.logs_placeholder = log_placeholder
            # 开启第二阶段全新的日志缓存，避免与第一阶段混淆
            st.session_state.logs = []
            log_placeholder.markdown("⏳ 工作流已激活：正在恢复执行第二阶段任务...")
            
            # 恢复运行
            for event in st.session_state.graph_app.stream(None, config):
                pass
                
            status_box.update(label="✅ 报告编撰与评估完成！", state="complete", expanded=False)
            
        state = st.session_state.graph_app.get_state(config)
        report = state.values.get("final_report", "")
        
        # 将本次对话记录并归档进历史
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": report,
            "img": st.session_state.comparison_img
        })
        st.session_state.workflow_status = "finished"
        st.rerun()

# D. 结束或空闲状态：渲染对话输入框
elif st.session_state.workflow_status in ["idle", "finished"]:
    user_input = st.chat_input("向智能交通助理提问，或指示诊断仿真日志...")
    if user_input:
        # 添加用户提问到对话历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.current_prompt = user_input
        st.session_state.workflow_status = "running_first_half"
        st.rerun()

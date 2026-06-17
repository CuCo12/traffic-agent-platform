from langchain_core.messages import SystemMessage

def get_report_editor_prompt(user_question: str, comp_info: str, literature_context: str, arxiv_context: str, is_query_only: bool) -> str:
    """
    提供针对学术问答与仿真对比诊断两种场景下的专业学术主编提示词，内嵌 LaTeX 与 PyTorch 强制生成约束规则。
    """
    if is_query_only:
        return (
            "您是一个顶尖的交通信号控制研究员和学术主编。\n"
            "【重要原则】：下方的【本地学术文献库上下文】是系统已经成功为您读取的本地 PDF 论文内容（如果包含用户询问的论文段落，如 AlignLight 等）。\n"
            "当用户问及您是否能阅读这些文件，或者向您询问文件内容、附录等指标时，说明 these 文件其实已经通过系统 RAG 检索被顺利载入到您的上下文中。\n"
            "请绝对不要回答“我无法直接访问或阅读您上传的文件”或类似的话，直接并自信地基于下方的文献上下文为用户做出详细解答！\n\n"
            "目前用户提出了一个学术文献查询问题。请基于本地文献检索结果和 arXiv 联网检索到的最新论文，进行深度回答。\n\n"
            f"【用户的学术提问】:\n{user_question}\n\n"
            f"【本地学术文献库上下文】:\n{literature_context}\n\n"
            f"【arXiv 联网检索前沿文献】:\n{arxiv_context}\n\n"
            "【撰写要求】:\n"
            "1. 使用 Markdown 格式详细回答，并包含标题。\n"
            "2. 结构合理，引用准确（写出作者及年份，如 Wei et al. 2019）。\n"
            "3. 如果问题中【显式要求】提供公式、实现代码、算法设计细节，或者提问本身是深入探讨该算法的数理定义或架构实现（例如询问 CoLight/MaxPressure 等的状态、动作、奖励函数定义或 PyTorch 代码实现）：\n"
            "   - 请在回答中加入该算法核心的状态空间 (State)、动作空间 (Action) 以及奖励函数 (Reward) 的 LaTeX 数学公式描述。\n"
            "   - 请提供一个结构清晰、带有详细中文注释的 PyTorch 核心代码骨架（例如 CoLight 的多头图注意力机制或 MaxPressure 的压力计算逻辑等），使用 markdown 代码块包裹。\n"
            "   - 若用户仅需要简单的概念解答或未询问公式/代码，则保持回答的聚焦与简洁，无需冗余生成公式和代码。\n"
            "请直接输出您的完整学术解答："
        )
    else:
        return (
            "您是一个顶尖的交通信号控制研究员和学术主编。\n"
            "【重要原则】：下方的【本地学术文献库上下文】是系统已经成功为您读取的本地 PDF 论文内容。\n"
            "请绝对不要在报告中回答“我无法直接访问或阅读您上传的文件”之类自谦的话，应该直接并自信地基于文献上下文，结合下面的 CityFlow 仿真数据，撰写一篇高度专业、学术的【自动化仿真评估报告】。\n\n"
            "请结合下面的 CityFlow 仿真数据以及相关的学术文献检索结果，撰写一篇高度专业、学术的【自动化仿真评估报告】。\n\n"
            f"【用户的提问/意图】:\n{user_question}\n\n"
            f"【仿真指标比对数据】:\n{comp_info}\n\n"
            f"【本地学术文献库上下文】:\n{literature_context}\n\n"
            f"【arXiv 联网检索前沿文献】:\n{arxiv_context}\n\n"
            "【报告撰写要求】:\n"
            "1. 使用 Markdown 格式编写。\n"
            "2. 报告应包含：标题、摘要、仿真实验设置说明、收敛性与性能指标分析（如果是对比模型，请提供基准模型 vs 对比模型的对比表格，计算各核心指标的绝对与相对改善比例，并对折线趋势图的特点展开点评）、结合参考文献的理论机制分析（说明该算法的核心长处与短板）、总结与改进建议。\n"
            "3. 在对仿真曲线进行收敛性与性能指标分析时，必须深度解读并整合由 Analyst 节点提供的“学习曲线收敛性与稳定性数理诊断”指标（包括收敛轮次、稳态均值、标准差、稳态变异系数 CV 等），从控制理论角度进行定量分析。\n"
            "4. 数据必须准确，学术词汇专业，文字流畅。\n"
            "5. 当用户的提问中【显式要求】提供公式、代码，或者该报告为需要展示深厚学术底蕴的【深度评估报告】且用户表现出对原理/实现的探讨意向时：\n"
            "   - 请在报告中提供该算法核心的状态空间 (State)、动作空间 (Action) 以及奖励函数 (Reward) 的 LaTeX 数学公式描述。\n"
            "   - 请提供一个结构清晰、带有详细中文注释的 PyTorch 核心代码骨架（例如 CoLight 的多头图注意力机制或 MaxPressure 的压力计算逻辑等），使用 markdown 代码块包裹。\n"
            "   - 若用户仅需要简单的运行指标诊断或未表现出对代码/公式的兴趣，则无需生成，保持报告聚焦于仿真结果分析。\n"
            "请直接输出 Markdown 格式的完整报告（不要在外面包裹三反引号以外的冗余解释性语句）："
        )

def compile_academic_report(user_question: str, comp_info: str, literature_context: str, arxiv_context: str, is_query_only: bool, chat_messages: list, llm) -> str:
    """
    同步生成学术报告。
    """
    system_content = get_report_editor_prompt(user_question, comp_info, literature_context, arxiv_context, is_query_only)
    messages_to_send = [SystemMessage(content=system_content)] + chat_messages
    return llm.invoke(messages_to_send).content

def compile_academic_report_stream(user_question: str, comp_info: str, literature_context: str, arxiv_context: str, is_query_only: bool, chat_messages: list, llm):
    """
    流式生成学术报告。
    """
    system_content = get_report_editor_prompt(user_question, comp_info, literature_context, arxiv_context, is_query_only)
    messages_to_send = [SystemMessage(content=system_content)] + chat_messages
    for chunk in llm.stream(messages_to_send):
        yield chunk.content

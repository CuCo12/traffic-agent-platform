import logging
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

def rewrite_query_for_rag(chat_messages: list, user_question: str, llm) -> str:
    """
    认知技能：跨语言学术意图检索词重写。
    根据多轮历史会话，将用户的中文/口语提问重写为纯英文学术检索关键词，以解决中英文 RAG 匹配精度差的问题。
    """
    if not chat_messages or len(chat_messages) <= 1:
        return user_question

    history_str = ""
    # 提取最后消息前的历史片段
    for msg in chat_messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content_snippet = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
        history_str += f"{role}: {content_snippet}\n"

    rewrite_prompt = (
        "您是一个学术检索词重写助手。\n"
        "请根据以下对话历史和用户最新的问题，为本地学术文献库生成一个最合适、最精准的英文检索词或短语（由于本地文献均为英文论文，因此检索词必须全部翻译为英文，例如将“推理时间”重写为“inference latency”或“inference time”）。\n"
        "注意：必须保留提及的核心算法名（如 AlignLight, CoLight 等），且只需输出最终的英文检索词，不要有任何前缀、解释、标点或多余文字。\n\n"
        f"【对话历史】:\n{history_str}\n"
        f"【用户最新问题】: \"{user_question}\"\n\n"
        "最精准英文检索词:"
    )

    try:
        response = llm.invoke(rewrite_prompt)
        rewritten_query = response.content.strip().strip('"').strip("'")
        if rewritten_query:
            return rewritten_query
    except Exception as e:
        logger.warning(f"Failed to rewrite query via LLM: {e}")
        
    return user_question

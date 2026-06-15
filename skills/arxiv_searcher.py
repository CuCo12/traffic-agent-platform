import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from langchain.tools import tool

@tool
def search_latest_arxiv_papers(query: str) -> str:
    """
    联网检索 arXiv 学术知识库中最新相关的学术论文。
    当用户询问“最新研究”、“前沿论文”，或者本地文献库无法回答时调用此工具。
    由于是国际知识库，请大模型自动将用户的中文搜索意图翻译并构建为精准的英文 arXiv 检索语句作为 query 参数。
    必须使用双引号包裹短语，并用 AND 连接不同的核心概念，以避免返回不相关的宽泛文献。
    例如，不要直接传 "traffic signal control reinforcement learning"，而是必须构建为：
    'all:"traffic signal control" AND all:"reinforcement learning"'
    或者：
    'all:"traffic light control" AND all:"deep reinforcement learning"'
    """
    # 1. 格式化查询关键词 (进行 URL 编码，并包含保底逻辑)
    clean_query = query.strip()
    # 如果大模型没有按格式输出（即不包含 any, all, AND 等条件），我们自动将其封装为更精确的双引号短语查询
    if not (clean_query.startswith("all:") or clean_query.startswith("ti:") or "AND" in clean_query or "OR" in clean_query):
        # 针对常见交通控制与强化学习短语进行智能拼接保底
        lower_query = clean_query.lower()
        if "reinforcement learning" in lower_query and "traffic signal" in lower_query:
            clean_query = 'all:"traffic signal control" AND all:"reinforcement learning"'
        elif "reinforcement learning" in lower_query and "traffic light" in lower_query:
            clean_query = 'all:"traffic light control" AND all:"reinforcement learning"'
        else:
            # 默认作为精确短语查询
            clean_query = f'all:"{clean_query}"'
            
    formatted_query = urllib.parse.quote(clean_query)
    
    # 2. 构建 arXiv API 请求 URL 
    # 设定按提交时间倒序 (sortBy=submittedDate&sortOrder=descending)，取前 3 篇最新论文
    url = f'http://export.arxiv.org/api/query?search_query={formatted_query}&start=0&max_results=3&sortBy=submittedDate&sortOrder=descending'
    
    try:
        # 3. 发送请求并读取 XML 数据
        response = urllib.request.urlopen(url, timeout=10)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        # 4. 解析 XML 提取论文信息
        ns = {'arxiv': 'http://www.w3.org/2005/Atom'} # arXiv 的命名空间
        results = []
        
        for entry in root.findall('arxiv:entry', ns):
            title = entry.find('arxiv:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('arxiv:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('arxiv:published', ns).text.strip()
            # 提取所有作者
            authors = [author.find('arxiv:name', ns).text for author in entry.findall('arxiv:author', ns)]
            
            # 组装单篇论文的结构化信息
            paper_info = (
                f"📄 【标题】: {title}\n"
                f"👥 【作者】: {', '.join(authors)}\n"
                f"📅 【发布时间】: {published[:10]}\n"
                f"📝 【摘要】: {summary}\n"
            )
            results.append(paper_info)
        
        if not results:
            return f"ℹ️ 未在 arXiv 联网库中找到与 '{query}' 相关的论文。"
        
        # 5. 拼装最终的观测结果给 Agent
        return "🌐 【arXiv 联网检索结果】(按时间倒序排布):\n\n" + "\n---\n".join(results)
        
    except Exception as e:
        return f"❌ 联网检索 arXiv 失败: {e}"
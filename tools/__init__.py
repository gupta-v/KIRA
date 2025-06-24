from .arxiv import arxiv_tool
from .wikipedia import wikipedia_tool
from .tavily import tavily_tool
from .web_scraper import web_scraper_tool
from .open_link import open_link_tool
from .rag_tool import make_rag_tool

def get_all_tools(rag_tool_instance=None):
    """
    Returns a list of all available tools.
    If a RAG tool instance is provided, it's prioritized (placed first).
    """
    all_tools = [
        arxiv_tool,
        tavily_tool,
        wikipedia_tool,
        web_scraper_tool,
        open_link_tool
    ]
    if rag_tool_instance:
        all_tools = [rag_tool_instance] + all_tools
    return all_tools
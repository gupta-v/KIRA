import logging
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.tools import Tool

# Configure logging
logger = logging.getLogger(__name__)

# --- Tool Setup ---
arxiv_api = ArxivAPIWrapper(
    top_k_results=2,
    ARXIV_MAX_QUERY_LENGTH=100,
    load_max_docs=2,
    load_all_available_meta=False,
    doc_content_chars_max=3000
)

def arxiv_search(query: str) -> str:
    """Search academic papers from Arxiv with improved error handling."""
    try:
        logger.info(f"ArXiv search for: {query}")
        clean_query = query.strip().replace('"', '').replace("'", "")
        result = arxiv_api.run(clean_query)
        logger.info("ArXiv search completed successfully.")
        return result
    except Exception as e:
        logger.error(f"ArXiv search error: {e}")
        return f"ArXiv search failed for '{query}'. Error: {str(e)}"

# --- Tool Definition ---
arxiv_tool = Tool(
    name="arxiv_search",
    func=arxiv_search,
    description="Search academic papers from Arxiv. Use simple keywords like 'quantum computing' or 'machine learning'."
)
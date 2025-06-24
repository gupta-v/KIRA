import logging
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool

# Configure logging
logger = logging.getLogger(__name__)

# --- Tool Setup ---
wiki_api = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        doc_content_chars_max=4000,
        top_k_results=2
    )
)

def wiki_search(query: str) -> str:
    """Search Wikipedia with improved error handling."""
    try:
        logger.info(f"Wikipedia search for: {query}")
        clean_query = query.strip().replace('"', '').replace("'", "")
        result = wiki_api.run(clean_query)
        logger.info("Wikipedia search completed successfully.")
        return result
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return f"Wikipedia search failed for '{query}'. Error: {str(e)}"

# --- Tool Definition ---
wikipedia_tool = Tool(
    name="wikipedia_search",
    func=wiki_search,
    description="Search Wikipedia articles for general knowledge. Use simple terms like 'quantum computing' or 'artificial intelligence'."
)
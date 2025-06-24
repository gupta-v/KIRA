import logging
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool

# Configure logging
logger = logging.getLogger(__name__)

# --- Tool Setup ---
try:
    tavily_api = TavilySearch(
        max_results=2,
        topic="general",
        include_answer=True,
        include_raw_content=True,
    )
except Exception as e:
    logger.error(f"Tavily initialization error: {e}")
    tavily_api = None

def tavily_search(query: str) -> str:
    """Search the web using Tavily with improved error handling."""
    try:
        if tavily_api is None:
            return "Tavily search service is not available."

        logger.info(f"Tavily search for: {query}")
        clean_query = query.strip()
        result = tavily_api.invoke(clean_query)
        logger.info("Tavily search completed successfully.")

        if isinstance(result, str):
            return result
        elif isinstance(result, list) and result:
            formatted_results = []
            for item in result[:2]:
                if isinstance(item, dict):
                    title = item.get('title', 'No title')
                    content = item.get('content', item.get('snippet', 'No content'))
                    formatted_results.append(f"Title: {title}\nContent: {content}\n")
            return "\n".join(formatted_results) if formatted_results else str(result)
        else:
            return str(result)

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Web search failed for '{query}'. Error: {str(e)}"

# --- Tool Definition ---
tavily_tool = Tool(
    name="tavily_search",
    func=tavily_search,
    description="Search the web for general information and recent topics. Use simple queries."
)
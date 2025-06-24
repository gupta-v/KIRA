from langchain.agents import initialize_agent, AgentType
from .agent_utils import LLMS, get_logger

def create_research_agent(tools):
    """Create a research agent with the provided tools."""
    logger = get_logger()
    if not tools:
        logger.warning("No tools provided for the research agent.")
        return None

    try:
        return initialize_agent(
            tools=tools,
            llm=LLMS['researcher'].llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,  # Increased for potentially more complex research
            early_stopping_method="generate"
        )
    except Exception as e:
        logger.error(f"Failed to create research agent: {e}")
        return None

def run_research(agent, query: str, context: str = "", rag_tool=None, rag_enabled=False) -> str:
    """
    Runs the research agent to gather information on a query, but uses RAG tool first if available.
    """
    logger = get_logger()
    logger.info(f"Executing research for query: {query[:100]}...")

    # Build enhanced query with memory context
    if context:
        enhanced_query = f"Previous conversation context:\n{context}\n\nCurrent query: {query}"
    else:
        enhanced_query = query

    # Strictly use RAG tool if available
    if rag_enabled and rag_tool:
        logger.info("Using RAG tool for retrieval...")
        try:
            rag_content = rag_tool.func(enhanced_query)
            if len(rag_content) > 200 and "No relevant information found" not in rag_content:
                logger.info(f"RAG research successful: {len(rag_content)} chars")
                return rag_content
            else:
                logger.info("RAG results insufficient")
        except Exception as e:
            logger.warning(f"RAG tool failed: {e}")

    # Only use agent fallback if RAG is not available or insufficient
    if not agent:
        return "Research agent is not available."

    try:
        # Combine query and context for a more informed search
        full_query = f"Query: {query}\n\nContext: {context}" if context else query
        result = agent.invoke({"input": full_query})
        return result.get("output", "No information found.")
    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        return f"Error during research: {e}"
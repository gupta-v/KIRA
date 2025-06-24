import time
from graph.graph_builder import build_graph
from graph.helper_functions.logger import get_logger

def run_tutor(query: str, user_id: str = "default_user", rag_enabled: bool = False, rag_tool=None) -> dict:
    """Run the optimized AI tutor system"""
    logger = get_logger()
    logger.info("="*60)
    logger.info(f"🚀 STARTING TUTOR SESSION")
    logger.info(f"Query: {query}")
    logger.info(f"User ID: {user_id}")
    logger.info("="*60)

    session_start_time = time.time()

    try:
        graph = build_graph()

        initial_state = {
            "query": query,
            "user_id": user_id,
            "original_query": None,
            "resolved_query": None,
            "topic": None,
            "memory_context": None,
            "research_summary": None,
            "research_complete": False,
            "draft": None,
            "verification_passed": False,
            "verifier_feedback": None,
            "need_more_research": False,
            "research_attempts": 0,
            "draft_attempts": 0,
            "final_output": None,
            "rag_enabled": rag_enabled,
            "rag_tool": rag_tool
        }

        final_state = graph.invoke(initial_state)

        session_duration = time.time() - session_start_time
        logger.info(f"SESSION COMPLETED in {session_duration:.2f}s")
        logger.info("="*60)

        return final_state

    except Exception as e:
        session_duration = time.time() - session_start_time
        logger.error(f"SESSION FAILED after {session_duration:.2f}s: {e}")

        fallback_output = f"I apologize, but I encountered a system error while processing your query: '{query}'\nError details: {str(e)}"
        return {
            "final_output": fallback_output,
            "error": True,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
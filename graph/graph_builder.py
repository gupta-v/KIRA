from langgraph.graph import StateGraph, START, END
from state.context import TutorState
from .nodes import memory_retrieval_node, researcher_node, drafter_node, verifier_node, enhancer_node
from .helper_functions.logger import get_logger

def should_continue_research(state: TutorState) -> str:
    """Decide if more research is needed"""
    logger = get_logger()

    need_more = state.get("need_more_research", False)
    attempts = state.get("research_attempts", 0)

    logger.info(f"Research decision - Need more: {need_more}, Attempts: {attempts}/2")

    if need_more and attempts < 2:
        logger.info("Continuing to researcher")
        return "researcher"
    else:
        logger.info("Proceeding to verifier")
        return "verifier"

def should_continue_verification(state: TutorState) -> str:
    """Decide if draft needs revision"""
    logger = get_logger()

    verified = state.get("verification_passed", False)
    attempts = state.get("draft_attempts", 0)

    logger.info(f"Verification decision - Verified: {verified}, Attempts: {attempts}/2")

    if not verified and attempts < 2:
        logger.info("Continuing to drafter for revision")
        return "drafter"
    else:
        logger.info("Proceeding to enhancer")
        return "enhancer"

def build_graph():
    """Build the optimized tutoring graph"""
    logger = get_logger()
    logger.info("🏗️ Building optimized graph")

    try:
        workflow = StateGraph(TutorState)

        # Add nodes
        workflow.add_node("memory_retrieval", memory_retrieval_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("drafter", drafter_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("enhancer", enhancer_node)

        # Set up workflow edges
        workflow.add_edge(START, "memory_retrieval")
        workflow.add_edge("memory_retrieval", "researcher")
        workflow.add_edge("researcher", "drafter")

        # Conditional edges
        workflow.add_conditional_edges(
            "drafter",
            should_continue_research,
            {
                "researcher": "researcher",
                "verifier": "verifier"
            }
        )

        workflow.add_conditional_edges(
            "verifier",
            should_continue_verification,
            {
                "drafter": "drafter",
                "enhancer": "enhancer"
            }
        )

        workflow.add_edge("enhancer", END)

        compiled_graph = workflow.compile()
        logger.info("Graph built successfully")
        return compiled_graph

    except Exception as e:
        logger.error(f"Graph building failed: {e}")
        raise
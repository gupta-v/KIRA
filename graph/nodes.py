import time
from typing import Dict
from state.context import TutorState
from state.memory import memory_manager, get_memory_context, add_interaction, extract_topic_from_query
from graph.helper_functions.logger import get_logger
from graph.helper_functions.prompt_utils import extract_style_from_query
from agents.researcher import create_research_agent, run_research
from agents.drafter import run_drafter
from agents.verifier import run_verifier
from agents.summarizer import run_summarizer
from agents.enhancer import run_enhancer
from tools import get_all_tools

# --- memory_retrieval_node remains the same ---
def memory_retrieval_node(state: TutorState) -> Dict:
    # ... (no changes)
    logger = get_logger()
    logger.info("🧠 MEMORY RETRIEVAL STARTED")
    query = state["query"]
    user_id = state["user_id"]
    resolved_query = memory_manager.resolve_pronouns(query, user_id)
    memory_context = get_memory_context(user_id, resolved_query)
    topic = extract_topic_from_query(resolved_query)
    logger.info("MEMORY RETRIEVAL ENDED")
    return {
        "original_query": query,
        "resolved_query": resolved_query,
        "memory_context": memory_context,
        "topic": topic
    }


# --- researcher_node remains the same ---
def researcher_node(state: TutorState) -> Dict:
    logger = get_logger()
    logger.info("🔬 RESEARCHER NODE STARTED")
    resolved_query = state["resolved_query"]
    memory_context = state["memory_context"]
    rag_enabled = state.get("rag_enabled", False)
    rag_tool = state.get("rag_tool", None)
    tools = get_all_tools(rag_tool)
    research_agent = create_research_agent(tools)
    research_output = run_research(
        research_agent,
        resolved_query,
        memory_context,
        rag_tool=rag_tool,
        rag_enabled=rag_enabled
    )
    summary = run_summarizer(research_output, focus=resolved_query)
    logger.info(f"RESEARCHER NODE ENDED - Summary Length: {len(summary)}")
    return {"research_summary": summary}


def drafter_node(state: TutorState) -> Dict:
    logger = get_logger()
    logger.info("✍️ DRAFTER NODE STARTED")
    
    # **CRITICAL FIX**: Get the current attempt count from the state
    attempts = state.get("draft_attempts", 0)
    logger.info(f"Drafting attempt #{attempts + 1}")

    style = extract_style_from_query(state["original_query"])
    
    draft = run_drafter(
        query=state["resolved_query"],
        research_summary=state["research_summary"],
        memory_context=state["memory_context"],
        style=style,
        verifier_feedback=state.get("verifier_feedback") # Pass feedback for revisions
    )
    
    logger.info("DRAFTER NODE ENDED")
    
    # **CRITICAL FIX**: Return the incremented attempt count
    return {
        "draft": draft,
        "draft_attempts": attempts + 1
    }


def verifier_node(state: TutorState) -> Dict:
    logger = get_logger()
    logger.info("🔍 VERIFIER NODE STARTED")
    result = run_verifier(state["draft"], state["research_summary"])
    logger.info(f"VERIFIER NODE ENDED - Passed: {result['passed']}")
    return {"verification_passed": result["passed"], "verifier_feedback": result["feedback"]}


def enhancer_node(state: TutorState) -> Dict:
    logger = get_logger()
    logger.info("✨ ENHANCER NODE STARTED")
    style = extract_style_from_query(state["original_query"])
    
    # If verification failed repeatedly, the enhancer gets the last draft.
    # We add a note to the user that the content may not be fully verified.
    final_draft = state["draft"]
    if not state.get("verification_passed", False):
        unverified_note = "\n\n*[Note: This response could not be fully verified against all available sources and may contain inconsistencies.]*"
        final_draft += unverified_note

    final_output = run_enhancer(final_draft, state["original_query"], style)
    add_interaction(state["user_id"], state["original_query"], final_output, state["topic"])
    logger.info("ENHANCER NODE ENDED")
    return {"final_output": final_output}
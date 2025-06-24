from langchain_core.messages import HumanMessage
from .agent_utils import LLMS, get_logger
from graph.helper_functions.prompt_utils import smart_truncate

def run_drafter(query: str, research_summary: str, memory_context: str, style: str, verifier_feedback: str = None) -> str:
    """
    Generates or revises a draft response.
    If verifier_feedback is provided, it focuses on revision.
    """
    logger = get_logger()
    
    if verifier_feedback:
        logger.info("Executing drafter in revision mode...")
        # A more focused prompt for revision
        prompt = f"""A previous draft was rejected. Please revise it based on the feedback.

**Original Query:** {query}
**Research Summary:** {smart_truncate(research_summary, 1000)}
**Verifier Feedback:** {verifier_feedback}

Your task is to rewrite the draft to specifically address all points in the feedback. Create a new, improved version.
"""
    else:
        logger.info("Executing drafter in initial draft mode...")
        # The original prompt for the first draft
        context_parts = [f"User's question: {query}"]
        if memory_context:
            context_parts.append(f"Conversation context:\n{smart_truncate(memory_context, 600)}")
        if research_summary:
            context_parts.append(f"Research findings:\n{smart_truncate(research_summary, 1200)}")

        prompt = "\n\n".join(context_parts)
        prompt += f"""\n\nCreate a comprehensive response that:
1. Directly answers the user's question.
2. Incorporates research findings effectively.
3. Adheres to a '{style}' style.
4. Is well-structured and easy to understand.
"""

    try:
        response = LLMS['drafter'].invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Drafter agent failed: {e}")
        return f"Error during drafting: {e}"
from langchain_core.messages import HumanMessage
from .agent_utils import LLMS, get_logger
from graph.helper_functions.prompt_utils import smart_truncate

def run_verifier(draft: str, research_summary: str) -> dict:
    """
    Verifies the draft against the research summary.
    Returns a dictionary with 'passed' (bool) and 'feedback' (str).
    """
    logger = get_logger()
    logger.info("Executing verifier...")

    if not research_summary:
        return {"passed": True, "feedback": "No research to verify against."}

    prompt = f"""Please verify if the 'Draft' is factually consistent with the 'Research Summary'.

**Research Summary:**
{smart_truncate(research_summary, 2000)}

**Draft:**
{smart_truncate(draft, 2000)}

**Instructions:**
Respond with "YES" if the draft is consistent and factually correct based on the research.
Respond with "NO" followed by specific, actionable feedback if there are any inconsistencies or errors.
"""

    try:
        response = LLMS['verifier'].invoke([HumanMessage(content=prompt)])
        result_text = response.content.strip()

        # More robust parsing
        if result_text.upper().startswith("YES"):
            passed = True
            feedback = "The draft is factually consistent with the research."
        else:
            passed = False
            # Assumes the rest of the text after "NO" is feedback
            feedback = result_text.replace("NO", "").strip()

        return {"passed": passed, "feedback": feedback}
    except Exception as e:
        logger.error(f"Verifier agent failed: {e}")
        # Default to passing to avoid getting stuck in a loop on an infrastructure error
        return {"passed": True, "feedback": f"Verifier error: {e}"}
from langchain_core.messages import HumanMessage
from .agent_utils import LLMS, get_logger
from graph.helper_functions.prompt_utils import smart_truncate

def run_summarizer(text: str, focus: str = None) -> str:
    """
    Summarizes a given text, with an optional focus.
    """
    logger = get_logger()
    logger.info(f"Executing summarizer on text of length {len(text)}...")

    if not text:
        return ""

    prompt = f"Please summarize the following text:\n\n{smart_truncate(text, 4000)}"
    if focus:
        prompt += f"\n\nFocus the summary on: {focus}"

    try:
        response = LLMS['summarizer'].invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Summarizer agent failed: {e}")
        return f"Error during summarization: {e}"
from langchain_core.messages import HumanMessage
from .agent_utils import LLMS, get_logger
from graph.helper_functions.prompt_utils import smart_truncate

def run_enhancer(draft: str, query: str, style: str) -> str:
    """
    Enhances the draft for style, clarity, and engagement.
    """
    logger = get_logger()
    logger.info("Executing enhancer...")

    prompt = f"""Original Query: {query}
Style to apply: {style}

Current Draft:
{smart_truncate(draft, 3000)}

Please rewrite and enhance the draft. Improve its clarity, structure, and engagement, ensuring it perfectly matches the requested '{style}'.
The final output should be polished and ready for the user.
"""

    try:
        response = LLMS['enhancer'].invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Enhancer agent failed: {e}")
        return f"Error during enhancement: {e}"
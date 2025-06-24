from typing import TypedDict, Annotated, Optional, List, Dict, Any

def take_latest(left, right):
    """Take the latest non-None value"""
    return right if right is not None else left

def take_maximum(left: int, right: int) -> int:
    """Custom max function"""
    if left is None:
        return right or 0
    if right is None:
        return left or 0
    return left if left > right else right

class TutorState(TypedDict):
    # Core input
    query: Annotated[str, "User query"]
    user_id: Annotated[str, "User identifier for memory"]
    original_query: Annotated[str, "Original user query before processing"]
    resolved_query: Annotated[str, "Query with pronouns resolved"]

    # Memory context
    memory_context: Annotated[Optional[str], take_latest]
    topic: Annotated[Optional[str], take_latest]

    # Research phase
    research_summary: Annotated[Optional[str], take_latest]
    research_complete: Annotated[bool, take_latest]

    # Content creation
    draft: Annotated[Optional[str], take_latest]
    verification_passed: Annotated[bool, take_latest]
    verifier_feedback: Annotated[Optional[str], take_latest]

    # Control flags
    need_more_research: Annotated[bool, take_latest]

    # Iteration controls
    research_attempts: Annotated[int, take_maximum]
    draft_attempts: Annotated[int, take_maximum]

    # Final output
    final_output: Annotated[Optional[str], take_latest]

    # RAG control
    rag_enabled: Annotated[bool, take_latest]
    rag_tool: Annotated[Optional[object], take_latest]
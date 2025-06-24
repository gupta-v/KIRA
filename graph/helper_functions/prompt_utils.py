import re

def smart_truncate(text: str, max_length: int, preserve_start: int = 500) -> str:
    """Smart truncation that preserves beginning and important content"""
    if len(text) <= max_length:
        return text

    # Always preserve the beginning (most important context)
    if max_length <= preserve_start:
        return text[:max_length]

    # Take from start and end
    start_chunk = text[:preserve_start]
    remaining_length = max_length - preserve_start - 100  # Buffer for separator

    if remaining_length > 0:
        end_chunk = text[-remaining_length:]
        return f"{start_chunk}\n\n[... content truncated ...]\n\n{end_chunk}"
    else:
        return start_chunk

def extract_style_from_query(query: str) -> str:
    """Extract style hints from the query"""
    query_lower = query.lower()

    # Style patterns to look for
    style_patterns = {
        r'simple.*explain|explain.*simple': 'simple and clear',
        r'detailed.*technical|technical.*detailed': 'detailed and technical',
        r'step.*by.*step': 'step-by-step',
        r'beginner.*friendly|friendly.*beginner': 'beginner-friendly',
    }

    for pattern, style in style_patterns.items():
        if re.search(pattern, query_lower):
            return style

    return "educational and engaging"
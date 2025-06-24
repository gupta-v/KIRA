import webbrowser
from langchain_core.tools import Tool

def open_link_in_browser(url: str) -> str:
    """Open a URL in the default web browser."""
    try:
        webbrowser.open(url)
        return f"Successfully opened link: {url}"
    except Exception as e:
        return f"Failed to open link: {url}. Error: {str(e)}"

open_link_tool = Tool(
    name="open_link",
    func=open_link_in_browser,
    description="Open a URL in the default web browser. Input must be a valid URL."
)
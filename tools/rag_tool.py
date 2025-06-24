import os
import sys
from typing import Any
from langchain_core.tools import Tool

# Ensure RAG pipeline is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.rag_pipeline import load_index, retrieve_answer, save_index, index_pdfs

def make_rag_tool(vector_store=None) -> Tool:
    """
    Factory function to create a RAG tool with a persistent vector store state.
    """
    state = {'vector_store': vector_store}

    def rag_action_fn(input_data: Any) -> str:
        if isinstance(input_data, str):
            # Default action: retrieve
            if not state['vector_store']:
                return "RAG Error: Document index not loaded."
            return retrieve_answer(input_data, state['vector_store'])

        elif isinstance(input_data, dict):
            action = input_data.get('action', 'retrieve')
            query = input_data.get('query', '')

            if action == 'retrieve':
                if not state['vector_store']:
                    return "RAG Error: Document index not loaded."
                return retrieve_answer(query, state['vector_store'])

            elif action == 'add':
                pdf_path = input_data.get('pdf_path')
                if not pdf_path:
                    return "RAG Error: No PDF path provided for 'add' action."
                new_store = index_pdfs(pdf_path)
                if new_store:
                    state['vector_store'] = new_store
                    return f"Successfully indexed and loaded PDF: {os.path.basename(pdf_path)}"
                else:
                    return f"Failed to index PDF: {pdf_path}"

            elif action == 'save':
                save_path = input_data.get('save_path', 'data/index/rag_index')
                if not state['vector_store']:
                    return "RAG Error: No vector store to save."
                save_index(state['vector_store'], save_path)
                return f"Index successfully saved to {save_path}"

            else:
                return f"RAG Error: Unknown action '{action}'."
        else:
            return "RAG Error: Invalid input type."

    return Tool(
        name="rag_tool",
        func=rag_action_fn,
        description=(
            "Interfaces with a RAG pipeline. Can retrieve answers from indexed documents (default), "
            "add new PDFs to the index (action='add'), or save the index (action='save')."
        )
    )
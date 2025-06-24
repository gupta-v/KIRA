import os
import argparse
from tutor import run_tutor
from voice_assist.speech_to_text import speech_to_text
from voice_assist.text_to_speech import say
from graph.helper_functions.logger import setup_logging
from state.memory import memory_manager
from RAG.rag_pipeline import index_pdfs, save_index, load_index
from tools.rag_tool import make_rag_tool


def interactive_session():
    """Run an interactive AI tutor session with memory"""
    rag_enabled = False
    rag_tool_instance = None

    print("\n" + "="*60)
    print("AI TUTOR WITH RAG CAPABILITIES & MEMORY")
    print("="*60)
    print("\nDo you want to use RAG capabilities?")
    print("1. Yes - Add a new document to index")
    print("2. No - Use standard research tools only")

    rag_choice = input("\nEnter your choice: ").strip()

    if rag_choice in ["1"]:
        try:
            if rag_choice == "1":
                print("\n" + "-"*60)
                print("DOCUMENT INDEXING")
                print("-"*60)

                pdf_path = input(
                    "\nEnter the path to your PDF document: ").strip()
                if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
                    print(f"\nIndexing document: {pdf_path}...")
                    os.makedirs("data/index", exist_ok=True)
                    vector_store = index_pdfs(pdf_path)
                    if vector_store:
                        save_index(vector_store, "data/index/rag_index")
                        rag_tool_instance = make_rag_tool(vector_store)
                        rag_enabled = True
                        print("\n✅ Document indexed successfully!")
                    else:
                        print(
                            "\n❌ Failed to index document. Proceeding without RAG capabilities.")
                else:
                    print("\n❌ Invalid PDF path. Proceeding without RAG capabilities.")

        except Exception as e:
            print(f"\n❌ Error with RAG setup: {e}")
            print("Proceeding without RAG capabilities.")

    # Setup memory
    if memory_manager.initialized:
        print("\n✅ Memory system initialized!")
    else:
        print("\n⚠️ Memory system not available")

    # Get user ID
    user_id = input(
        "\nEnter your user ID (or press Enter for 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"

    print("\n" + "-"*60)
    print(f"AI TUTOR SESSION - User: {user_id} (type 'exit' to quit)")
    print("-"*60)

    log_configured = False
    session_count = 0

    while True:
        print("\nChoose your AI tutor mode:\n1. Text Input - Ask questions using text input (Default)\n2. Voice Input - Ask questions using voice input")
        choice = (input("\nEnter your choice (1 or 2): ").strip())
        if choice in ["2"]:
            print("\nVoice Input Mode Activated")
            print(f"\n[{session_count + 1}] Please ask your question: ")
            voice_input = speech_to_text().strip()
            user_query = voice_input if voice_input else input(
                "No speech detected. Please type your question: ").strip()
        else:
            user_query = input(
                f"\n[{session_count + 1}] Enter your question: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting tutor session. Goodbye!")
            break

        if not user_query:
            continue

        if not log_configured:
            setup_logging(user_query)
            log_configured = True

        session_count += 1

        # Run tutor with memory
        result = run_tutor(
            user_query,
            user_id=user_id,
            rag_enabled=rag_enabled,
            rag_tool=rag_tool_instance
        )

        # Display response
        ai_response = result.get("final_output", "No output generated")
        print("\n" + "="*60)
        print("TUTOR RESPONSE:")
        print("="*60)
        print(ai_response)
        print("="*60)
        if choice in ["1"]:
            say(ai_response)

        # Show error details if any
        if result.get("error"):
            print(f"\n⚠️ Error Type: {result.get('error_type', 'Unknown')}")
            print(
                f"Error Message: {result.get('error_message', 'No details')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Tutor.")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode.")
    args = parser.parse_args()

    if args.interactive:
        try:
            interactive_session()
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Not able to run the agent graph")

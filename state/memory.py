import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️ Mem0 not available. Install with: pip install mem0ai")

class SimplifiedMemoryManager:
    """Simplified memory manager with long-term and short-term memory"""

    def __init__(self):
        self.client = None
        self.initialized = False
        self.session_memory = {}  # Short-term memory per session
        self._setup_client()

    def _setup_client(self):
        """Initialize Mem0 client"""
        if not MEM0_AVAILABLE:
            return

        try:
            api_key = os.getenv("MEM_AI_API_KEY")
            if not api_key:
                print("❌ MEM_AI_API_KEY not found in environment variables")
                return

            self.client = MemoryClient(api_key=api_key)
            self.initialized = True
            print("✅ Mem0 client initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize Mem0 client: {e}")
            self.initialized = False

    def add_to_session(self, user_id: str, query: str, response: str, topic: str = None):
        """Add to short-term session memory"""
        if user_id not in self.session_memory:
            self.session_memory[user_id] = []

        interaction = {
            "query": query,
            "response": response,
            "topic": topic or "general",
            "timestamp": datetime.now().isoformat()
        }

        self.session_memory[user_id].append(interaction)

        # Keep only last 5 interactions for short-term memory
        if len(self.session_memory[user_id]) > 5:
            self.session_memory[user_id] = self.session_memory[user_id][-5:]

    def add_to_longterm(self, user_id: str, query: str, response: str, topic: str = None):
        """Add to long-term memory (Mem0)"""
        if not self.initialized:
            return False

        try:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]

            metadata = {
                "topic": topic or "general",
                "timestamp": datetime.now().isoformat()
            }

            self.client.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata
            )

            print(f"✅ Added to long-term memory for user {user_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to add to long-term memory: {e}")
            return False

    def get_session_context(self, user_id: str, current_query: str) -> str:
        """Get short-term session context"""
        if user_id not in self.session_memory or not self.session_memory[user_id]:
            return ""

        # Get last few interactions
        recent_interactions = self.session_memory[user_id][-3:]  # Last 3 interactions

        context_parts = []
        for interaction in recent_interactions:
            context_parts.append(f"Q: {interaction['query']}")
            context_parts.append(f"A: {interaction['response'][:200]}...")  # Truncate response

        # Add current topic context for pronoun resolution
        if recent_interactions:
            last_topic = recent_interactions[-1]['topic']
            context_parts.append(f"Current topic context: {last_topic}")

        return "\n".join(context_parts)

    def get_longterm_context(self, user_id: str, query: str, limit: int = 3) -> str:
        """Get relevant long-term context"""
        if not self.initialized:
            return ""

        try:
            memories = self.client.search(
                query=query,
                user_id=user_id,
                limit=limit
            )

            if not memories:
                return ""

            context_parts = []
            for memory in memories:
                memory_content = memory.get('memory', '')
                score = memory.get('score', 0)

                if score >= 0.35 and memory_content:
                    context_parts.append(f"- {memory_content}")

            if context_parts:
                return "Relevant past conversations:\n" + "\n".join(context_parts)
            else:
                return ""

        except Exception as e:
            print(f"❌ Failed to get long-term context: {e}")
            return ""

    def get_combined_context(self, user_id: str, query: str) -> str:
        """Get both short-term and long-term context combined"""
        session_context = self.get_session_context(user_id, query)
        longterm_context = self.get_longterm_context(user_id, query)

        context_parts = []

        if session_context:
            context_parts.append(f"Recent conversation:\n{session_context}")

        if longterm_context:
            context_parts.append(f"\n{longterm_context}")

        return "\n\n".join(context_parts) if context_parts else ""

    def resolve_pronouns(self, query: str, user_id: str) -> str:
        """Resolve pronouns like 'it', 'this', 'that' using session context"""
        query_lower = query.lower()
        pronouns = ['it', 'this', 'that', 'them', 'they']

        # Check if query contains pronouns
        has_pronouns = any(pronoun in query_lower for pronoun in pronouns)

        if not has_pronouns or user_id not in self.session_memory:
            return query

        # Get the last topic/subject discussed
        recent_interactions = self.session_memory[user_id]
        if not recent_interactions:
            return query

        last_interaction = recent_interactions[-1]
        last_topic = last_interaction.get('topic', 'general')
        last_query = last_interaction.get('query', '')

        # Extract main subject from last query (simple approach)
        if last_topic != 'general':
            resolved_query = f"[Context: referring to {last_topic}] {query}"
        else:
            # Try to extract subject from last query
            words = last_query.split()
            subjects = [word for word in words if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where']]
            if subjects:
                resolved_query = f"[Context: referring to {subjects[0]}] {query}"
            else:
                resolved_query = query

        return resolved_query

# Global memory manager instance
memory_manager = SimplifiedMemoryManager()

def add_interaction(user_id: str, query: str, response: str, topic: str = None):
    """Add interaction to both short-term and long-term memory"""
    memory_manager.add_to_session(user_id, query, response, topic)
    memory_manager.add_to_longterm(user_id, query, response, topic)

def get_memory_context(user_id: str, query: str) -> str:
    """Get combined memory context with pronoun resolution"""
    resolved_query = memory_manager.resolve_pronouns(query, user_id)
    return memory_manager.get_combined_context(user_id, resolved_query)

def extract_topic_from_query(query: str) -> str:
    """Extract main topic from query for memory categorization"""
    query_lower = query.lower()

    topic_keywords = {
        'math': ['math', 'calculus', 'algebra', 'geometry', 'statistics', 'equation'],
        'science': ['physics', 'chemistry', 'biology', 'science', 'molecule', 'atom'],
        'programming': ['code', 'python', 'javascript', 'programming', 'algorithm', 'function'],
        'history': ['history', 'historical', 'war', 'ancient', 'medieval', 'century'],
        'language': ['language', 'grammar', 'writing', 'literature', 'essay', 'poem'],
        'business': ['business', 'marketing', 'finance', 'management', 'economics', 'strategy']
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return topic

    return 'general'
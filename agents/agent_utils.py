import time
from langchain_groq import ChatGroq
from graph.helper_functions.logger import get_logger

class RateLimitedLLM:
    def __init__(self, llm, min_delay=1.0):
        self.llm = llm
        self.min_delay = min_delay
        self.last_call = 0

    def invoke(self, messages, **kwargs):
        current_time = time.time()
        elapsed = current_time - self.last_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        try:
            result = self.llm.invoke(messages, **kwargs)
            self.last_call = time.time()
            return result
        except Exception as e:
            if "rate limit" in str(e).lower():
                logger = get_logger()
                logger.warning(f"Rate limit hit, waiting 30s: {e}")
                time.sleep(30)
                return self.llm.invoke(messages, **kwargs)
            raise

LLMS = {
    'researcher': RateLimitedLLM(ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1800), min_delay=2.0),
    'drafter': RateLimitedLLM(ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=2500), min_delay=1.5),
    'summarizer': RateLimitedLLM(ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=1000), min_delay=1.0),
    'verifier': RateLimitedLLM(ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=800), min_delay=1.0),
    'enhancer': RateLimitedLLM(ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=3000), min_delay=1.5)
}
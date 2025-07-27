# tools.py
import logging
from langchain.tools import tool
from typing import List, Dict

# Import RAG chain runner and its output schema
from rag_chains import run_rag_chain, RagToolOutput
from output_schemas import AgentResponse

# --- logger for this module ---
logger = logging.getLogger(__name__)

# --- Mock Web Search Tool ---
@tool
def web_search(query: str) -> str:
    """
    Performs a mock web search for the given query.
    This is a placeholder. In a real application, this would integrate
    with a search API like Tavily, Google Search, Bing Search, etc.
    """
    logger.info(f"Web Search Tool called with query: '{query}'")
    # Simulate a web search result
    if "latest machine model" in query.lower():
        result = "The latest machine model is 'AlphaPro 2000' released in Q3 2024, featuring enhanced AI diagnostics."
    elif "company contact" in query.lower():
        result = "You can contact support at support@example.com or call +1-800-123-4567."
    else:
        result = f"Search results for '{query}': No specific information found in mock database. You might want to try a real search engine."
    
    logger.info(f"Web Search Tool returned: {result[:100]}...")
    return result


# --- RAG/Manual Lookup Tool ---
@tool
def manual_lookup_structured(
    query: str,
    chat_history: List[List[str]] = [] # Gradio history format
) -> RagToolOutput: # Returns the Pydantic model directly
    """
    Looks up information in the machine manuals using RAG.
    Use this when the question is specifically about machine operation, errors, specifications, or troubleshooting.
    Returns a structured output including the answer and source documents.
    """
    logger.info(f"Manual Lookup Tool called with query: '{query}'")
    # The run_rag_chain function handles the history-aware retrieval internally
    rag_output = run_rag_chain(query, chat_history)
    
    logger.info(f"Manual Lookup Tool returned: {rag_output.answer[:100]}...")
    return rag_output

# --- List of all available tools ---
tools = [web_search, manual_lookup_structured]
logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")
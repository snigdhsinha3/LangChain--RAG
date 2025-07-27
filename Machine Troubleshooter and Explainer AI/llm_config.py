# llm_config.py
"""
llm_config.py

Configures and initializes multiple instances of LangChain-compatible LLMs 
using a locally hosted `llama.cpp-server` via OpenAI-compatible API.

This module sets up LLMs for:
- Main response generation
- Query rephrasing with historical context
- Planning tasks (e.g., agent workflow decisions)

Attributes:
    LOCAL_LLM_URL (str): Base URL for locally running LLM server.
    LLAMA_MODEL_NAME_ON_SERVER (str): Model name as served by the local server.
    llm (ChatOpenAI): Primary LLM used for answering user queries.
    question_rephrase_llm (ChatOpenAI): LLM tuned for rewriting user questions 
        using conversation history.
    planner_llm (ChatOpenAI): LLM used to plan agent steps or task orchestration.

Notes:
    - All models use the same base configuration and model name.
    - API key is passed for compatibility; local servers typically ignore it.
    - Different temperature settings help optimize behavior per role.
    - Streaming is enabled for the main LLM to support real-time token output.

Example:
    >>> from llm_config import llm
    >>> response = llm.invoke("What's the capital of Japan?")
    >>> print(response.content)
"""

from langchain_openai import ChatOpenAI
import os
import logging

# --- Configuration ---
LOCAL_LLM_URL = "http://localhost:8081/v1"
LLAMA_MODEL_NAME_ON_SERVER = "phi-3-mini-4k-instruct"

# --- logger for this module ---
logger = logging.getLogger(__name__)


# Initialize LLM for generation
llm = ChatOpenAI(
    model_name=LLAMA_MODEL_NAME_ON_SERVER,
    base_url = LOCAL_LLM_URL,             # Correct parameter name for initialization
    api_key = "sk-no-key-required",        # Correct parameter name for initialization
    temperature= 0.3,
    streaming= True
)
logger.info(f"Main LLM configured. Model= '{llm.model_name}', API Base= '{LOCAL_LLM_URL}'") 

# Initialize LLM for history-aware rephrasing (can be same LLM or a lighter one)
question_rephrase_llm = ChatOpenAI(
    model_name=LLAMA_MODEL_NAME_ON_SERVER,
    base_url = LOCAL_LLM_URL,             
    api_key = "sk-no-key-required",        
    temperature= 0.1, 
)
logger.info(f"Questions Rephrase LLM configured: Model= '{question_rephrase_llm.model_name}', API Base= '{LOCAL_LLM_URL}'")

# LLM for planning (can be smaller, faster model if available)
planner_llm = ChatOpenAI(
    model_name=LLAMA_MODEL_NAME_ON_SERVER,
    base_url = LOCAL_LLM_URL,             
    api_key = "sk-no-key-required",        
    temperature= 0.1 
)
logger.info(f"Planner LLM configured: Model= '{planner_llm.model_name}', API Base= '{LOCAL_LLM_URL}'")
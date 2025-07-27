# agents/planner_agent.py
import logging
from typing import TypedDict, List # TypedDict is needed if AgentState is defined as such
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
# Ensure pydantic_v1 is used if you are sticking with pydantic v1.
# If you are using pydantic v2, import from pydantic directly.
# From your previous logs, LangChainDeprecationWarning suggests to use 'from pydantic import BaseModel, Field'
from langchain_core.pydantic_v1 import Field 
from pydantic import BaseModel

from llm_config import planner_llm
from agents.base_agent import Agent as CustomAgent, AgentState


# Use the custom CustomAgent's logger directly for consistent logging
planner_logger = CustomAgent("Planner", CustomAgent.BLUE).logger


# --- Node: Call the Planner LLM ---
def call_planner_node(state: AgentState) -> AgentState:
    planner_logger.info("--- Node: Call Planner ---")
    messages = state['messages']

    # Get the latest human message for planning
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    if not user_query:
        planner_logger.error("No human message found in the state for planning")
        state['decision'] = "error_no_input" # Set the decision in the state
        # FIX 2: Return the state object itself, not a new dictionary
        state['tool_output'] = "Planner could not find user input." # Add any relevant info
        return state 

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert planning assistant focused solely on providing information by looking up documentation. "
                     "Your task is to formulate a plan that uses the 'manual_lookup_structured' tool to answer the user's query. "
                     "The plan must contain exactly one step. "
                     "Clearly state the tool name in parentheses after the task, e.g., '1. Look up [user's main query or relevant keywords] (manual_lookup_structured)'. "
                     "Start your response with 'PLAN:'.\n\n"
                     "Current conversation history:\n{chat_history}"),
        ("human", "{input}"),
    ])

    # Filter chat history to only include the Human and AI message for prompt context
    lc_chat_history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    plan_chain = planner_prompt | planner_llm
    plan_response_content = plan_chain.invoke({
        "input": user_query,
        "chat_history": lc_chat_history
    }).content

    # Parse the plan from the LLM response
    if "PLAN:" in plan_response_content:
        plan_str = plan_response_content.split("PLAN:", 1)[1].strip()
    else:
        plan_str = plan_response_content # Fallback if LLM doesn't follow format


    # Simple parsing: split by numbers for now. More robust parsing might use regex or a Pydantic output parser.
    plan_steps = [step.strip() for step in plan_str.split('\n') if step.strip()]
    plan_steps = [step for step in plan_steps if step.startswith(tuple(str(i) + '.' for i in range(1, 10)))] # Ensure it's numbered list
    
    # --- Fallback for empty/unparseable plan ---
    if not plan_steps:
        planner_logger.warning(f"Planner generated empty or unparseable plan: {plan_response_content}")
        # Set the correct keys and a valid decision for the graph
        state["plan"] = ["Provide a general answer based on query."] # Use "plan" key
        state["next_step_index"] = 0
        state["decision"] = "synthesize" # Fallback to synthesize if no actionable plan
        state["tool_output"] = "Planner could not generate a clear plan based on the input."
        return state # Return the modified state

    planner_logger.info(f"Generated Plan:\n{plan_response_content}")
    
    # Initialize next_step_index to 0 for the first step
    # Set the decision for the graph to proceed to execution
    state["plan"] = plan_steps
    state["next_step_index"] = 0
    state["decision"] = "execute_step" # This is the decision for the graph to follow

    return state # Return the modified state
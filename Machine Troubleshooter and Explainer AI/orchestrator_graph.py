# orchestrator_graph.py

import logging
from typing import List, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import the custom agent nodes
from agents.planner_agent import call_planner_node
from agents.executor_agent import call_executor_node
from agents.synthesizer_agent import call_synthesizer_node
from agents.base_agent import Agent as CustomAgent, AgentState

# --- logger for this module ---
logger = CustomAgent("OrchestratorGraph", CustomAgent.RED).logger

# --- Define the custom routing logic ---
def route_decisions(state:AgentState):
    """
    Decides the next step in the graph base on the 'decision' key in the state.
    """
    decision = state.get('decision', 'unknown')

    logger.info(f"Routing decision: {decision}")

    # All these returns should be the *keys* defined in the conditional edges map
    if decision == 'execute_step':
        # If planner decides to execute a step, it should return 'execute_step' as the decision key
        # to match the edge "execute_step": "executor"
        if state['next_step_index'] < len(state['plan']):
            return "execute_step"
        else:
            # If all planned steps are done, then the decision is to synthesize
            return "synthesize"
    elif decision == "synthesize":
        return "synthesize"
    elif decision == "handle_execution_error":
        logger.warning("Execution error occurred. Proceeding to synthesizer with error information.")
        return "handle_execution_error" # Return the decision key
    elif decision == "end":
        logger.info("Graph execution ending successfully.")
        return "end" # Return the decision key
    elif decision == "end_with_error":
        logger.error("Graph execution ending with error.")
        return "end_with_error" # Return the decision key
    elif decision == "error_no_input":
        logger.critical("No valid human input for planning. Ending graph.")
        return "error_no_input" # Return the decision key
    else:
        # If an unknown decision is encountered (likely because planner didn't set it)
        # We need to pick a valid decision key from our map as a fallback.
        # If there's a plan, the most logical default is to try and execute the first step.
        logger.warning(f"Unknown decision '{decision}'. Defaulting to 'execute_step' if plan exists, else 'synthesize'.")
        if state['plan'] and state['next_step_index'] < len(state['plan']): # Check if there's a plan and steps left
            return "execute_step" # Default to this decision key, which maps to "executor"
        else:
            return "synthesize" # If no plan or plan finished, go to synthesize
    

# --- Build LangGraph Workflow ---
def build_langgraph_workflow():
    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", call_planner_node)
    workflow.add_node("executor", call_executor_node)
    workflow.add_node("synthesizer", call_synthesizer_node)

    # 2. Set the entry point
    workflow.set_entry_point("planner") # Always start with the planning 

    # 3. Add Edges
    # Use conditional edges from 'planner' to handle ALL its decisions using route_decisions
    workflow.add_conditional_edges(
        "planner",          # The node from which the edges originate
        route_decisions,    # Use your custom routing function
        {
            "execute_step": "executor",      # If route_decisions returns "execute_step", go to "executor" node
            "synthesize": "synthesizer",     # If route_decisions returns "synthesize", go to "synthesizer" node
            "handle_execution_error": "synthesizer", # Maps decision to node
            "end": END,                      # Maps decision to END
            "end_with_error": END,           # Maps decision to END
            "error_no_input": END            # Maps decision to END
            # Make sure every possible string return value of route_decisions
            # (when called from the planner node) is a key in this dictionary.
        }
    )
    
    # After executor, use conditional routing (already correctly set up based on route_decisions)
    workflow.add_conditional_edges(
        "executor",
        route_decisions, # Use your custom routing function
        {
            "execute_step": "executor",     # Loop back to executor for next step
            "synthesize": "synthesizer",    # All steps done, go to synthesizer
            "handle_execution_error": "synthesizer", # Error, go to synthesizer (as defined in route_decisions)
            "end": END,                     # Executor might explicitly end if it fulfilled the request directly
            "end_with_error": END,          # Executor explicit end with error
        }
    )

    # After synthesizer, the workflow always ends (this is correct)
    workflow.add_edge("synthesizer", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return app

# Compile the graph immediately on module load
langgraph_app = build_langgraph_workflow()
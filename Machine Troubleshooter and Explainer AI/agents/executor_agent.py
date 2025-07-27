import logging
from typing import List, Tuple
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentFinish
import json

from llm_config import llm
from tools import tools
from agents.base_agent import Agent as CustomAgent, AgentState
from rag_chains import RagToolOutput

# Use the custom CustomAgent's logger directly for consitent logging
executor_logger = CustomAgent("Executor", CustomAgent.GREEN).logger

# --- Node Call the Executor LLM / Tool Execution ---
def call_executor_node(state: AgentState):
    executor_logger.info(" --- NODE: Call Executor ---")
    plan = state ['plan']
    next_step_index = state['next_step_index']
    messages = state['messages']

    if next_step_index>=len(plan):
        executor_logger.info("No more steps in the plan. Proceeding to synthesis.")
        return{"decision": "synthesize"}
    
    current_step = plan[next_step_index]
    executor_logger.info(f"Executing step {next_step_index + 1}/{len(plan)}: '{current_step}'")

    tool_name = None
    tool_input = None # Initialize as None, to indicate no clear input yet

    # Attempt to parse tool call from the description
    for tool in tools:
        if f"({tool.name})" in current_step:
            tool_name = tool.name
             # Extract input after the tool name
            tool_input = current_step.split(f"({tool.name})", 1)[1].strip().strip(':').strip()
            # If input is empty, try to derive it from the main part of the step
            if not tool_input:
                tool_input = current_step.split(f"({tool.name})", 1)[0].strip().replace(f"{next_step_index+1}.", "").strip()
            break
    
    tool_result = ""
    if tool_name:
        executor_logger.info(f"Identified tool call: {tool_name} with input: '{tool_input}'")
        try:
            # Find the actual tool object
            selected_tool = next(t for t in tools if t.name == tool_name)

            # Pass chat history to RAG tool if it's in manual lookup
            if tool_name == "manual_lookup_structured":
                # Convert LangGraph message to Gradio history format for tool
                gradio_chat_history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        current_human = msg.content
                    elif isinstance(msg, AIMessage) and current_human: # Only pair if there is a preceeding human message
                        gradio_chat_history.append((current_human, msg.content))
                        current_human = None # Reset
                
                tool_response: RagToolOutput = selected_tool.invoke({"query": tool_input, "chat_history": gradio_chat_history})
                tool_result = tool_response.answer # Just the answer content for now
                # You might want to store source_documents too if needed later
            else:
                # For other tools, just invoke with the string input
                tool_result = selected_tool.invoke(tool_input)

            executor_logger.info(f"Tool '{tool_name}' returned: {tool_result[:100]}...")

            # Optionally, add a ToolMessage to the messages list for agent memory/context
            # This is good practice for the agent to "see" tool outputs
            # tool_message = ToolMessage(content=tool_result, tool_call_id="mock_tool_call_id") # real tool_call_id needed
            # new_messages = state['messages'] + [tool_message]
            # return {"tool_output": tool_result, "next_step_index": next_step_index + 1, "messages": new_messages, "decision": "execute_step"}

        except Exception as e:
            executor_logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            tool_result = f"Error executing tool '{tool_name}': {e}"
            # Decide if this error means re-plan or synthesize with error info
            return {"tool_output": tool_result, "next_step_index": next_step_index + 1, "decision": "handle_execution_error"}
    else:
        executor_logger.info(f"No specific tool identified for step: '{current_step}'. Using general LLM reasoning")
        # If no tool, use the LLM to process the step generally
        executer_prompt = ChatPromptTemplate([
            ("system", "You are an assistant capable of processing a single step of a plan. "
                       "Given the current step and any previous tool output, complete the step or provide information"),
            ("user", "Plan Step: {step}\nPrevious Tool Output: {tool_output}"),
        ])

        general_response = (executer_prompt | llm).invoke({
            "step": current_step,
            "tool_output": state.get('tool_output', '') # Pass previous tool output if exists
        }).content
        tool_result = general_response
        executor_logger.info(f"General LLM response for step: {tool_result[:100]}...")

    # After execution, increment step index and decide next
    new_next_step_index = next_step_index + 1
    if new_next_step_index < len(plan):
        # If there are more steps, continue executing
        return {"tool_output": tool_result, "next_step_index": new_next_step_index, "decision":"execute_step"}
    else:
        # If all steps are done, go to synthesis
        return {"tool_output": tool_result, "next_step_index": new_next_step_index, "decision":"synthesize"}    




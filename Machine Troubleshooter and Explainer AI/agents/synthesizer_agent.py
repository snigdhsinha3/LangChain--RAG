# agents/synthesizer_agent.py
import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from llm_config import llm 
from output_schemas import AgentResponse
from agents.base_agent import Agent as CustomAgent, AgentState

# Use the CustomAgent's logger directly for consistent logging
synthesizer_logger = CustomAgent("Synthesizer", CustomAgent.YELLOW).logger

# --- Pydantic Parser for the Agent's Final Answer ---
final_answer_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# --- Node: Synthesize Final Answer ---
def call_synthesizer_node(state: AgentState):
    synthesizer_logger.info("--- Node: Call Synthesizer ---")
    messages = state["messages"]
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
        
    plan = state.get('plan', [])
    tool_output = state.get('tool_output', "No specific tool output")

    synthesis_prompt = ChatPromptTemplate([
        ("system", "You are an expert AI assistant tasked with providing comprehensive and structured answers. "
                     "Based on the user's original query, the plan that was executed, and the results from tool calls/general reasoning, "
                     "formulate a final answer in the specified JSON format. "
                     "Determine the 'answer_source' (manual, web_search, general_knowledge, mixed, none) and 'confidence' (high, medium, low) based on the information provided."
                     "If the information was obtained using the 'manual_lookup_structured' tool, set 'answer_source' to 'manual'."
                     "If no specific information was found, state that explicitly and suggest searching the web if appropriate."
                     "\n\n{format_instructions}\n\n"
                     "Original User Query: {original_query}\n\n"
                     "Execution Plan: {plan_summary}\n\n"
                     "Execution Results (Tool Output/General Reasoning):\n{execution_results}\n\n"
                     "Conversation History:\n{chat_history}"),
        ("human", "Please provide the final structured answer."),
    ]).partial(format_instructions=final_answer_parser.get_format_instructions())

    # Prepare chat history for prompt
    lc_chat_history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    # Combine plan steps into a summary string
    plan_summary = "\n".join(plan) if plan else "No specific plan was generated"

    try:
        # Invoke the LLM with the structured output parser
        structured_response: AgentResponse = (synthesis_prompt | llm | final_answer_parser).invoke({
            "original_query": user_query,
            "plan_summary": plan_summary,
            "execution_results": tool_output,
            "chat_history": lc_chat_history
        })

        synthesizer_logger.info(f"Successfully synthesized final answer. Content: {structured_response.content[:100]}...")


        # The LangGraph node expects a dictionary back to update the state.
        # This dictionary should contain 'final_answer_content' and 'decision'
        # The 'final_answer_content' key is what your app.py's `respond_to_user`
        # expects to parse.
        new_messages = state['messages'] + [AIMessage(content=structured_response.model_dump_json())]
        return {
            "final_answer_content": structured_response.model_dump_json(),
            "messages": new_messages,
            "decision": "end" # Signal the graph to end successfully
        }

    except Exception as e:
        synthesizer_logger.error(f"Error synthesizing final answer: {e}", exc_info=True)
        clean_error_message = "I apologize, but an internal error occurred while trying to generate a comprehensive answer. Please check the logs for more details or try rephrasing your query."
        
        error_response = AgentResponse(
            content=clean_error_message,
            answer_source="none",
            confidence="low",
            follow_up_questions=["What specific error occurred?", "Can you try again?"]
        )
        new_messages = state['messages'] + [AIMessage(content=error_response.model_dump_json())]
        return {
            "final_answer_content": error_response.model_dump_json(),
            "messages": new_messages,
            "decision": "end_with_error" # Signal the graph to end with an error
        }
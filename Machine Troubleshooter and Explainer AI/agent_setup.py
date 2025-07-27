# agent_setup.py
import logging
from typing import List, Tuple, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import the compiled LangGraph application
from orchestrator_graph import langgraph_app, AgentState
from agents.base_agent import Agent as CustomAgent # For custom logging

# --- Logger for this module ---
logger = CustomAgent("MachineAgentManager", CustomAgent.RED).logger

class MachineAgentManager(CustomAgent):
    def __init__(self, name: str = "MainAgent", color: str = CustomAgent.RED):
        super().__init__(name, color)
        self.info("Initializing MachineAgentManager with LangGraph workflow.")
        self.langgraph_workflow = langgraph_app # Use the compiled LangGraph app
        self.chat_history: List[Tuple[str, str]] = [] # Storing history in Gradio format for convenience
        self._last_final_state: AgentState = {} # To store the final state after streaming

    # The original invoke_agent is for non-streaming. Keeping it for completeness if needed elsewhere.
    def invoke_agent(self, lc_history: List[BaseMessage], current_user_message: str) -> AgentState:
        self.info(f"Received user query (non-streaming): '{current_user_message}'")

        initial_state: AgentState = {
            "messages": lc_history,
            "user_query": current_user_message,
            "plan": [],
            "next_step_index": 0,
            "tool_output": "",
            "final_answer_content": None,
            "decision": "plan"
        }

        try:
            final_state: AgentState = self.langgraph_workflow.invoke(initial_state)
            self.info(f"LangGraph execution finished (non-streaming). Final decision: {final_state.get('decision')}")
            self.info(f"Final state returned by LangGraph invoke (non-streaming): {final_state}")
            return final_state

        except Exception as e:
            self.critical(f"Error during LangGraph agent invocation (non-streaming): {e}", exc_info=True)
            return {
                "messages": lc_history,
                "user_query": current_user_message,
                "plan": [],
                "next_step_index": 0,
                "tool_output": "",
                "final_answer_content": "AI: I apologize, but a critical error occurred during processing. Please check the application logs for details.",
                "decision": "error"
            }

    # Asynchronous streaming method ---
    async def astream_agent(self, lc_history: List[BaseMessage], current_user_message: str) -> AsyncGenerator[dict, None]:
        self.info(f"Received user query (streaming): '{current_user_message}'")

        initial_state: AgentState = {
            "messages": lc_history,
            "user_query": current_user_message,
            "plan": [],
            "next_step_index": 0,
            "tool_output": "",
            "final_answer_content": None,
            "decision": "plan"
        }

        self._last_final_state = {} # Reset last final state for new stream

        try:
            # Iterate over the streamed events from LangGraph
            async for event in self.langgraph_workflow.astream_events(
                initial_state,
                version="v1",
                stream_mode=["messages", "updates", "values"] # Stream messages (LLM tokens), updates (node progress), and full values (state)
            ):
                yield event # Yield each event directly to app.py

            # After the loop finishes, get the final state of the graph
            # This is important because the streaming events might not contain the full final state
            # including the structured final_answer_content.
            final_state = await self.langgraph_workflow.ainvoke(initial_state)
            self._last_final_state = final_state # Store it for retrieval by app.py

            self.info(f"LangGraph execution finished (streaming). Final decision: {final_state.get('decision')}")
            self.info(f"Final state returned by LangGraph ainvoke (streaming): {final_state}")

        except Exception as e:
            self.critical(f"Error during LangGraph agent streaming invocation: {e}", exc_info=True)
            # In case of an error during streaming, set a generic error state
            self._last_final_state = {
                "messages": lc_history,
                "user_query": current_user_message,
                "plan": [],
                "next_step_index": 0,
                "tool_output": "",
                "final_answer_content": "AI: I apologize, but a critical error occurred during streaming. Please check the application logs for details.",
                "decision": "error"
            }
            # Yield an event indicating an error, so app.py can handle it
            yield {"event": "on_error", "data": {"message": self._last_final_state["final_answer_content"]}}

    # --- Method to retrieve the last final state ---
    async def get_last_final_state(self) -> AgentState:
        # This method is called by app.py after the streaming loop completes
        # It returns the final state that was stored after the astream_events call
        return self._last_final_state


    def clear_memory(self):
        self.info("Clearing agent's conversation memory.")
        self._last_final_state = {} # Clear the stored state as well
        # If your RAG vectorstore or other components have memory, clear them here too
        pass

# Initialize the manager
machine_agent_manager = MachineAgentManager()

# app.py
import gradio as gr
from typing import List, Tuple
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
import logging
from langchain_core.messages import HumanMessage, AIMessage

# Import the initialized agent manager and our output schema
from agent_setup import machine_agent_manager # This now wraps the LangGraph app
from output_schemas import AgentResponse
# Import the update_vectorstore function
from vectorstore_manager import update_vectorstore

# --- GLOBAL LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)
# ------------------------------------

# --- Pydantic Parser for the Agent's Final Answer ---
final_answer_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# --- Gradio Chatbot Function ---
def respond_to_user(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    # Convert Gradio history to LangChain BaseMessage format for the agent
    lc_history = []
    for human_msg, ai_msg in history:
        if human_msg:
            lc_history.append(HumanMessage(content=human_msg))
        if ai_msg:
            lc_history.append(AIMessage(content=ai_msg))
            
    # Add the current user message to the LangChain history
    lc_history.append(HumanMessage(content=message))

    try:
        agent_output_state = machine_agent_manager.invoke_agent(lc_history, message)

        # --- NEW ERROR HANDLING LOGIC HERE ---
        # Check if the agent returned an error state
        if agent_output_state.get("decision") == "error":
            error_message_from_agent = agent_output_state.get("final_answer_content", "AI: An unknown error occurred in the agent.")
            history.append([message, error_message_from_agent])
            return "", history
        # --- END NEW ERROR HANDLING LOGIC ---

        # If not an error, proceed with parsing the final_answer_content
        # Ensure that if "final_answer_content" isn't present, it defaults to something parsable (e.g., empty JSON)
        agent_output_string = agent_output_state.get("final_answer_content", "{}") 

        try:
            structured_output: AgentResponse = final_answer_parser.parse(agent_output_string)
            formatted_output = (
                f"**Source:** {structured_output.answer_source.replace('_', ' ').title()}\n\n"
                f"**Answer:** {structured_output.content}\n\n"
                f"**Confidence:** {structured_output.confidence.capitalize()}"
            )
            if structured_output.follow_up_questions:
                formatted_output += "\n\n**Follow-up Questions:**\n" + "\n".join(
                    [f"- {q}" for q in structured_output.follow_up_questions]
                )
            
            # Append the current interaction to Gradio's history
            history.append([message, formatted_output])
            return "", history

        except ValidationError as ve:
            logging.error(f"Agent's final output did not conform to schema: {ve}", exc_info=True)
            logging.error(f"Raw agent output: {agent_output_string}")
            error_response = (
                "AI: I generated an answer, but I had trouble formatting it correctly. "
                "Here's the raw response:\n\n"
                f"```\n{agent_output_string}\n```\n\n"
                "Could you please try rephrasing your question?"
            )
            history.append([message, error_response])
            return "", history
        except Exception as parse_e:
            logging.error(f"Unexpected error during final output parsing: {parse_e}", exc_info=True)
            logging.error(f"Raw agent output: {agent_output_string}")
            error_response = (
                "AI: I encountered an unexpected error while trying to present my answer. "
                "Here's what I was able to get:\n\n"
                f"```\n{agent_output_string}\n```\n\n"
                "Please check the console for more details."
            )
            history.append([message, error_response])
            return "", history

    except Exception as e:
        logging.critical(f"Top-level error during agent response: {e}", exc_info=True)
        error_message = "AI: I apologize, but a critical error occurred. Please check the 'application.log' file."
        history.append([message, error_message])
        return "", history

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚙️ Machine Troubleshooter & Explainer AI 🛠️")
    gr.Markdown(
        "Ask questions about machine operation, specs, troubleshooting (from manuals) "
        "or general queries (web search). Type 'exit' to restart the conversation."
    )

    chatbot = gr.Chatbot(
        label="AI Assistant",
        height=400,
        render_markdown=True,
        show_copy_button=True
    )
    msg = gr.Textbox(label="Your Question", placeholder="Ask me anything about the machine or general topics...")
    
    with gr.Row():
        clear_btn = gr.ClearButton([msg, chatbot])
        update_rag_btn = gr.Button("Update Manuals (Re-index RAG)")

    def clear_history_gradio():
        machine_agent_manager.clear_memory()
        return None, []

    clear_btn.click(clear_history_gradio, inputs=None, outputs=[msg, chatbot])

    update_rag_btn.click(
        fn=update_vectorstore,
        inputs=None,
        outputs=gr.Textbox(label="Update Status", interactive=False),
        api_name="update_rag_status"
    )

    msg.submit(respond_to_user, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=False)

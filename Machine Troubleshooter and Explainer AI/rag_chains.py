# rag_chains.py
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import  create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Import LLMS and retriever from their respective modules
from llm_config import llm, question_rephrase_llm
from vectorstore_manager import get_retriever

# --- Logger for this module ---
logger = logging.getLogger(__name__)

# --- Pydantic model for RAG tool optuput ---
class RagToolOutput(BaseModel):
    """Output schema for the RAG tool."""
    answer: str = Field(description=" The answer derived from the RAG process.")
    source_documents: List[str] = Field(description="List og document snippets used to generate the answer.")

def get_rag_retriever():
    """Returns the configured RAG retriever."""
    return get_retriever()

def run_rag_chain(question: str, chat_history: List[tuple]) -> RagToolOutput:
    """
    Runs the RAG chain to answer a question based on retrived documents.
    This function will be exposed as a tool to the executor agent.
    """
    logger.info(f"Running RAG chain for question: '{question}'")

    retriever_instance = get_rag_retriever() # Get the up-to-date retriever

    if not retriever_instance:
        logger.warning("RAG retriever no available. Cannot run RAG chain.")
        return RagToolOutput(answer= "I cannot answer this question as the document knowledge base is not loaded", source_documents=[])
    
    # Convert history for Langchain format
    lc_chat_history = []
    for human_msg, ai_msg in chat_history:
        lc_chat_history.append(HumanMessage(content=human_msg))
        lc_chat_history.append(AIMessage(ai_msg))

    # A. History-aware Retriever Chain: Rephrases new questinos considering chat history
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. If the question is standalone, return it as it is. Do NOT add any extra context or conversational phrases."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        question_rephrase_llm,
        retriever_instance,
        history_aware_retriever_prompt
    )

    # B. Document stuffing chain: Combines retrieved docs and question for final answer
    qa_prompt  = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant specialized in manuals.Use the following retrieved context to answer the question. If the answer is not in the context, state that you don't have enough information from the manuals but you can try searching the web.\n\n"
                    "Context:\n{context}\n\n"
                    "Chat History: \n{chat_history}"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # C. Full Conversational Retrieval Chain: Combines A and B
    rag_chain_with_history = create_retrieval_chain(history_aware_retriever, qa_chain)

    try:
        response = rag_chain_with_history.invoke({
            "input": question,
            "chat_history": lc_chat_history
        })

        # Extract source documnts for the output schema
        source_docs = [doc.page_content for doc in response['context']]

        return RagToolOutput(answer=response["answer"], source_documents=source_docs)
    except Exception as e:
        logger.error(f"Error running RAG chain: {e}", exc_info=True)
        return RagToolOutput(answer=f"An error occured while trying to find information in the manuals: {e}", source_documents= [])
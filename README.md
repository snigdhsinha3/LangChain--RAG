# Machine Troubleshooter & Explainer AI
An intelligent AI assistant for diagnosing, explaining, and troubleshooting machines using large language models (LLMs) orchestrated with LangGraph and LangChain. This project demonstrates a multi-agent architecture combining planning, execution, and answer synthesis for complex workflows, backed by a real-time streaming Gradio web interface.

##🚀 Project Overview
This system accepts user queries related to machine operation, specifications, troubleshooting manuals, and general knowledge. It intelligently plans the next best steps, executes sub-tasks (like querying a knowledge base), and synthesizes comprehensive answers with source and confidence indicators. The architecture enables conditional routing, robust error handling, and provides a highly responsive user experience through streaming. The project currently leverages a local LLM server powered by llama.cpp, specifically utilizing the capybarahermes-2.5-mistral-7b.Q5_K_M.gguf model.

##🧩 Architecture & Components

### LangGraph Workflow

Custom graph-based orchestration with conditional edges managing flow between distinct AI agents.

Manages the shared AgentState for seamless data flow across the graph.

### Agents

**Planner:** Analyzes the user's query and formulates a stepwise plan, explicitly suggesting tools for execution. Currently configured to prioritize manual_lookup_structured for machine-related queries.

**Executor:** Executes individual steps from the plan, invoking specific tools (like RAG lookups) or employing general LLM reasoning if no tool is applicable.

**Synthesizer:** Combines intermediate results and tool outputs into a final, coherent, and structured response for the user.

### Agent Manager

Wraps the core LangGraph workflow, providing both synchronous and asynchronous (streaming) invocation methods.

Handles top-level error management and conversation memory clearing.

### Retrieval-Augmented Generation (RAG)

Integrates a vector database for efficient retrieval of information from uploaded manuals.

Leverages HuggingFaceEmbeddings and FAISS for document indexing and similarity search.

### Tools

Callable functions or modules that the Executor agent can invoke (e.g., manual_lookup_structured for RAG queries, web_search for general web searches, file_opener, text_reader, etc.).

### Pydantic Output Validation

Utilizes Pydantic models to define and enforce a strict schema for the AI's final output, ensuring consistency, confidence scores, and optional follow-up questions.

### Gradio UI

An interactive web chatbot providing a user-friendly interface for real-time interaction, displaying streamed responses, conversation history, and a manual update button.

### Local LLM Integration

The project is configured to use an LLM served locally via llama.cpp, specifically the capybarahermes-2.5-mistral-7b.Q5_K_M.gguf model, for all language model operations.

### ⚙️ Features
Multi-step LLM Orchestration: Custom LangGraph workflow with dynamic decision routing for complex queries.

Real-time Streaming: Asynchronous processing and token-by-token streaming of AI responses and intermediate agent steps (tool calls, thoughts) for a highly responsive UX.

Retrieval-Augmented Generation (RAG): Answers questions by retrieving relevant information from a custom knowledge base (e.g., machine manuals).

Structured AI Output: Ensures consistent and machine-readable AI responses using Pydantic schema validation.

Robust Error Handling: Comprehensive try-except blocks and graceful fallback responses throughout the agent workflow.

Conversation Memory: Maintains chat history for context-aware interactions.

Manuals Management: Ability to update/re-index the RAG vector store via a dedicated UI button.

Modular & Extensible: Designed for easy addition of new agent nodes, tools, or external API integrations.

### 📦 Installation & Setup
Follow these steps to get the project up and running locally:

Clone the repository:

git clone https://github.com/snigdhsinha3/machine-troubleshooter-ai.git
cd machine-troubleshooter-ai


### Create and activate a virtual environment:

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate


### Install dependencies:

pip install -r requirements.txt


(Note: You might see LangChainDeprecationWarning regarding pydantic_v1 or HuggingFaceEmbeddings. These are informational and the project is designed to handle them. For HuggingFaceEmbeddings, consider pip install -U langchain-huggingface for the latest version if you encounter issues.)

### Set up Local LLM (llama.cpp):

Download llama.cpp: Follow the instructions on the official llama.cpp GitHub repository to download and compile it for your system.

Download Model: Obtain the capybarahermes-2.5-mistral-7b.Q5_K_M.gguf model file. You can usually find this on Hugging Face (e.g., search "capybarahermes-2.5-mistral-7b.Q5_K_M.gguf" on Hugging Face Hub).

Start Local Server: Run the llama.cpp server (typically main or server executable) and ensure it's serving the model on http://localhost:8081 (or adjust llm_config.py if using a different port).

Example command (adjust paths and executable name as per your llama.cpp setup):

./server -m /path/to/your/capybarahermes-2.5-mistral-7b.Q5_K_M.gguf -c 4096 --host 0.0.0.0 --port 8081


(Ensure --host 0.0.0.0 if running in a container or accessing from another machine, otherwise localhost is fine.)

API Keys (Not required for local llama.cpp): If your llm_config.py still references an OPENAI_API_KEY for a different LLM, you can remove that or comment it out if you are only using the local llama.cpp server. The current setup assumes llm_config.py points to http://localhost:8081/v1 for the LLM API.

### Prepare Manuals:

Create a directory named manuals in the root of your project.

Place your .pdf or .txt machine manuals (or any other documents you want the RAG system to use) inside this manuals directory.

Run the application:

python app.py


### Access the UI:

Open your web browser and navigate to the local URL displayed in your terminal (e.g., http://127.0.0.1:7860).

## 🛠 Usage
Type questions related to machines, their operation, specifications, or troubleshooting.

The system will plan and execute multiple steps (e.g., looking up information in your provided manuals) before synthesizing a comprehensive, streamed answer.

Use the "Update Manuals (Re-index RAG)" button to re-process and update your knowledge base if you add or change documents in the manuals directory.

Click "Clear" to reset the conversation memory and start a new interaction.

## 🧑‍💻 Code Structure
orchestrator_graph.py — Defines the LangGraph state graph, nodes, and conditional routing logic for the multi-agent workflow.

agent_setup.py — Wraps the LangGraph workflow in a custom agent manager, providing both synchronous and asynchronous (streaming) invocation methods.

app.py — The Gradio frontend for user interaction, handling streamed results, and displaying the AI's output.

output_schemas.py — Pydantic models for validating the AI's structured output format (AgentResponse).

vectorstore_manager.py — Handles the creation, updating, and retrieval from the RAG vector store.

document_processor.py — Manages loading and splitting raw documents into chunks for the RAG system.

rag_chains.py — Implements the core Retrieval-Augmented Generation (RAG) logic.

tools.py — Defines the callable tools (e.g., manual_lookup_structured, web_search) that the Executor agent can use.

llm_config.py — Centralized configuration for Large Language Model instances.

agents/ — Directory containing the individual agent implementations:

base_agent.py: Base class for consistent logging.

planner_agent.py: Logic for generating multi-step plans.

executor_agent.py: Logic for executing plan steps and calling tools.

synthesizer_agent.py: Logic for synthesizing final, structured answers.

## 📈 Extending the Project
This project is designed with extensibility in mind. Here are some ideas for further development:

Add New Agent Nodes: Integrate new specialized agents to handle different types of queries (e.g., a "Code Generator Agent," a "Diagnostic Agent" for specific machine types).

Integrate More Tools: Connect to external APIs (e.g., weather APIs, inventory systems, CRM) by defining new tools.

Enhance RAG Sources: Expand the vector store to include more diverse data sources (e.g., databases, web pages, internal wikis).

Advanced UI Features: Implement more sophisticated UI elements for displaying tool outputs, progress bars, or interactive elements.

User Authentication: Add user login and management for multi-user deployments.

Deployment: Containerize the application using Docker for easier deployment to cloud platforms.

Evaluation Metrics: Implement metrics to evaluate the quality of answers and tool usage.

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Contributions
Contributions and feedback are welcome! Please feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

## 📞 Contact
Created by Snigdh Sinha - feel free to connect on LinkedIn Profile URL- linkedin.com/in/snigdhsinha.

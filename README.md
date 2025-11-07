# Generative AI Agent (RAG-based)

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and LangGraph. The agent retrieves relevant context from local documents and generates intelligent answers using OpenAI's language models.

## Features

- Document loading and embedding from text files
- Vector storage using Chroma DB
- Multi-step reasoning workflow:
  - Planning: Determines if retrieval is needed
  - Retrieval: Searches relevant documents
  - Answer Generation: Creates contextualized responses
  - Reflection: Evaluates answer quality

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation & Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
OR 
```bash
uv add -r requirements.txt
```

2. Set up your OpenAI API key:
   - For Windows PowerShell:
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```
   - Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Project Structure

```
.
├── data/                  # Knowledge base text files
│   ├── ai_future.txt
│   └── renewable_energy.txt
├── main.py               # Main application code
├── requirements.txt      # Python dependencies
└── .env                 # Environment variables (optional)
```

## Usage

Run the main script:
```bash
python main.py
```
or 

Run the main script:
```bash
uv run main.py
```

The agent will prompt you to enter a question. It will then:
1. Plan whether retrieval is needed
2. Search the knowledge base if required
3. Generate a contextual answer
4. Reflect on the answer's quality

Example questions:
- "What are the benefits of renewable energy?"
- "How will AI impact the future?"

## How It Works

1. **Document Processing**: Text files from the `data/` directory are loaded and embedded using OpenAI's embedding model.
2. **Vector Storage**: Embeddings are stored in a Chroma vector database for efficient similarity search.
3. **Workflow Nodes**:
   - `plan_node`: Decides if retrieval is needed based on question type
   - `retrieve_node`: Performs similarity search in the vector database
   - `answer_node`: Generates responses using retrieved context
   - `reflect_node`: Evaluates answer quality

## Dependencies

Main libraries used:
- langchain-openai: For OpenAI LLM and embeddings
- chromadb: Vector storage
- langgraph: Workflow orchestration
- python-dotenv: Environment variable management

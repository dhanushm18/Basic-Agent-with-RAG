import os
from dotenv import load_dotenv
from typing import TypedDict, Optional

# ENVIRONMENT SETUP
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
else:
    print("✅ OpenAI API key loaded successfully!")

# IMPORTS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph

# STEP 1: Load Documents
def load_documents():
    docs = []
    data_path = "data"
    if not os.path.exists(data_path):
        raise FileNotFoundError("'data/' folder not found. Please create it and add .txt files.")
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs


# Load the knowledge base
docs = load_documents()
print(f"Loaded {len(docs)} documents for the knowledge base.")

# STEP 2: Create Vector Store 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="db")
vectordb.persist()
print("Vector database created and persisted successfully!")

# STEP 3: Define the LLM 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# STEP 4: Define LangGraph Nodes

def plan_node(state):
    question = state["question"]
    if any(word in question.lower() for word in ["what", "how", "why", "benefit", "explain", "describe"]):
        next_step = "retrieve"
    else:
        next_step = "answer"
    return {"next": next_step, "question": question}


def retrieve_node(state):
    """Retrieve relevant context from ChromaDB."""
    question = state["question"]
    results = vectordb.similarity_search(question, k=2)
    context = "\n".join([r.page_content for r in results])
    return {"next": "answer", "question": question, "context": context}


def answer_node(state):
    """Generate an answer using LLM and retrieved context."""
    question = state["question"]
    context = state.get("context", "")
    prompt = f"""
    You are an intelligent assistant. Use the provided context to answer clearly and concisely.
    Context: {context}
    Question: {question}
    """
    response = llm.invoke(prompt)  
    answer = response.content if hasattr(response, "content") else str(response)
    return {"next": "reflect", "question": question, "answer": answer}


def reflect_node(state):
    """Reflect on the answer for relevance and completeness."""
    question = state["question"]
    answer = state["answer"]
    reflect_prompt = f"""
    Evaluate if the answer below is relevant and complete for the question.
    Question: {question}
    Answer: {answer}
    Respond briefly (max 30 words) like 'Yes, relevant and complete' or 'Partially relevant because...'
    """
    response = llm.invoke(reflect_prompt)  
    reflection = response.content if hasattr(response, "content") else str(response)
    return {"question": question, "answer": answer, "reflection": reflection}


# STEP 5: Define the Shared State Schema
class AgentState(TypedDict, total=False):
    question: str
    context: Optional[str]
    answer: Optional[str]
    reflection: Optional[str]
    next: Optional[str]


# STEP 6: Build LangGraph Workflow
graph = StateGraph(AgentState)

graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("reflect", reflect_node)

graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")

graph.set_entry_point("plan")

# Make the compiled workflow globally available (for Streamlit)
workflow = graph.compile()


if __name__ == "__main__":
    print("\n🚀 LangGraph RAG Agent Ready!")
    question = input("🧠 Ask a question: ")
    result = workflow.invoke({"question": question})
    print("\n===========================")
    print("✅ Final Answer:\n", result["answer"])
    print("🪞 Reflection:\n", result["reflection"])
    print("===========================")

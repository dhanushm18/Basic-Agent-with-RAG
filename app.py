import streamlit as st
from dotenv import load_dotenv
from main import workflow 

# Load environment variables
load_dotenv()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LangGraph RAG Agent", page_icon="🤖", layout="centered")
st.title("💬 Generative AI Q&A Agent")
st.caption("Built with LangGraph + ChromaDB + GPT-4o-mini")

# --- Input Section ---
user_question = st.text_input("🔍 Ask a question:", placeholder="e.g., What are the benefits of renewable energy?")

# --- Submit Button ---
if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("🤔 Thinking... Please wait..."):
            try:
                # Run the LangGraph workflow
                result = workflow.invoke({"question": user_question})

                # --- Display Results ---
                st.subheader("🧠 Answer")
                st.success(result.get("answer", "No answer generated."))

                st.subheader("🪞 Reflection")
                st.info(result.get("reflection", "No reflection generated."))

            except Exception as e:
                st.error(f"❌ Error while processing: {e}")

# --- Optional Footer ---
st.markdown("---")
st.caption("Developed by **Dhanush M** | Powered by GPT-4o-mini + LangGraph")

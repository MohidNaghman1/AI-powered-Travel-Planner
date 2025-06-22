import streamlit as st
import os
from dotenv import load_dotenv

# Import LangChain components
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

# --- CONFIGURATION (from your app.py) ---
JSON_PATH = "countries.json"
VECTOR_STORE_PATH = "faiss_index_countries"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# ==============================================================================
#      INGESTION LOGIC (Only run if needed)
# ==============================================================================

def load_and_process_json(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for country_data in data['countries']:
        page_content = f"{country_data['name']}: {country_data['description']}"
        metadata = {"source_country": country_data['name']}
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

def create_vector_store():
    with st.spinner("Performing first-time setup: Creating vector store... This may take a moment."):
        documents = load_and_process_json(JSON_PATH)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            separators=[". ", "! ", "? ", ", ", " ", ""],
            is_separator_regex=False
        )
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
    st.success("Vector store created and saved successfully!")

# ==============================================================================
#      CORE RAG LOGIC (Wrapped in Streamlit Caching for Performance)
# ==============================================================================

@st.cache_resource
def load_rag_chain():
    """
    Loads all the necessary components for the RAG chain and returns the chain.
    Uses Streamlit's caching to load these heavy components only once.
    """
    # Load environment variables for the Hugging Face API key
    load_dotenv()
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        # Stop the app if the key is not found
        st.error("🚨 HUGGINGFACEHUB_API_TOKEN not found. Please set it in your .env file.")
        st.stop()
    
    # --- 1. Load Retriever ---
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- 2. Create Prompt Template (Using your improved version) ---
    template = """
    <|system|>
    You are a factual travel assistant. Your primary directive is to answer user questions exclusively based on the text provided in the CONTEXT section.

    STRICT RULE: Do not use any of your external or pre-trained knowledge. If the information is not in the CONTEXT, you MUST reply with the exact phrase: "I do not have enough information in my knowledge base to answer that."

    Do not add any extra explanations or apologies. Just use that phrase.
    </s>
    <|user|>
    CONTEXT:
    {context}

    QUESTION:
    {question}</s>
    <|assistant|>
    """
    prompt = PromptTemplate.from_template(template)

    # --- 3. Load LLM ---
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_REPO_ID,
        task="text-generation",
        temperature=0.5, # Your specified temperature
        max_new_tokens=512
    )
    
    # --- 4. Create RAG Chain ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ==============================================================================
#                         STREAMLIT FRONTEND
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# --- Header ---
st.title("✈️ AI-Powered Travel Planner")
st.markdown(
    "Ask a question about a travel destination, and the AI will provide an answer based on its knowledge base."
)
st.divider()

# --- Main Application Logic ---
# Check if the vector store exists. If not, create it.
if not os.path.exists(VECTOR_STORE_PATH):
    st.info("Vector store not found. Starting first-time setup...")
    create_vector_store()

# Load the RAG chain (this will be cached for performance)
try:
    rag_chain = load_rag_chain()
    # A small success message that disappears after a few seconds
    st.toast("AI components loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load AI components: {e}", icon="🚨")
    st.stop()

# --- Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about a travel destination..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        # Show a thinking spinner while processing
        with st.spinner("🧠 Thinking..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
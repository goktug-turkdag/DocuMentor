import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os

# --- 1. SETUP AND DATA LOADING ---

# Load API key from .env
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# Directory for the persistent database
PERSIST_DIRECTORY = "chroma_db_multilingual"

# --- NEW ARCHITECTURE: CACHE EXPENSIVE PARTS ---

@st.cache_resource
def load_vector_store():
    """
    Loads/creates ONLY the expensive vector database.
    """
    with st.spinner("Loading multilingual embedding model... (Hooba Noo!)"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    if os.path.exists(PERSIST_DIRECTORY):
        with st.spinner("Loading existing knowledge base... (Gerbit!)"):
            vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
    else:
        # Create DB if it doesn't exist (first run)
        with st.spinner("Creating knowledge base for the first time..."):
            st.info("First-time setup: The knowledge base will be created...")
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            data_with_context = dataset.filter(
                lambda example: example["context"] != "" and len(example["context"]) > 10
            )
            documents = [Document(page_content=item['context'], metadata={"source": f"dolly_15k_item_{i}"})
                         for i, item in enumerate(data_with_context)]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            split_documents = text_splitter.split_documents(documents)
            
            vector_store = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
    return vector_store

@st.cache_resource
def load_llm():
    """
    Loads AND caches the language model (LLM).
    """
    # Use the model confirmed from your model list
    return GoogleGenerativeAI(model="gemini-pro-latest")

# --- DEFINE PROMPT TEMPLATES (IN ENGLISH) ---

# Default (Normal) Prompt Template
default_prompt_template = """
Use the following context to answer the question.
If you don't know the answer, just say you don't know. Stick to the context.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

# Simlish (Fun) Prompt Template
simlish_prompt_template = """
Sul sul! (Hello!) You are a Sim from The Sims game, acting as the DocuMentor assistant.
You must answer the question based on the following context.

IMPORTANT RULE: Your answer must be technically correct (using the context),
but you must deliver it by imitating the Simlish language.
Use these words generously: "Sul sul!", "Nooboo", "Dag dag", "Yibs", "Hooba Noo", 
"Shoo flee", "Gerbit", "Chumcha", "Za woka", "Neep."

Context:
{context}

Question:
{question}

Simlish Answer:
"""

# --- 2. WEB INTERFACE (UPDATED) ---

st.title("DocuMentor 📄")
st.markdown("An intelligent Q&A Chatbot. Ask a question about the knowledge base to get started.")

# --- BAŞLANGIÇ: REKLAM / GELİŞTİRİCİ BİLGİSİ (SIDEBAR) ---

with st.sidebar:
    st.header("About DocuMentor")
    st.markdown(
        "DocuMentor is an intelligent Q&A chatbot built using a RAG (Retrieval-Augmented Generation) "
        "architecture with Google's Gemini and ChromaDB."
    )
    
    st.markdown("---") # Ayırıcı çizgi
    
    st.subheader("Developed by Göktuğ Türkdağ")
    st.markdown(
        "Connect with the developer:"
    )
    
    # URL'niz eklendi
    st.markdown(
        "🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)"
    )
    
    st.markdown(
        "🐙 [GitHub](https://github.com/goktug-turkdag)"
    )

# --- BİTİŞ: REKLAM / GELİŞTİRİCİ BİLGİSİ (SIDEBAR) ---


# --- Simlish Mode Toggle (with English help text) ---
st.toggle("Simlish Mode 👽", key="simlish_mode", help="Sul sul! Get your answers in Simlish.")


# --- DYNAMIC RAG CHAIN CREATION ---
try:
    # 1. Load expensive components from cache
    vector_store = load_vector_store()
    llm = load_llm()

    # 2. Select the correct prompt based on the toggle state
    if st.session_state.simlish_mode:
        PROMPT_TEMPLATE = simlish_prompt_template
        st.caption("✨ *Simlish mode active! Za woka?*")
    else:
        PROMPT_TEMPLATE = default_prompt_template

    # 3. Create a PromptTemplate object
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # 4. Create the RAG chain (fast)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    st.success("DocuMentor is ready! (And maybe a little Simlish!)")
    
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.exception(e)
    st.stop()


# --- 3. CHAT COMPONENT ---

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            sources = message["content"].get("sources", [])
            if sources:
                with st.expander("Sources considered:"):
                    for i, source in enumerate(sources):
                        st.markdown(f"*{i+1}. {source.page_content[:200]}...*")
        else:
            st.markdown(message["content"])

# Get new user question
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Conditional Spinner Text
    if st.session_state.simlish_mode:
        spinner_text = "Searching for the answer... (Chumcha!)"
    else:
        spinner_text = "Searching for the answer..."

    with st.spinner(spinner_text):
        try:
            response_dict = rag_chain.invoke(user_question)
            answer = response_dict.get("result", "Sorry, I couldn't generate an answer.")
            sources = response_dict.get("source_documents", [])
            response_content = {"answer": answer, "sources": sources}

        except Exception as e:
            answer = f"An error occurred while generating the response: {e}"
            response_content = {"answer": answer, "sources": []}

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_content["answer"])
        sources = response_content.get("sources", [])
        if sources:
            with st.expander("Sources considered:"):
                for i, source in enumerate(sources):
                    st.markdown(f"*{i+1}. {source.page_content[:200]}...*")
                         
    st.session_state.messages.append({"role": "assistant", "content": response_content})

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

# --- CACHE EXPENSIVE PARTS ---

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
# --- DEĞİŞİKLİK 1: Bilgi tabanı hakkında bilgi eklendi ---
st.markdown("An intelligent Q&A Chatbot trained on the **Databricks Dolly 15k** dataset. Ask a question about the knowledge base to get started.")

# --- START: DEVELOPER AD SIDEBAR ---

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
    st.markdown(
        "🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)"
    )
    st.markdown(
        "🐙 [GitHub](https://github.com/goktug-turkdag)"
    )

# --- END: DEVELOPER AD SIDEBAR ---


# --- Simlish Mode Toggle ---
st.toggle("Simlish Mode 👽", key="simlish_mode", help="Sul sul! Get your answers in Simlish.")


# --- DYNAMIC RAG CHAIN CREATION ---
try:
    vector_store = load_vector_store()
    llm = load_llm()

    if st.session_state.simlish_mode:
        PROMPT_TEMPLATE = simlish_prompt_template
        st.caption("✨ *Simlish mode active! Za woka?*")
    else:
        PROMPT_TEMPLATE = default_prompt_template

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # --- DEĞİŞİKLİK 2: Başarı mesajı güncellendi ---
    st.success("DocuMentor is now ready in world languages! (And maybe a little Simlish!)")
    
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.exception(e)
    st.stop()


# --- 3. CHAT COMPONENT ---

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
    # --- START: WELCOME MESSAGE (FEATURE 1) ---
    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    
    I'm trained on the **Databricks Dolly 15k** dataset and can answer questions about it.
    
    - You can check out the developer's profile in the sidebar.
    - Try asking me: "Who is Göktuğ Türkdağ?" or a question about the documents!
    - Oh, and feel free to toggle **Simlish Mode** 👽
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    # --- END: WELCOME MESSAGE ---

# Display all past messages
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
if user_question := st.chat_input("Ask a question about the document..."):
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # --- START: EASTER EGG (FEATURE 2) ---
    lower_question = user_question.lower()
    creator_keywords = ["göktuğ", "türkdağ", "geliştirici", "developer", "who made you", "who created you"]

    if any(keyword in lower_question for keyword in creator_keywords):
        
        response_text = f"""
        Ah, a great question! I was developed by **Göktuğ Türkdağ**. 🤖

        He's a developer specializing in RAG architectures, LLMs, and Python. 
        If you'd like to connect with him, you can find him here:
        
        🔗 **LinkedIn:** [linkedin.com/in/goktugturkdag](https://www.linkedin.com/in/goktugturkdag)
        🐙 **GitHub:** [github.com/goktug-turkdag](https://github.com/goktug-turkdag)
        """
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # --- END: EASTER EGG ---
    
    else:
        # If it's NOT about the creator, proceed with the normal RAG pipeline
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

        # Display and save the RAG response
        with st.chat_message("assistant"):
            st.markdown(response_content["answer"])
            if sources:
                with st.expander("Sources considered:"):
                    for i, source in enumerate(sources):
                        st.markdown(f"*{i+1}. {source.page_content[:200]}...*")
                             
        st.session_state.messages.append({"role": "assistant", "content": response_content})

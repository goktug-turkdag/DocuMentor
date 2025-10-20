import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import os
import time

# --- 1. KURULUM VE VERİ YÜKLEME ---

# API anahtarını .env dosyasından yükle
load_dotenv()

# API anahtarının yüklenip yüklenmediğini kontrol et
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file and its contents.")
    st.stop()

# Bu fonksiyon, pahalı işlemleri (model yükleme, veri işleme) hafızada tutar.
@st.cache_resource
def setup_rag_pipeline():
    """
    Veri setini yükler, RAG pipeline'ını kurar ve hazır bir 'chain' objesi döndürür.
    """
    with st.spinner("Loading and preparing the knowledge base... This may take a few minutes on the first run."):
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        data_with_context = dataset.filter(
            lambda example: example["context"] != "" and len(example["context"]) > 10
        )
        contexts = [item['context'] for item in data_with_context]

    with st.spinner("Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with st.spinner("Creating vector database..."):
        vector_store = FAISS.from_texts(contexts, embeddings)

    with st.spinner("Initializing the language model and RAG chain..."):
        llm = GoogleGenerativeAI(model="gemini-pro-latest")
        retriever = vector_store.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
    
    return rag_chain

# Pipeline'ı başlat
try:
    rag_chain = setup_rag_pipeline()
    st.success("DocuMentor is ready to answer your questions!")
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.stop()


# --- 2. WEB ARAYÜZÜ (GÜNCELLENMİŞ METİNLERLE) ---

st.title("DocuMentor 📄")
st.markdown("An intelligent Q&A Chatbot for navigating technical documents. Ask a question about the knowledge base to get started.")

# Chat geçmişini tutmak için session state kullanalım
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni bir soru al
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Searching for the answer..."):
        try:
            response = rag_chain.run(user_question)
        except Exception as e:
            response = f"An error occurred while generating the response: {e}"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
from dotenv import load_dotenv
import os

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import time

# --- 1. KURULUM VE VERİ YÜKLEME ---

# API anahtarını .env dosyasından yükle
load_dotenv()

# API anahtarının yüklenip yüklenmediğini kontrol et
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı ve içeriğini kontrol edin.")
    st.stop()

# Bu fonksiyon, pahalı işlemleri (model yükleme, veri işleme) hafızada tutar.
@st.cache_resource
def setup_rag_pipeline():
    """
    Veri setini yükler, RAG pipeline'ını kurar ve hazır bir 'chain' objesi döndürür.
    Bu işlem ilk çalıştırmada biraz zaman alabilir.
    """
    with st.spinner("Veri seti hazırlanıyor... Bu işlem ilk seferde biraz zaman alabilir."):
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        data_with_context = dataset.filter(
            lambda example: example["context"] != "" and len(example["context"]) > 10
        )
        contexts = [item['context'] for item in data_with_context]

    with st.spinner("Embedding modeli hazırlanıyor..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with st.spinner("Vektör veritabanı oluşturuluyor..."):
        vector_store = FAISS.from_texts(contexts, embeddings)

    with st.spinner("RAG zinciri ve üretici model hazırlanıyor..."):
        # --- FİNAL DEĞİŞİKLİK: HESABINLA UYUMLU MODEL ADI KULLANILIYOR ---
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
    st.success("✅ Chatbot başarıyla yüklendi ve hazır!")
except Exception as e:
    st.error(f"Pipeline kurulurken bir hata oluştu: {e}")
    st.stop()


# --- 2. WEB ARAYÜZÜ ---

st.title("🤖 Akbank GenAI Bootcamp RAG Chatbot")
st.markdown("Bu chatbot, **Databricks Dolly 15k** veri setindeki bilgilerle sorularınızı yanıtlamak üzere tasarlanmıştır.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Sorunuzu buraya yazın...")

if user_question:
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Cevap aranıyor..."):
        try:
            response = rag_chain.run(user_question)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"Cevap oluşturulurken bir hata meydana geldi:\n\n{e}"
            st.error(error_message)
            st.chat_message("assistant").markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

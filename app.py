import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
# <-- Geliştirme Önerisi 1: Chunking için gerekli importlar eklendi -->
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document # Metinleri Document objesine dönüştürmek için
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
    Veri setini yükler, metinleri chunk'lara ayırır, RAG pipeline'ını kurar
    ve hazır bir 'chain' objesi döndürür.
    """
    with st.spinner("Loading and preparing the knowledge base... This may take a few minutes on the first run."):
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        data_with_context = dataset.filter(
            lambda example: example["context"] != "" and len(example["context"]) > 10
        )
        # Veri setindeki context'leri Document objelerine dönüştürelim
        # Her bir Document'a basit bir metadata ekleyebiliriz (opsiyonel, kaynak takibi için)
        documents = [Document(page_content=item['context'], metadata={"source": f"dolly_15k_item_{i}"})
                     for i, item in enumerate(data_with_context)]

    # <-- Geliştirme Önerisi 1: Chunking Adımı Eklendi -->
    with st.spinner("Chunking documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Chunk başına karakter sayısı (ayarlanabilir)
            chunk_overlap=200, # Chunk'lar arası örtüşme (ayarlanabilir)
            length_function=len,
        )
        split_documents = text_splitter.split_documents(documents)
        st.write(f"Split into {len(split_documents)} chunks.") # Bilgi mesajı

    with st.spinner("Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with st.spinner("Creating vector database from chunks..."):
        # Metinler yerine chunk'lanmış Document objelerini kullan
        vector_store = FAISS.from_documents(split_documents, embeddings)

    with st.spinner("Initializing the language model and RAG chain..."):
        llm = GoogleGenerativeAI(model="gemini-pro-latest")
        retriever = vector_store.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            # <-- Geliştirme Önerisi 3: Kaynak Gösterimi İçin Eklendi -->
            return_source_documents=True
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
        # <-- Geliştirme Önerisi 3: Kaynakları da göstermek için düzenleme -->
        if isinstance(message["content"], dict): # Eğer mesaj hem cevabı hem kaynakları içeriyorsa
             st.markdown(message["content"]["answer"])
             if message["content"]["sources"]:
                 st.markdown("---")
                 st.markdown("**Sources considered:**")
                 for i, source in enumerate(message["content"]["sources"]):
                     # Kaynağın içeriğini veya metadata'sını gösterelim
                     # Çok uzunsa bir kısmını gösterebiliriz
                     source_text = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                     st.markdown(f"*{i+1}. {source_text}*")
                     # Eğer metadata'da kaynak bilgisi varsa onu da ekleyebiliriz
                     # if "source" in source.metadata:
                     #     st.markdown(f"   (Source ID: {source.metadata['source']})")
        else: # Sadece metin içeren eski mesajlar veya kullanıcı mesajları
            st.markdown(message["content"])


# Kullanıcıdan yeni bir soru al
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    # Kullanıcı mesajını normal metin olarak göster ve kaydet
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Searching for the answer..."):
        try:
            # .invoke() metodu cevapla birlikte kaynakları da içeren bir dictionary döndürür
            response_dict = rag_chain.invoke(user_question)
            answer = response_dict["result"]
            # Kaynakları alalım (varsa)
            sources = response_dict.get("source_documents", [])

            # Cevabı ve kaynakları içeren bir dictionary oluşturalım
            response_content = {"answer": answer, "sources": sources}

        except Exception as e:
            # Hata durumunda sadece hata mesajını gösterelim
            response_content = f"An error occurred while generating the response: {e}"
            sources = [] # Hata durumunda kaynak listesi boş olsun

    # Asistan mesajını (cevap ve kaynaklar dict'i veya hata mesajı) göster ve kaydet
    with st.chat_message("assistant"):
        if isinstance(response_content, dict):
             st.markdown(response_content["answer"])
             if sources:
                 st.markdown("---")
                 st.markdown("**Sources considered:**")
                 for i, source in enumerate(sources):
                     source_text = source.page_content[:200] + "..."

import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
# Geliştirme Önerisi 2: Kalıcı DB için Chroma'yı da import edebilirsiniz
# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
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
        documents = [Document(page_content=item['context'], metadata={"source": f"dolly_15k_item_{i}"})
                     for i, item in enumerate(data_with_context)]

    # <-- Geliştirme Önerisi 1: Chunking Adımı -->
    with st.spinner("Chunking documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_documents = text_splitter.split_documents(documents)
        
        # <-- İSTEK 1: "Split into X chunks" mesajı kaldırıldı -->
        # st.write(f"Split into {len(split_documents)} chunks.") 

    with st.spinner("Loading embedding model..."):
        # <-- Geliştirme Önerisi 4: Çok dilli model için not -->
        # Farklı dillerde destek için: model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with st.spinner("Creating vector database from chunks..."):
        # <-- Geliştirme Önerisi 2: Kalıcı veritabanı için not -->
        # Kalıcı bir çözüm için (örn: ChromaDB):
        # db_directory = "chroma_db"
        # vector_store = Chroma.from_documents(split_documents, embeddings, persist_directory=db_directory)
        
        # Şu anki geçici (in-memory) FAISS çözümü:
        vector_store = FAISS.from_documents(split_documents, embeddings)

    with st.spinner("Initializing the language model and RAG chain..."):
        llm = GoogleGenerativeAI(model="gemini-pro")
        
        # Retriever'ı en alakalı 3 sonucu getirecek şekilde ayarla
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            # Kaynak Gösterimi İçin Eklendi
            return_source_documents=True
        )

    return rag_chain

# Pipeline'ı başlat
try:
    rag_chain = setup_rag_pipeline()
    st.success("DocuMentor is ready to answer your questions!")
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.exception(e) # Hatanın tüm detaylarını logla
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
        if isinstance(message["content"], dict): # Eğer mesaj hem cevabı hem kaynakları içeriyorsa
            st.markdown(message["content"]["answer"])
            
            # <-- İSTEK 2: Kaynaklar boşsa başlığı gösterme (Geçmiş için) -->
            sources = message["content"].get("sources", [])
            if sources and len(sources) > 0:
                # UI İyileştirmesi: Kaynakları expander içine al
                with st.expander("Sources considered:"):
                    for i, source in enumerate(sources):
                        source_text = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                        st.markdown(f"*{i+1}. {source_text}*")
                        if "source" in source.metadata:
                             st.markdown(f"   _(Source ID: {source.metadata['source']})_")
        
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
            response_dict = rag_chain.invoke(user_question)
            answer = response_dict.get("result", "Sorry, I couldn't generate an answer.")
            sources = response_dict.get("source_documents", [])
            response_content = {"answer": answer, "sources": sources}

        except Exception as e:
            answer = f"An error occurred while generating the response: {e}"
            sources = [] # Hata durumunda kaynak listesi boş olsun
            response_content = {"answer": answer, "sources": sources}

    # Asistan mesajını (cevap ve kaynaklar dict'i) göster ve kaydet
    with st.chat_message("assistant"):
        st.markdown(response_content["answer"])
        
        # <-- İSTEK 2: Kaynaklar boşsa başlığı gösterme (Yeni cevap için) -->
        if sources and len(sources) > 0:
            # UI İyileştirmesi: Kaynakları expander içine al
            with st.expander("Sources considered:"):
                for i, source in enumerate(sources):
                    source_text = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    st.markdown(f"*{i+1}. {source_text}*")
                    if "source" in source.metadata:
                         st.markdown(f"   _(Source ID: {source.metadata['source']})_")
                         
    # Mesajı (cevap + kaynaklar) session state'e kaydet
    st.session_state.messages.append({"role": "assistant", "content": response_content})

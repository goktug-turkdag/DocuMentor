import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import Chroma # FAISS yerine ChromaDB import edildi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

# --- 1. KURULUM VE VERİ YÜKLEME ---

# API anahtarını .env dosyasından yükle
load_dotenv()

# API anahtarının yüklenip yüklenmediğini kontrol et
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file and its contents.")
    st.stop()

# Kalıcı veritabanının saklanacağı klasörün adı
PERSIST_DIRECTORY = "chroma_db_multilingual"

# Bu fonksiyon, pahalı işlemleri (model yükleme, veri işleme) hafızada tutar.
@st.cache_resource
def setup_rag_pipeline():
    """
    Veri setini yükler, metinleri chunk'lara ayırır, RAG pipeline'ını kurar
    ve hazır bir 'chain' objesi döndürür. Veritabanını diske kaydeder ve
    sonraki çalıştırmalarda diskten yükler.
    """
    
    # <-- Geliştirme Önerisi 4: Çok Dilli Embedding Modeli Entegre Edildi -->
    with st.spinner("Loading multilingual embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    # <-- Geliştirme Önerisi 2: Kalıcı ChromaDB Veritabanı Entegre Edildi -->
    # Eğer veritabanı diskte mevcutsa, onu yükle.
    if os.path.exists(PERSIST_DIRECTORY):
        with st.spinner("Loading existing knowledge base from disk..."):
            vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
    # Eğer veritabanı yoksa, oluştur ve diske kaydet.
    else:
        with st.spinner("Creating knowledge base for the first time. This might take a while..."):
            st.info("First-time setup: The knowledge base will be created and saved to disk for faster startups later.")
            
            # Veri setini yükle
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            data_with_context = dataset.filter(
                lambda example: example["context"] != "" and len(example["context"]) > 10
            )
            documents = [Document(page_content=item['context'], metadata={"source": f"dolly_15k_item_{i}"})
                         for i, item in enumerate(data_with_context)]

            # Metinleri parçalara ayır (chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_documents = text_splitter.split_documents(documents)
            
            # Veritabanını oluştur ve diske kalıcı olarak kaydet
            vector_store = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )

    with st.spinner("Initializing the language model and RAG chain..."):
        llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest")
        
        # Retriever'ı en alakalı 3 sonucu getirecek şekilde ayarla
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    return rag_chain

# Pipeline'ı başlat
try:
    rag_chain = setup_rag_pipeline()
    st.success("DocuMentor is ready to answer your questions in multiple languages!")
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.exception(e) # Hatanın tüm detaylarını logla
    st.stop()


# --- 2. WEB ARAYÜZÜ (DEĞİŞİKLİK YOK) ---

st.title("DocuMentor 📄")
st.markdown("An intelligent Q&A Chatbot for navigating technical documents. Ask a question about the knowledge base to get started.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            sources = message["content"].get("sources", [])
            if sources:
                with st.expander("Sources considered:"):
                    for i, source in enumerate(sources):
                        source_text = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                        st.markdown(f"*{i+1}. {source_text}*")
                        if "source" in source.metadata:
                             st.markdown(f"   _(Source ID: {source.metadata['source']})_")
        else:
            st.markdown(message["content"])

user_question = st.chat_input("Ask a question about the document...")

if user_question:
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
            response_content = {"answer": answer, "sources": []}

    with st.chat_message("assistant"):
        st.markdown(response_content["answer"])
        sources = response_content.get("sources", [])
        if sources:
            with st.expander("Sources considered:"):
                for i, source in enumerate(sources):
                    source_text = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    st.markdown(f"*{i+1}. {source_text}*")
                    if "source" in source.metadata:
                         st.markdown(f"   _(Source ID: {source.metadata['source']})_")
                         
    st.session_state.messages.append({"role": "assistant", "content": response_content})

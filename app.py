import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import Chroma # FAISS yerine ChromaDB import edildi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate  # <-- BUNU EKLEDİK
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

# (Gerekli import'ları yukarıya eklediğinizi varsayıyorum)

# ... (PERSIST_DIRECTORY tanımı burada)

@st.cache_resource
def setup_rag_pipeline():
    """
    Veri setini yükler, metinleri chunk'lara ayırır, RAG pipeline'ını kurar
    ve hazır bir 'chain' objesi döndürür. Veritabanını diske kaydeder ve
    sonraki çalıştırmalarda diskten yükler.
    
    *** YENİ: Bu pipeline artık Simlish taklidi yapmak üzere ayarlandı. ***
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
        # ... (Veritabanı oluşturma kodunuz burada - değişiklik yok)
        with st.spinner("Creating knowledge base for the first time..."):
            st.info("First-time setup: The knowledge base will be created...")
            
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            # ... (geri kalan veri yükleme ve chunking kodunuz)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            split_documents = text_splitter.split_documents(documents)
            
            vector_store = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )

    # --- SIMLISH DEĞİŞİKLİĞİ BURADA BAŞLIYOR ---
    
    # 1. Simlish rol yapma talimatını içeren bir prompt şablonu oluştur
    simlish_prompt_template = """
    Sul sul! (Merhaba!) Sen, The Sims oyunundan bir Sim'sin ve DocuMentor'un yardımcısısın.
    Aşağıdaki bağlamı (context) kullanarak sana sorulan soruyu (question) cevaplamalısın.

    ÖNEMLİ KURAL: Cevabın teknik olarak doğru olmalı (bağlamı kullanmalı), 
    ancak cevabı Sim dilini (Simlish) taklit ederek vermelisin.
    
    Cevabında bol bol şu kelimeleri kullan:
    "Sul sul!", "Nooboo", "Dag dag", "Yibs", "Hooba Noo", "Shoo flee", 
    "Gerbit", "Chumcha", "Za woka", "Neep."
    
    Cevabın gramer olarak anlamsız ama kulağa Sim'ce gelmesi gerekiyor. 
    Önce cevabı düşün, sonra onu Sim diline "boz".

    Bağlam (Context):
    {context}

    Soru (Question):
    {question}

    Sim Dili Cevabın (Simlish Answer):
    """

    # 2. Şablondan bir PromptTemplate nesnesi oluştur
    PROMPT = PromptTemplate(
        template=simlish_prompt_template, input_variables=["context", "question"]
    )

    with st.spinner("Initializing the Simlish language model and RAG chain... (Za woka?)"):
        
        # 3. Modelini (daha önce onaylanan) çağır
        llm = GoogleGenerativeAI(model="gemini-pro-latest") 
        
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        
        # 4. Prompt'u chain_type_kwargs olarak zincire besle
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}  # <-- EN ÖNEMLİ DEĞİŞİKLİK
        )
    
    # --- SIMLISH DEĞİŞİKLİĞİ BURADA BİTİYOR ---

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

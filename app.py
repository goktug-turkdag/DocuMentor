import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate  # Prompt şablonunu import ediyoruz
import os

# --- 1. KURULUM VE VERİ YÜKLEME ---

# API anahtarını .env dosyasından yükle
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# Kalıcı veritabanının saklanacağı klasör
PERSIST_DIRECTORY = "chroma_db_multilingual"

# --- YENİ MİMARİ: PAHALI KISIMLARI ÖNBELLEĞE AL ---

@st.cache_resource
def load_vector_store():
    """
    SADECE pahalı olan veritabanı yükleme/oluşturma işini yapar.
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
        # Veritabanı yoksa oluştur (ilk çalıştırma)
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
    SADECE dil modelini (LLM) yükler ve önbelleğe alır.
    """
    # Listenizden onaylanan modeli kullanıyoruz
    return GoogleGenerativeAI(model="gemini-pro-latest")

# --- PROMPT ŞABLONLARINI TANIMLA ---

# Normal (Varsayılan) Cevap Şablonu
default_prompt_template = """
Aşağıdaki bağlamı kullanarak soruyu cevaplayın. 
Eğer bilmiyorsanız, bilmediğinizi söyleyin. Bağlama sadık kalın.

Bağlam:
{context}

Soru:
{question}

Yardımcı Cevap:
"""

# Simlish (Eğlenceli) Cevap Şablonu
simlish_prompt_template = """
Sul sul! (Merhaba!) Sen, The Sims oyunundan bir Sim'sin ve DocuMentor'un yardımcısısın.
Aşağıdaki bağlamı (context) kullanarak sana sorulan soruyu (question) cevaplamalısın.

ÖNEMLİ KURAL: Cevabın teknik olarak doğru olmalı (bağlamı kullanmalı), 
ancak cevabı Sim dilini (Simlish) taklit ederek vermelisin.
Cevabında bol bol "Sul sul!", "Nooboo", "Dag dag", "Yibs", "Hooba Noo", 
"Shoo flee", "Gerbit", "Chumcha", "Za woka", "Neep" kelimelerini kullan.

Bağlam:
{context}

Soru:
{question}

Sim Dili Cevabın:
"""

# --- 2. WEB ARAYÜZÜ (GÜNCELLENMİŞ) ---

st.title("DocuMentor 📄")
st.markdown("Akıllı bir Soru-Cevap Chatbot'u. Başlamak için bir soru sorun.")

# --- YENİ: Simlish Modu Düğmesi ---
# Bu düğme, session_state'i kullanarak durumunu korur
st.toggle("Simlish Mode 👽", key="simlish_mode", help="Sul sul! Cevapları Simlish al.")


# --- DİNAMİK RAG ZİNCİRİ OLUŞTURMA ---
try:
    # 1. Pahalı bileşenleri önbellekten hızla yükle
    vector_store = load_vector_store()
    llm = load_llm()

    # 2. Düğmenin durumuna göre doğru prompt'u seç
    if st.session_state.simlish_mode:
        PROMPT_TEMPLATE = simlish_prompt_template
        st.caption("✨ *Simlish modu aktif! Za woka?*")
    else:
        PROMPT_TEMPLATE = default_prompt_template

    # 3. Seçilen prompt ile bir PromptTemplate nesnesi oluştur
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # 4. RAG zincirini (hızlıca) oluştur
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    st.success("DocuMentor is ready! (Ve belki biraz Simlish!)")
    
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.exception(e)
    st.stop()


# --- 3. CHATBİLEŞENİ (DEĞİŞİKLİK YOK) ---

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları yazdır
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

# Kullanıcıdan yeni soru al
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Searching for the answer... (Chumcha!)"):
        try:
            # rag_chain artık DİNAMİK olarak doğru prompt'u kullanıyor
            response_dict = rag_chain.invoke(user_question)
            answer = response_dict.get("result", "Sorry, I couldn't generate an answer.")
            sources = response_dict.get("source_documents", [])
            response_content = {"answer": answer, "sources": sources}

        except Exception as e:
            answer = f"An error occurred while generating the response: {e}"
            response_content = {"answer": answer, "sources": []}

    # Asistan cevabını göster
    with st.chat_message("assistant"):
        st.markdown(response_content["answer"])
        sources = response_content.get("sources", [])
        if sources:
            with st.expander("Sources considered:"):
                for i, source in enumerate(sources):
                    st.markdown(f"*{i+1}. {source.page_content[:200]}...*")
                         
    st.session_state.messages.append({"role": "assistant", "content": response_content})

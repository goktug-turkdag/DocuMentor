import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import os

# --- YENİ KÜTÜPHANE IMPORTLARI ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Chat modeli (streaming için daha iyi)
from langchain.chains import ConversationalRetrievalChain # Hafızalı zincir
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage # Sohbet geçmişi için

# Dosya yükleme için loader'lar
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader 
# --- BİTİŞ: YENİ IMPORTLAR ---

# --- 1. SETUP AND CACHING ---

# Load API key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

PERSIST_DIRECTORY = "chroma_db_multilingual"

# --- PAHALI PARÇALARI ÖNBELLEĞE AL ---
@st.cache_resource
def get_embeddings():
    """Embedding modelini yükler (Pahalı)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def load_llm():
    """LLM'i yükler (Pahalı)"""
    # Streaming için Chat modelini kullanıyoruz
    return ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.6)

@st.cache_resource
def load_default_retriever(_embeddings):
    """
    Varsayılan (Dolly-15k) bilgi tabanını yükler. (Pahalı)
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        with st.spinner("Creating default knowledge base for the first time..."):
            st.info("First-time setup: The default (Dolly-15k) knowledge base is being created.")
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
                embedding=_embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
    else:
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=_embeddings
        )
    return vector_store.as_retriever(search_kwargs={'k': 3})

# --- YENİ FONKSİYON: YÜKLENEN DOSYAYI İŞLEME ---
@st.cache_data(max_entries=5) # Son 5 yüklenen dosyayı hafızada tut
def process_uploaded_file(uploaded_file):
    """
    Yüklenen dosyayı okur, parçalar ve geçici bir vektör veritabanı (retriever) oluşturur.
    """
    if uploaded_file is None:
        return None

    # Dosyayı geçici bir yere yazmak, loader'ların okuyabilmesi için en stabil yoldur
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Dosya uzantısına göre doğru loader'ı seç
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None

        documents = loader.load()
        
        # Metinleri parçala (chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        
        # Yüklenen dosya için *geçici* bir vektör veritabanı oluştur
        embeddings = get_embeddings() # Önbellekten al
        vector_store = Chroma.from_documents(split_documents, embeddings)
        
        st.success(f"Successfully processed '{uploaded_file.name}'. You can now ask questions about it.")
        
        # Retriever'ı (arayıcıyı) döndür
        return vector_store.as_retriever(search_kwargs={'k': 3})

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None
    finally:
        # Geçici dosyayı sil
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- PROMPT ŞABLONLARI ---

# Normal Prompt
default_prompt_template = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know. Stick to the context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Helpful Answer:
"""

# Simlish Prompt
simlish_prompt_template = """
Sul sul! You are a Sim from The Sims game.
Use the following context to answer the question, but you must answer by imitating Simlish.
Be technically correct, but sound like a Sim.
Use these words: "Sul sul!", "Nooboo", "Dag dag", "Yibs", "Hooba Noo", 
"Shoo flee", "Gerbit", "Chumcha", "Za woka", "Neep."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Simlish Answer:
"""

# --- 2. WEB ARAYÜZÜ (TAMAMEN YENİLENDİ) ---

st.title("DocuMentor 📄")
st.markdown("An intelligent Q&A Chatbot trained on the **Databricks Dolly 15k** dataset. Ask a question, or upload your own document!")

# --- Sidebar ---
with st.sidebar:
    st.header("About DocuMentor")
    st.markdown(
        "DocuMentor is an intelligent Q&A chatbot built using a RAG "
        "architecture with Google's Gemini and ChromaDB."
    )
    
    st.markdown("---")
    
    st.subheader("Developed by Göktuğ Türkdağ")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)")
    st.markdown("🐙 [GitHub](https://github.com/goktug-turkdag)")
    
    st.markdown("---")

    # --- YENİ ÖZELLİK 4: SOHBETİ TEMİZLE BUTONU ---
    if st.button("Clear Chat History 🧹"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.file_retriever = None # Yüklenen dosya hafızasını da temizle
        st.success("Chat history cleared!")
        st.rerun() # Sayfayı temizle

    st.markdown("---")
    
    # --- YENİ ÖZELLİK 2: DOSYA YÜKLEME ---
    st.subheader("Upload Your Document")
    st.markdown("Chat with your own PDF, DOCX, or TXT file.")
    uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

# --- Simlish Mode Toggle ---
st.toggle("Simlish Mode 👽", key="simlish_mode", help="Sul sul! Get your answers in Simlish.")

# --- ANA SOHBET MANTIĞI ---

# Pahalı bileşenleri yükle
llm = load_llm()
default_retriever = load_default_retriever(get_embeddings())

# --- Session State (Oturum Hafızası) Başlatma ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = [] # Hafızalı sohbet için
    st.session_state.file_retriever = None # Yüklenen dosyanın retriever'ı
    
    # Karşılama Mesajı
    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    I'm trained on the **Databricks Dolly 15k** dataset, but you can also **upload your own document** in the sidebar!
    
    - Try asking me: "Who is Göktuğ Türkdağ?"
    - Or toggle **Simlish Mode** 👽
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# --- Dosya Yükleme Mantığı ---
# Eğer yeni bir dosya yüklendiyse, onu işle ve retriever'ı session state'e kaydet
if uploaded_file:
    # Dosyanın daha önce işlenip işlenmediğini kontrol et
    if "processed_file_name" not in st.session_state or st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
            file_retriever = process_uploaded_file(uploaded_file)
            if file_retriever:
                st.session_state.file_retriever = file_retriever
                st.session_state.processed_file_name = uploaded_file.name
                # Dosya değiştiğinde sohbet geçmişini temizle
                st.session_state.messages = [{"role": "assistant", "content": f"OK, I'm ready to answer questions about '{uploaded_file.name}'."}]
                st.session_state.chat_history = []
                st.rerun() # Arayüzü temizle

# Hangi retriever'ın (bilgi tabanının) aktif olduğunu belirle
if st.session_state.file_retriever is not None:
    active_retriever = st.session_state.file_retriever
    st.caption(f"ℹ️ *Querying document: {st.session_state.processed_file_name}*")
else:
    active_retriever = default_retriever

# --- Dinamik RAG Zinciri Oluşturma ---

# 1. Simlish modu için doğru prompt'u seç
if st.session_state.simlish_mode:
    PROMPT_TEMPLATE = simlish_prompt_template
    st.caption("✨ *Simlish mode active! Za woka?*")
else:
    PROMPT_TEMPLATE = default_prompt_template

# 2. Seçilen prompt ile bir PromptTemplate nesnesi oluştur
COMBINE_DOCS_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

# 3. YENİ ÖZELLİK 1: HAFİALI SOHBET ZİNCİRİ (ConversationalRetrievalChain)
# Bu zincir, sohbet geçmişini (chat_history) kullanarak takip sorularını anlar
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=active_retriever,
    combine_docs_chain_kwargs={"prompt": COMBINE_DOCS_PROMPT},
    return_source_documents=True # Kaynakları göstermek için
)

# --- Sohbet Arayüzü ---

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Kaynakları göster (eğer varsa)
        if "sources" in message:
            with st.expander("Sources considered:"):
                for src in message["sources"]:
                    st.markdown(f"*{src.page_content[:200]}...*")

# Yeni kullanıcı sorusu al
if user_question := st.chat_input("Ask a question..."):
    # Kullanıcı mesajını ekle
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # --- Easter Egg ---
    lower_question = user_question.lower()
    creator_keywords = ["göktuğ", "türkdağ", "geliştirici", "developer", "who made you", "who created you"]

    if any(keyword in lower_question for keyword in creator_keywords):
        response_text = f"""
        Ah, a great question! I was developed by **Göktuğ Türkdağ**. 🤖
        He's a developer specializing in RAG architectures, LLMs, and Python. 
        You can find him here:
        🔗 **LinkedIn:** [linkedin.com/in/goktugturkdag](https://www.linkedin.com/in/goktugturkdag)
        🐙 **GitHub:** [github.com/goktug-turkdag](https://github.com/goktug-turkdag)
        """
        
        # Easter egg cevabını UI'a ve hafızaya ekle
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response_text))

    else:
        # --- YENİ ÖZELLİK 3: STREAMING (YAZIYOR...) EFEKTİ ---
        
        # Spinner (dönme) metnini ayarla
        spinner_text = "Searching for the answer... (Chumcha!)" if st.session_state.simlish_mode else "Searching for the answer..."
        
        with st.spinner(spinner_text):
            # Cevabı UI'da yazmak için bir "boş kutu" oluştur
            with st.chat_message("assistant"):
                response_container = st.empty()
                
                # Zinciri .stream() ile çağır (Hafızalı sohbet için chat_history'i gönder)
                stream = rag_chain.stream({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                
                full_response = ""
                sources = []
                
                # Gelen cevabı kelime kelime yakala
                for chunk in stream:
                    # 'answer' anahtarı cevap metnini içerir
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_container.markdown(full_response + "▌") # "Yazıyor" imleci
                
                # 'source_documents' anahtarı kaynakları içerir (genellikle en sonda gelir)
                    if "source_documents" in chunk:
                        sources = chunk["source_documents"]
                
                # Yazma bittiğinde, imleci kaldır
                response_container.markdown(full_response)

        # Cevabı ve kaynakları UI mesaj listesine ekle
        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
        # Cevabı zincirin hafızasına ekle
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=full_response))

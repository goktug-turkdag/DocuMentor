import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import os
import tempfile # Dosyaları geçici işlemek için

# --- GEREKLİ LANGCHAIN IMPORTLARI ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Chat modeli
from langchain.chains import ConversationalRetrievalChain # Hafızalı zincir
# --- DÜZELTME: Import yolu güncellendi ---
from langchain.chains.summarize import create_summarization_chain # YENİ: Özetleyici için
# --- DÜZELTME BİTTİ ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- DOSYA OKUYUCULAR ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader 

# --- 1. SETUP, CACHING VE API ANAHTARLARI ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

PERSIST_DIRECTORY = "chroma_db_multilingual"

# --- ÖNBELLEKLEME (PAHALI İŞLEMLER) ---
@st.cache_resource
def get_embeddings():
    """Embedding modelini yükler"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def load_llm():
    """LLM'i yükler"""
    return ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.6)

@st.cache_resource
def load_default_retriever(_embeddings):
    """Varsayılan (Dolly-15k) bilgi tabanını yükler."""
    if not os.path.exists(PERSIST_DIRECTORY):
        with st.spinner("Creating default knowledge base for the first time..."):
            st.info("First-time setup: The default (Dolly-15k) knowledge base is being created.")
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            data_with_context = dataset.filter(
                lambda example: example["context"] != "" and len(example["context"]) > 10
            )
            documents = [Document(page_content=item['context'], metadata={"source": f"dolly_15k_item_{i}"})
                         for i, item in enumerate(data_with_context)]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

# --- YENİ ÖZELLİK: ÇOKLU DOSYA İŞLEME ---
@st.cache_data(max_entries=1)
def process_uploaded_files(uploaded_files):
    """
    Yüklenen (birden fazla) dosyayı okur, parçalar ve geçici bir vektör veritabanı (retriever) oluşturur.
    """
    if not uploaded_files:
        return None

    all_documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(temp_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(temp_path, encoding="utf-8")
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                    continue
                
                all_documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                
    if not all_documents:
        st.error("No documents could be processed.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(split_documents, embeddings)
    
    file_names = ", ".join([f.name for f in uploaded_files])
    st.success(f"Successfully processed {len(uploaded_files)} files: {file_names}.")
    
    return vector_store.as_retriever(search_kwargs={'k': 3})

# --- PROMPT ŞABLONLARI (Hafızalı sohbet için güncellendi) ---

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

# --- 2. WEB ARAYÜZÜ (SEKMELİ YAPI) ---
st.title("DocuMentor 📄")

# --- YENİ ÖZELLİK: SEKMELER (CHAT vs. ÖZETLEYİCİ) ---
tab_chat, tab_summarize = st.tabs(["💬 Chatbot", "✍️ Document Summarizer"])

# --- Sidebar ---
with st.sidebar:
    st.header("About DocuMentor")
    st.markdown(
        "An intelligent Q&A chatbot built using a RAG "
        "architecture with Google's Gemini and ChromaDB."
    )
    st.markdown("---")
    st.subheader("Developed by Göktuğ Türkdağ")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)")
    st.markdown("🐙 [GitHub](https://github.com/goktug-turkdag)")
    st.markdown("---")
    
    if st.button("Clear Chat History 🧹"):
        st.session_state.clear()
        st.success("Chat history and uploaded files cleared!")
        st.rerun()

    st.markdown("---")
    
    st.subheader("Chat with Your Documents")
    st.markdown("Upload one or multiple PDF, DOCX, or TXT files.")
    uploaded_files = st.file_uploader(
        "Upload files for chat:", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.toggle("Simlish Mode 👽", key="simlish_mode", help="Sul sul! Get your answers in Simlish.")

# --- BİLEŞENLERİ YÜKLE ---
llm = load_llm()
default_retriever = load_default_retriever(get_embeddings())

# --- SESSION STATE BAŞLATMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.file_retriever = None
    
    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    I'm trained on **Dolly 15k**, but you can also **upload your own documents** in the sidebar to chat with them!
    
    - Try asking me: "Who is Göktuğ Türkdağ?"
    - Or toggle **Simlish Mode** 👽
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# --- SEKME 1: CHATBOT ---
with tab_chat:
    
    if uploaded_files:
        new_file_names = [f.name for f in uploaded_files]
        if "processed_files" not in st.session_state or st.session_state.processed_files != new_file_names:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                file_retriever = process_uploaded_files(uploaded_files)
                if file_retriever:
                    st.session_state.file_retriever = file_retriever
                    st.session_state.processed_files = new_file_names
                    
                    file_names_str = ", ".join(st.session_state.processed_files)
                    st.session_state.messages = [{"role": "assistant", "content": f"OK, I'm ready to answer questions about: '{file_names_str}'."}]
                    st.session_state.chat_history = []
                    st.rerun()

    if st.session_state.file_retriever is not None:
        active_retriever = st.session_state.file_retriever
        file_names_str = ", ".join(st.session_state.processed_files)
        st.caption(f"ℹ️ *Querying document(s): {file_names_str}*")
    else:
        active_retriever = default_retriever

    if st.session_state.simlish_mode:
        PROMPT_TEMPLATE = simlish_prompt_template
        st.caption("✨ *Simlish mode active! Za woka?*")
    else:
        PROMPT_TEMPLATE = default_prompt_template

    COMBINE_DOCS_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=active_retriever,
        combine_docs_chain_kwargs={"prompt": COMBINE_DOCS_PROMPT},
        return_source_documents=True
    )

    avatars = {"human": "👤", "assistant": "👽" if st.session_state.simlish_mode else "🤖"}

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatars.get(message["role"])):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources considered:"):
                    for src in message["sources"]:
                        st.markdown(f"*{src.page_content[:200]}...*")

    if user_question := st.chat_input("Ask a question..."):
        st.chat_message("human", avatar=avatars["human"]).markdown(user_question)
        st.session_state.messages.append({"role": "human", "content": user_question})

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
            
            with st.chat_message("assistant", avatar=avatars["assistant"]):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response_text))

        else:
            spinner_text = "Searching for the answer... (Chumcha!)" if st.session_state.simlish_mode else "Searching for the answer..."
            
            with st.chat_message("assistant", avatar=avatars["assistant"]):
                response_container = st.empty()

                # Stream generator fonksiyonu (SyntaxError Düzeltmesi ile)
                def stream_generator():
                    full_response = ""
                    sources = []
                    
                    stream = rag_chain.stream({
                        "question": user_question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    for chunk in stream:
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            yield full_response + "▌" 
                        if "source_documents" in chunk:
                            sources = chunk["source_documents"]
                    
                    yield full_response
                    return full_response, sources

                returned_values = st.write_stream(stream_generator())

                if returned_values:
                    full_response, sources = returned_values
                else:
                    full_response = "Sorry, I couldn't generate a response."
                    sources = []
                
            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=full_response))

# --- SEKME 2: DOKÜMAN ÖZETLEYİCİ ---
with tab_summarize:
    st.header("✍️ Document Summarizer")
    st.markdown("Upload a document (PDF, DOCX, TXT) and get a quick summary.")
    
    summary_file = st.file_uploader(
        "Upload a document for summary:", 
        type=["pdf", "docx", "txt"], 
        key="summarizer_file"
    )
    
    summary_type = st.selectbox(
        "Select summary type:",
        ["Brief Paragraph", "Bullet Points (Top 5)", "One-sentence Headline"],
        key="summary_type"
    )
    
    if st.button("Generate Summary", key="summarize_button"):
        if summary_file:
            with st.spinner(f"Summarizing '{summary_file.name}'..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{summary_file.name.split('.')[-1]}") as temp_file:
                        temp_file.write(summary_file.getvalue())
                        temp_path = temp_file.name

                    if summary_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    elif summary_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(temp_path)
                    else:
                        loader = TextLoader(temp_path, encoding="utf-8")
                    
                    docs = loader.load()
                    
                    summarize_chain = create_summarization_chain(llm, chain_type="map_reduce")
                    
                    if summary_type == "Bullet Points (Top 5)":
                        prompt_instruction = "Generate a concise summary of the key findings in 5 bullet points."
                    elif summary_type == "One-sentence Headline":
                        prompt_instruction = "Generate a single, descriptive headline for this document."
                    else:
                        prompt_instruction = "Generate a brief, one-paragraph summary of this document."
                    
                    summary_output = summarize_chain.invoke({
                        "input_documents": docs, 
                        "question": prompt_instruction
                    })
                    
                    st.success("Summary Generated!")
                    st.markdown(f"### {summary_type}")
                    st.markdown(summary_output.get("output_text", "Could not generate summary."))
                
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
                finally:
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.warning("Please upload a document to summarize.")

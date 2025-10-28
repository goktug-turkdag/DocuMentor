import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import os
import tempfile
import random # Games için
import time # Cashout mesajı için
from collections import Counter # Video Poker eli kontrolü için

# --- GEREKLİ LANGCHAIN IMPORTLARI ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
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


# --- ÇOKLU DOSYA İŞLEME ---
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


# --- PROMPT ŞABLONLARI ---
# Normal Prompt
default_prompt_template = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know. Stick to the context.
Context: {context}
Chat History: {chat_history}
Question: {question}
Helpful Answer:
"""

# Simlish Prompt
simlish_prompt_template = """
Sul sul! You are a Sim from The Sims game.
Use the following context to answer the question, but you must answer by imitating Simlish.
Be technically correct, but sound like a Sim.
Use these words: "Sul sul!", "Nooboo", "Dag dag", "Yibs", "Hooba Noo",
"Shoo flee", "Gerbit", "Chumcha", "Za woka", "Neep."
Context: {context}
Chat History: {chat_history}
Question: {question}
Simlish Answer:
"""

# --- 2. WEB ARAYÜZÜ (Sekmeli Yapı) ---
st.set_page_config(page_title="DocuMentor", layout="wide")
st.title("DocuMentor 📄")

# --- Sekmeler ---
tab_chat, tab_blackjack, tab_coinflip, tab_roulette, tab_slots, tab_vpoker, tab_music, tab_settings = st.tabs([
    "💬 Chatbot",
    "🃏 Blackjack",
    "🪙 Coin Flip",
    "🎡 Roulette",
    "🎰 Slots",
    " Video Poker",
    "🎶 Music Player",
    "⚙️ Settings"
])

# --- Sidebar ---
with st.sidebar:
    st.header("About DocuMentor")
    st.markdown(
        """
        **DocuMentor** demonstrates advanced application development, combining:
        1.  An **intelligent Q&A chatbot** leveraging cutting-edge AI.
        2.  Several **interactive features & games of chance** demonstrating Python logic and state handling.

        **Technical Architecture & Features:**
        * **UI:** Built with **Streamlit**.
        * **Core Logic:** **Python**.
        * **Chatbot Engine:** RAG architecture, Google Gemini Pro LLM (via LangChain), HuggingFace multilingual embeddings, ChromaDB vector store, Dolly 15k dataset baseline, multi-document upload (.pdf, .docx, .txt), conversational memory, streaming responses, and Simlish mode easter egg.
        * **Interactive Features (Games):** Blackjack (with 4 side bets, dynamic cashout, 5-Card Charlie), Coin Flip, Roulette, Slots, and Video Poker (Jacks or Better) implemented using pure Python logic and Streamlit Session State for complex state management (balance, bets, game flow, history).

        **Developed by Göktuğ Türkdağ.** This project highlights proficiency in building complex, interactive AI applications and sophisticated state management.

        The codebase exceeds **1500+ lines** and is **open-source** on GitHub. Find the repository via the link below!
        """
    )
    st.markdown("---")
    st.subheader("Connect with the Developer")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)")
    st.markdown("🐙 [GitHub](https://github.com/goktug-turkdag)")
    st.markdown("---")

    # Reset fonksiyonlarını global scope'a ekleyelim (Bu satırlar önemli)
    # Bu fonksiyonlar ilgili sekmelerde tanımlanacak
    bj_reset_func = lambda reset_balance=False: None
    cf_reset_func = lambda reset_balance=False: None
    rl_reset_func = lambda reset_balance=False: None
    sl_reset_func = lambda reset_balance=False: None
    vp_reset_func = lambda reset_balance=False: None


    # Butonlar
    if st.button("Clear Chat History 🧹"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.file_retriever = None
        st.session_state.processed_files = []
        st.success("Chat history and uploaded files cleared!")
        st.rerun()

    if st.button("Reset Interactive Features 💰"):
        # Reset fonksiyonlarını çağır
        bj_reset_func(reset_balance=True)
        cf_reset_func(reset_balance=True)
        rl_reset_func(reset_balance=True)
        sl_reset_func(reset_balance=True)
        vp_reset_func(reset_balance=True)

        st.session_state.player_balance = 1000 # Bakiyeyi sıfırla
        st.success("Balance and feature states reset!")
        st.rerun()

    st.markdown("---")

    st.subheader("Chat with Your Documents")
    st.markdown("Upload one or multiple PDF, DOCX, or TXT files.")
    uploaded_files = st.file_uploader(
        "Upload files for chat:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

# --- BİLEŞENLERİ YÜKLE ---
llm = load_llm()
default_retriever = load_default_retriever(get_embeddings())

# --- SESSION STATE BAŞLATMA (GENEL VE OYUNLAR İÇİN) ---
# Sohbet için
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.file_retriever = None
    st.session_state.processed_files = []

    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    I'm trained on **Dolly 15k**, but you can also **upload your own documents** in the sidebar to chat with them!
    Feel free to explore the other tabs for some **interactive features** and **easter eggs**!

    - Try asking me: "Who is Göktuğ Türkdağ?" or "What technologies were used?"
    - Or check the **Settings** tab to enable **Simlish Mode** 👽
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Ortak Bakiye ve Genel Ayarlar
if "player_balance" not in st.session_state:
    st.session_state.player_balance = 1000
if "simlish_mode" not in st.session_state:
    st.session_state.simlish_mode = False
if "bj_deck_count" not in st.session_state:
     st.session_state.bj_deck_count = 6

# Oyunlar için state başlatmaları, ilgili sekmelerin başına taşındı

# --- Yardımcı Fonksiyon: Bakiye Geçmişi ---
def add_history(game_key, bet, outcome, balance):
    history_key = f"{game_key}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    st.session_state[history_key].insert(0, {"bet": bet, "outcome": outcome, "balance": balance})
    st.session_state[history_key] = st.session_state[history_key][:5]

def display_history(game_key):
     history_key = f"{game_key}_history"
     if history_key in st.session_state and st.session_state[history_key]:
        with st.expander("Show Recent History"):
            for entry in st.session_state[history_key]:
                outcome_sign = "+" if entry["outcome"] > 0 else "" if entry["outcome"] == 0 else ""
                outcome_val = entry['outcome'] if entry['outcome'] != 0 else 'Push'
                st.markdown(f"- Bet: {entry['bet']}, Outcome: {outcome_sign}{outcome_val}, New Balance: {entry['balance']}")

# --- İçerik Güvenlik Bariyeri Kelimeleri ---
BANNED_KEYWORDS = [
    "kill", "murder", "bomb", "terror", "illegal", "suicide",
    "hate", "nazi", "racist", "sex", "porn",
    # Daha fazla kelime eklenebilir
]

# --- SEKME 1: CHATBOT ---
with tab_chat:

    # Dosya yükleme mantığı
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

    # Aktif retriever'ı seç
    if st.session_state.file_retriever is not None:
        active_retriever = st.session_state.file_retriever
        file_names_str = ", ".join(st.session_state.get("processed_files", []))
        st.caption(f"ℹ️ *Querying document(s): {file_names_str}*")
    else:
        active_retriever = default_retriever

    # Prompt'u seç (Ayarlar sekmesinden kontrol ediliyor)
    if st.session_state.simlish_mode:
        PROMPT_TEMPLATE = simlish_prompt_template
        st.caption("✨ *Simlish mode active! Za woka?*")
    else:
        PROMPT_TEMPLATE = default_prompt_template

    COMBINE_DOCS_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # RAG zincirini oluştur
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=active_retriever,
        combine_docs_chain_kwargs={"prompt": COMBINE_DOCS_PROMPT},
        return_source_documents=True
    )

    # Avatarları tanımla
    avatars = {"human": "👤", "assistant": "👽" if st.session_state.simlish_mode else "🤖"}

    # Sohbet geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatars.get(message["role"])):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources considered:"):
                    for src in message["sources"]:
                        st.markdown(f"*{src.page_content[:200]}...*")

    # Yeni soru al
    if user_question := st.chat_input("Ask a question..."):
        st.chat_message("human", avatar=avatars["human"]).markdown(user_question)
        st.session_state.messages.append({"role": "human", "content": user_question})

        lower_question = user_question.lower()

        # İçerik Güvenlik Bariyeri
        is_safe = True
        for keyword in BANNED_KEYWORDS:
            if keyword in lower_question:
                is_safe = False
                break

        if not is_safe:
            response_text = "I cannot answer questions that are inappropriate, unethical, or promote harmful content. Please ask something else."
            with st.chat_message("assistant", avatar=avatars["assistant"]):
                st.warning(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Gelişmiş Easter Egg (SSS)
        elif any(keyword in lower_question for keyword in ["göktuğ", "türkdağ", "developer", "geliştirici"]):
            response_text = ""
            if any(k in lower_question for k in ["who", "kimdir", "about"]):
                response_text = f"""
                Ah, a great question! I was developed by **Göktuğ Türkdağ**. 🤖
                He's a developer specializing in RAG architectures, Large Language Models (LLMs), and Python application development using frameworks like Streamlit and LangChain.
                You can find his contact details below or ask about the technologies used in this project!
                """
            elif any(k in lower_question for k in ["tech", "skills", "uzmanlık", "teknoloji", "architecture"]):
                 response_text = f"""
                 This DocuMentor application utilizes several modern technologies:
                 * **UI:** Streamlit
                 * **Core:** Python
                 * **AI Engine:** Google Gemini Pro (via LangChain)
                 * **RAG:** LangChain's `ConversationalRetrievalChain`
                 * **Embeddings:** HuggingFace Sentence Transformers (multilingual)
                 * **Vector DB:** ChromaDB
                 * **State Management:** Streamlit Session State

                 Göktuğ Türkdağ has experience implementing these and building complex interactive applications.
                 """
            elif any(k in lower_question for k in ["contact", "iletişim", "connect", "linkedin", "github"]):
                 response_text = f"""
                 You can connect with Göktuğ Türkdağ here:
                 🔗 **LinkedIn:** [linkedin.com/in/goktugturkdag](https://www.linkedin.com/in/goktugturkdag)
                 🐙 **GitHub:** [github.com/goktug-turkdag](https://github.com/goktug-turkdag)
                 """
            else:
                 response_text = "It looks like you're asking about my developer, Göktuğ Türkdağ! You can ask 'Who is Göktuğ Türkdağ?', 'What technologies were used?', or 'How to contact him?'."

            with st.chat_message("assistant", avatar=avatars["assistant"]):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response_text))

        else:
            # Normal RAG cevabı (Streaming)
            spinner_text = "Searching for the answer... (Chumcha!)" if st.session_state.simlish_mode else "Searching for the answer..."

            with st.chat_message("assistant", avatar=avatars["assistant"]):
                response_container = st.empty()

                # Stream generator fonksiyonu (Hata Ayıklama ile)
                def stream_generator():
                    full_response = ""
                    sources = []
                    try:
                        stream = rag_chain.stream({
                            "question": user_question,
                            "chat_history": st.session_state.chat_history
                        })

                        for chunk in stream:
                            if "answer" in chunk and chunk["answer"] is not None:
                                full_response += chunk["answer"]
                                yield full_response + "▌"
                            if "source_documents" in chunk and chunk["source_documents"] is not None:
                                sources = chunk["source_documents"]

                        yield full_response
                        return full_response, sources

                    except Exception as e:
                        error_message = f"Sorry, an error occurred while generating the response: {e}. Please check the API key or network connection."
                        st.error(error_message)
                        yield error_message
                        return error_message, []

                returned_values = st.write_stream(stream_generator())

                if isinstance(returned_values, tuple) and len(returned_values) == 2:
                    full_response, sources = returned_values
                elif isinstance(returned_values, str):
                     full_response = returned_values
                     sources = []
                else:
                    full_response = "Sorry, I couldn't generate a response properly."
                    sources = []

            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=full_response))


# --- SEKME 2: BLACKJACK ---
with tab_blackjack:
    st.header("🃏 Blackjack")
    st.markdown("Place your bet, and optional side bets, to beat the dealer!")

    # State başlatma
    if "game_state" not in st.session_state:
        st.session_state.game_state = "betting"
        st.session_state.deck = []
        st.session_state.player_hand = []
        st.session_state.dealer_hand = []
        st.session_state.game_message = ""
        st.session_state.side_bet_message = ""
        st.session_state.current_bet = 0
        st.session_state.bet_21_3 = 0
        st.session_state.bet_perfect_pairs = 0
        st.session_state.bet_lucky_seven = 0
        st.session_state.bet_bust = 0
        st.session_state.bj_history = []

    with st.expander("Show/Hide Basic Strategy & Side Bet Info"):
        st.markdown("""
        **Basic Blackjack Strategy:** ...
        **Side Bet Payouts:** ...
        **Special Wins:** 5-Card Charlie ...
        """) # Kısaltıldı

    # --- Blackjack Oyun Fonksiyonları ---
    def reset_blackjack_state(reset_balance=False):
        st.session_state.game_state = "betting"
        # ... (Diğer state sıfırlamaları) ...
        st.session_state.deck = []
        st.session_state.player_hand = []
        st.session_state.dealer_hand = []
        st.session_state.game_message = ""
        st.session_state.side_bet_message = ""
        if reset_balance: st.session_state.player_balance = 1000
        st.session_state.current_bet = 0
        st.session_state.bet_21_3 = 0
        st.session_state.bet_perfect_pairs = 0
        st.session_state.bet_lucky_seven = 0
        st.session_state.bet_bust = 0
        st.session_state.bj_history = []
    globals()["bj_reset_func"] = reset_blackjack_state

    def create_deck(num_decks=6):
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'rank': rank, 'suit': suit} for suit in suits for rank in ranks] * st.session_state.bj_deck_count
        random.shuffle(deck)
        return deck

    # ... (get_card_value, calculate_score değişmedi) ...
    def get_card_value(card_rank):
        if card_rank in ['J', 'Q', 'K']: return 10
        if card_rank == 'A': return 11
        return int(card_rank)

    def calculate_score(hand):
        score = sum(get_card_value(card['rank']) for card in hand)
        num_aces = sum(1 for card in hand if card['rank'] == 'A')
        while score > 21 and num_aces > 0:
            score -= 10
            num_aces -= 1
        return score

    def display_hand_visual(hand, title):
        st.subheader(title)
        card_html = ""
        # ... (Kart görselleştirme kodu) ...
        for card in hand:
            color = "red" if card['suit'] in ['♥', '♦'] else "black"
            card_html += f"<div style='border:1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px; display:inline-block; text-align: center; width: 60px; background-color: white; color: {color};'> <span style='font-size: 1.5em; font-weight: bold;'>{card['rank']}</span><br><span style='font-size: 1.5em;'>{card['suit']}</span> </div>"
        st.markdown(card_html, unsafe_allow_html=True)
        score = calculate_score(hand)
        st.markdown(f"**Score: {score}**")
        return score

    def display_dealer_hand_hidden_visual(hand):
        st.subheader("Dealer's Hand")
        card_html = ""
        # ... (Gizli kart görselleştirme kodu) ...
        card = hand[0]
        color = "red" if card['suit'] in ['♥', '♦'] else "black"
        card_html += f"<div style='border:1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px; display:inline-block; text-align: center; width: 60px; background-color: white; color: {color};'> <span style='font-size: 1.5em; font-weight: bold;'>{card['rank']}</span><br><span style='font-size: 1.5em;'>{card['suit']}</span> </div>"
        card_html += f"<div style='border:1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px; display:inline-block; text-align: center; width: 60px; background-color: #aaa; color: #aaa;'> <span style='font-size: 1.5em; font-weight: bold;'>?</span><br><span style='font-size: 1.5em;'>?</span> </div>"
        st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("**Score: ?**")

    # ... (Yan Bahis Kontrol Fonksiyonları - check_perfect_pairs, check_21_plus_3, check_lucky_sevens, check_bust_it değişmedi) ...
    def check_perfect_pairs(hand):
        card1, card2 = hand[0], hand[1]
        if card1['rank'] == card2['rank']:
            if card1['suit'] == card2['suit']: return 25
            card1_color_is_red = card1['suit'] in ['♥', '♦']
            card2_color_is_red = card2['suit'] in ['♥', '♦']
            if card1_color_is_red == card2_color_is_red: return 12
            return 6
        return 0

    def check_21_plus_3(player_hand, dealer_up_card):
        cards = [player_hand[0], player_hand[1], dealer_up_card]
        ranks = [c['rank'] for c in cards]
        suits = [c['suit'] for c in cards]
        is_flush = suits[0] == suits[1] == suits[2]
        is_trips = ranks[0] == ranks[1] == ranks[2]
        rank_map = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13, 'A':14}
        num_ranks = sorted([rank_map[r] for r in ranks])
        is_straight = (num_ranks[0] + 1 == num_ranks[1]) and (num_ranks[1] + 1 == num_ranks[2])
        if num_ranks == [2, 3, 14]: is_straight = True
        if is_straight and is_flush: return 40
        if is_trips: return 30
        if is_straight: return 10
        if is_flush: return 5
        return 0

    def check_lucky_sevens(hand, bet):
        if bet == 0: return 0, ""
        num_sevens = sum(1 for card in hand if card['rank'] == '7')
        if num_sevens == 3:
            winnings = bet * 100
            return winnings + bet, f"**Lucky 7s Win: +{winnings}!** (100:1)"
        elif num_sevens == 2:
            winnings = bet * 50
            return winnings + bet, f"**Lucky 7s Win: +{winnings}!** (50:1)"
        elif num_sevens == 1:
            winnings = bet * 3
            return winnings + bet, f"**Lucky 7s Win: +{winnings}!** (3:1)"
        return 0, "Lucky 7s bet lost."

    def check_bust_it(dealer_hand, dealer_score, bet):
        if bet == 0: return 0, ""
        if dealer_score > 21:
            num_cards = len(dealer_hand)
            if num_cards >= 5: payout = 5
            elif num_cards == 4: payout = 3
            elif num_cards == 3: payout = 2
            else: payout = 0

            if payout > 0:
                winnings = bet * payout
                return winnings + bet, f"**Bust It! Win: +{winnings}!** (Dealer {num_cards} cards, {payout}:1)"

        return 0, "Bust It! bet lost."

    # ... (Dinamik Cashout Hesaplayıcı - get_cashout_offer_heuristic değişmedi) ...
    def get_cashout_offer_heuristic(player_hand, dealer_up_card, bet):
        p_score = calculate_score(player_hand)
        if p_score == 21 or (len(player_hand) >= 5 and p_score <= 21) or p_score > 21:
             return 0

        d_val = get_card_value(dealer_up_card['rank'])
        p_win = 0.48
        if p_score == 20: p_win = 0.85
        elif p_score == 19: p_win = 0.75
        elif p_score == 18: p_win = 0.65
        elif p_score in [12, 13, 14, 15, 16]:
            if d_val in [2, 3, 4, 5, 6]: p_win = 0.40
            elif d_val in [7, 8, 9]: p_win = 0.25
            elif d_val in [10, 11]: p_win = 0.20
        elif p_score == 11:
            if d_val in [10, 11]: p_win = 0.45
            else: p_win = 0.60
        elif p_score == 10:
            if d_val in [10, 11]: p_win = 0.40
            else: p_win = 0.55
        elif p_score == 9:
            if d_val in [10, 11]: p_win = 0.30
            else: p_win = 0.40
        elif p_score == 17 and (get_card_value('A') in [get_card_value(c['rank']) for c in player_hand]):
             p_win = 0.45

        if p_win >= 0.5: multiplier = 1.0 + (p_win - 0.5) * 1.5
        else: multiplier = 1.0 - (0.5 - p_win) * 3.0
        offer = int(bet * multiplier)
        if offer < 0: offer = 0
        if offer > (bet * 1.9): offer = int(bet * 1.9)
        return offer

    # --- Oyun Arayüzü ---
    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    # Bahis Aşaması
    if st.session_state.game_state == "betting":
        # ... (Bahis formu ve Deal butonu mantığı - history eklemeleri dahil) ...
        st.session_state.side_bet_message = ""

        if st.session_state.player_balance <= 0:
            st.error("You are out of money! Reset features from the sidebar.")
        else:
            with st.form(key="bet_form"):
                st.subheader("Place Your Bets")
                bet_amount = st.number_input(
                    "Main Bet:", min_value=10, max_value=st.session_state.player_balance, value=50, step=10
                )

                st.markdown("---")
                st.markdown("**Side Bets (Optional)**")

                max_side_bet = min(100, st.session_state.player_balance)
                bet_21_3_amount = st.number_input("21+3 Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_pp_amount = st.number_input("Perfect Pairs Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_lucky_seven_amount = st.number_input("Lucky 7s Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_bust_amount = st.number_input("Bust It! Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)

                deal_button = st.form_submit_button("Deal")

            if deal_button:
                total_bet = bet_amount + bet_21_3_amount + bet_pp_amount + bet_lucky_seven_amount + bet_bust_amount
                if total_bet > st.session_state.player_balance:
                    st.error(f"Total bet ({total_bet}) cannot exceed your balance ({st.session_state.player_balance}). Please adjust.")
                else:
                    st.session_state.player_balance -= total_bet
                    st.session_state.current_bet = bet_amount
                    st.session_state.bet_21_3 = bet_21_3_amount
                    st.session_state.bet_perfect_pairs = bet_pp_amount
                    st.session_state.bet_lucky_seven = bet_lucky_seven_amount
                    st.session_state.bet_bust = bet_bust_amount

                    st.session_state.deck = create_deck()
                    st.session_state.player_hand = [st.session_state.deck.pop(), st.session_state.deck.pop()]
                    st.session_state.dealer_hand = [st.session_state.deck.pop(), st.session_state.deck.pop()]

                    side_messages = []
                    # PP ve 21+3 kontrolü
                    if st.session_state.bet_perfect_pairs > 0:
                        pp_payout = check_perfect_pairs(st.session_state.player_hand)
                        if pp_payout > 0:
                            winnings = st.session_state.bet_perfect_pairs * pp_payout
                            st.session_state.player_balance += winnings + st.session_state.bet_perfect_pairs
                            side_messages.append(f"**Perfect Pairs Win: +{winnings}!** ({pp_payout}:1)")
                            add_history("bj", st.session_state.bet_perfect_pairs, winnings, st.session_state.player_balance)

                    if st.session_state.bet_21_3 > 0:
                        p3_payout = check_21_plus_3(st.session_state.player_hand, st.session_state.dealer_hand[0])
                        if p3_payout > 0:
                            winnings = st.session_state.bet_21_3 * p3_payout
                            st.session_state.player_balance += winnings + st.session_state.bet_21_3
                            side_messages.append(f"**21+3 Win: +{winnings}!** ({p3_payout}:1)")
                            add_history("bj", st.session_state.bet_21_3, winnings, st.session_state.player_balance)

                    if not side_messages and (st.session_state.bet_21_3 > 0 or st.session_state.bet_perfect_pairs > 0):
                        st.session_state.side_bet_message = "21+3 / Perfect Pairs bets lost."
                    else:
                        st.session_state.side_bet_message = " \n".join(side_messages)

                    # Ana Oyun Blackjack Kontrolü
                    player_score = calculate_score(st.session_state.player_hand)
                    dealer_score = calculate_score(st.session_state.dealer_hand)
                    
                    if player_score == 21 and dealer_score != 21:
                        st.session_state.game_state = "game_over"
                        st.session_state.game_message = "Blackjack! 🎉 You win!"
                        bj_win = int(st.session_state.current_bet * 1.5)
                        st.session_state.player_balance += st.session_state.current_bet + bj_win # Bahis + 3:2 kazanç
                        add_history("bj", st.session_state.current_bet, bj_win, st.session_state.player_balance)
                        
                        l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                        if l7_winnings > 0:
                            st.session_state.player_balance += l7_winnings
                            st.session_state.side_bet_message += "\n" + l7_msg
                            add_history("bj", st.session_state.bet_lucky_seven, l7_winnings - st.session_state.bet_lucky_seven, st.session_state.player_balance)

                    elif player_score == 21 and dealer_score == 21:
                        st.session_state.game_state = "game_over"
                        st.session_state.game_message = "Push! Both have Blackjack. 😐"
                        st.session_state.player_balance += st.session_state.current_bet
                        add_history("bj", st.session_state.current_bet, 0, st.session_state.player_balance)

                    elif dealer_score == 21:
                         st.session_state.game_state = "game_over"
                         st.session_state.game_message = "Dealer has Blackjack. 😕 You lose."
                         add_history("bj", st.session_state.current_bet, -st.session_state.current_bet, st.session_state.player_balance)

                    else:
                        st.session_state.game_state = "player_turn"
                        st.session_state.game_message = "Your turn! Hit, Stand, or Cash Out?"
                    st.rerun()


    # Oyun Akışı
    if st.session_state.game_state in ["player_turn", "dealer_turn", "game_over"]:
        # ... (Oyun akışı mantığı - Hit, Stand, Cashout, Dealer'ın sırası, 5-Card Charlie, history eklemeleri dahil) ...
        # ... (Bu kısım bir önceki kodla aynı) ...
        if st.session_state.side_bet_message:
            st.info(st.session_state.side_bet_message)

        if st.session_state.game_state == "player_turn":
            display_dealer_hand_hidden_visual(st.session_state.dealer_hand) # Görsel kullan
        else:
            display_hand_visual(st.session_state.dealer_hand, "Dealer's Hand") # Görsel kullan

        st.markdown("---")

        player_score = display_hand_visual(st.session_state.player_hand, "Your Hand") # Görsel kullan

        if st.session_state.game_state == "game_over":
            st.header(st.session_state.game_message)
            if st.button("Play Again?", key="play_again"):
                reset_blackjack_state()
                st.rerun()

        if st.session_state.game_state == "player_turn":

            cashout_offer = get_cashout_offer_heuristic(
                st.session_state.player_hand,
                st.session_state.dealer_hand[0],
                st.session_state.current_bet
            )

            col1, col2, col3 = st.columns(3)

            if col1.button("Hit", key="hit"):
                st.session_state.player_hand.append(st.session_state.deck.pop())
                player_score = calculate_score(st.session_state.player_hand)
                num_player_cards = len(st.session_state.player_hand)

                # 5-Card Charlie kontrolü
                if num_player_cards == 5 and player_score <= 21:
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = "5-Card Charlie! 🎉 You win!"
                    charlie_win = st.session_state.current_bet
                    st.session_state.player_balance += st.session_state.current_bet + charlie_win # Bahis + 1:1 kazanç
                    add_history("bj", st.session_state.current_bet, charlie_win, st.session_state.player_balance)

                    l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                    if l7_winnings > 0:
                        st.session_state.player_balance += l7_winnings
                        st.session_state.side_bet_message += "\n" + l7_msg
                        add_history("bj", st.session_state.bet_lucky_seven, l7_winnings - st.session_state.bet_lucky_seven, st.session_state.player_balance)
                elif player_score > 21:
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = "Bust! 💥 You lose."
                    add_history("bj", st.session_state.current_bet, -st.session_state.current_bet, st.session_state.player_balance)

                    l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                    if l7_winnings > 0:
                        st.session_state.player_balance += l7_winnings
                        st.session_state.side_bet_message += "\n" + l7_msg
                        add_history("bj", st.session_state.bet_lucky_seven, l7_winnings - st.session_state.bet_lucky_seven, st.session_state.player_balance)

                st.rerun()

            if col2.button("Stand", key="stand"):
                st.session_state.game_state = "dealer_turn"
                l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                if l7_winnings > 0:
                    st.session_state.player_balance += l7_winnings
                    st.session_state.side_bet_message += "\n" + l7_msg
                    add_history("bj", st.session_state.bet_lucky_seven, l7_winnings - st.session_state.bet_lucky_seven, st.session_state.player_balance)
                st.rerun()

            if cashout_offer > 0:
                if col3.button(f"Cash Out for 💰 {cashout_offer}", key="cashout"):
                    cashout_profit = cashout_offer - st.session_state.current_bet
                    st.session_state.player_balance += cashout_offer
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = f"You cashed out for {cashout_offer}! (Bet was {st.session_state.current_bet})"
                    add_history("bj", st.session_state.current_bet, cashout_profit, st.session_state.player_balance)
                    st.rerun()

    if st.session_state.game_state == "dealer_turn":
        dealer_score = calculate_score(st.session_state.dealer_hand)

        while dealer_score < 17:
            st.session_state.dealer_hand.append(st.session_state.deck.pop())
            dealer_score = calculate_score(st.session_state.dealer_hand)

        player_score = calculate_score(st.session_state.player_hand)
        bet = st.session_state.current_bet

        # Bust It! kontrolü
        bust_winnings, bust_msg = check_bust_it(st.session_state.dealer_hand, dealer_score, st.session_state.bet_bust)
        if bust_winnings > 0:
            st.session_state.player_balance += bust_winnings
            st.session_state.side_bet_message += "\n" + bust_msg
            add_history("bj", st.session_state.bet_bust, bust_winnings - st.session_state.bet_bust, st.session_state.player_balance)
        elif st.session_state.bet_bust > 0:
             st.session_state.side_bet_message += "\nBust It! bet lost."

        # Ana bahsi öde ve mesajı oluştur
        if dealer_score > 21:
            st.session_state.game_message = "Dealer busts! 🎉 You win!"
            win_amount = bet
            st.session_state.player_balance += bet + win_amount # Bahis + Kazanç
            add_history("bj", bet, win_amount, st.session_state.player_balance)
        elif dealer_score > player_score:
            st.session_state.game_message = "Dealer wins. 😕"
            add_history("bj", bet, -bet, st.session_state.player_balance)
        elif player_score > dealer_score:
            st.session_state.game_message = "🎉 You win!"
            win_amount = bet
            st.session_state.player_balance += bet + win_amount
            add_history("bj", bet, win_amount, st.session_state.player_balance)
        else:
            st.session_state.game_message = "It's a tie! (Push) 😐"
            st.session_state.player_balance += bet
            add_history("bj", bet, 0, st.session_state.player_balance)

        st.session_state.game_state = "game_over"
        st.rerun()


    # Bakiye Geçmişi
    display_history("bj")


# --- SEKME 3: COIN FLIP ---
with tab_coinflip:
    st.header("🪙 Coin Flip")
    st.markdown("A simple Heads or Tails betting game.")

    # State başlatma
    if "coin_flip_result" not in st.session_state:
        st.session_state.coin_flip_result = ""
        st.session_state.coin_flip_message = ""
        st.session_state.last_coin_flip_bet = 10
        st.session_state.last_coin_flip_choice = "Heads"
        st.session_state.cf_history = []

    def reset_coin_flip_state(reset_balance=False):
        st.session_state.coin_flip_result = ""
        st.session_state.coin_flip_message = ""
        st.session_state.last_coin_flip_bet = 10
        st.session_state.last_coin_flip_choice = "Heads"
        st.session_state.cf_history = []
        if reset_balance: st.session_state.player_balance = 1000
    globals()["cf_reset_func"] = reset_coin_flip_state


    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset features from the sidebar.")
    else:
        st.caption(f"Last bet: {st.session_state.last_coin_flip_bet} on {st.session_state.last_coin_flip_choice}")

        with st.form("coin_flip_form"):
            cf_bet = st.number_input("Bet Amount:", min_value=1, max_value=st.session_state.player_balance,
                                     value=st.session_state.last_coin_flip_bet, step=1)
            cf_choice_options = ["Heads", "Tails"]
            cf_choice_index = cf_choice_options.index(st.session_state.last_coin_flip_choice) if st.session_state.last_coin_flip_choice in cf_choice_options else 0
            cf_choice = st.radio("Choose:", cf_choice_options, index=cf_choice_index, horizontal=True)

            flip_button = st.form_submit_button("Flip Coin")

        if flip_button:
            st.session_state.player_balance -= cf_bet
            result = random.choice(["Heads", "Tails"])
            st.session_state.coin_flip_result = result
            st.session_state.last_coin_flip_bet = cf_bet
            st.session_state.last_coin_flip_choice = cf_choice

            if result == cf_choice:
                winnings = cf_bet # 1:1 payout
                st.session_state.player_balance += cf_bet + winnings # Bet + Winnings
                st.session_state.coin_flip_message = f"🎉 It's {result}! You win {winnings}!"
                add_history("cf", cf_bet, winnings, st.session_state.player_balance)
                st.balloons()
            else:
                st.session_state.coin_flip_message = f"😕 It's {result}. You lost."
                add_history("cf", cf_bet, -cf_bet, st.session_state.player_balance)

            st.rerun()

    if st.session_state.coin_flip_result:
        st.subheader(f"Result: {st.session_state.coin_flip_result}")
        st.write(st.session_state.coin_flip_message)

    display_history("cf")


# --- SEKME 4: ROULETTE ---
with tab_roulette:
    st.header("🎡 Roulette (European)")
    st.markdown("Place your bets on the table and spin the wheel!")

    # State başlatma
    if "roulette_bets" not in st.session_state:
        st.session_state.roulette_bets = {}
        st.session_state.roulette_result = ""
        st.session_state.roulette_message = ""
        st.session_state.last_roulette_bets = {}
        st.session_state.rl_history = []

    def reset_roulette_state(reset_balance=False):
        st.session_state.roulette_bets = {}
        st.session_state.roulette_result = ""
        st.session_state.roulette_message = ""
        st.session_state.last_roulette_bets = {}
        st.session_state.rl_history = []
        if reset_balance: st.session_state.player_balance = 1000
    globals()["rl_reset_func"] = reset_roulette_state

    numbers = list(range(37))
    red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    black_numbers = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

    def get_color(number):
        if number == 0: return "Green"
        if number in red_numbers: return "Red"
        if number in black_numbers: return "Black"
        return ""

    def get_odd_even(number):
        if number == 0: return ""
        return "Odd" if number % 2 != 0 else "Even"

    def get_low_high(number):
        if number == 0: return ""
        return "Low" if 1 <= number <= 18 else "High"

    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset features from the sidebar.")
    else:
        st.subheader("Place Your Bets:")

        if st.session_state.last_roulette_bets:
            with st.expander("Show/Repeat Last Bets"):
                last_bets_str = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in st.session_state.last_roulette_bets.items()])
                st.markdown(last_bets_str)

        current_bets = {}
        total_current_bet = 0

        num_cols = st.columns(4)
        selected_number = num_cols[0].selectbox("Number (0-36):", ["-"] + numbers, key="r_num_select")
        last_num_key = next((k for k in st.session_state.last_roulette_bets if k.startswith("number_")), None)
        default_num_bet = st.session_state.last_roulette_bets.get(last_num_key, 0) if last_num_key else 0
        if selected_number != "-":
            number_bet = num_cols[1].number_input("Bet on Number (35:1):", min_value=0, value=default_num_bet, step=1, key="r_num_bet")
            if number_bet > 0:
                current_bets[f"number_{selected_number}"] = number_bet
                total_current_bet += number_bet

        st.markdown("**Outside Bets (1:1 Payout)**")
        ext_cols = st.columns(3)

        red_bet = ext_cols[0].number_input("Bet on Red:", min_value=0, value=st.session_state.last_roulette_bets.get("Red", 0), step=1, key="r_red_bet")
        if red_bet > 0: current_bets["Red"] = red_bet; total_current_bet += red_bet
        black_bet = ext_cols[0].number_input("Bet on Black:", min_value=0, value=st.session_state.last_roulette_bets.get("Black", 0), step=1, key="r_black_bet")
        if black_bet > 0: current_bets["Black"] = black_bet; total_current_bet += black_bet

        odd_bet = ext_cols[1].number_input("Bet on Odd:", min_value=0, value=st.session_state.last_roulette_bets.get("Odd", 0), step=1, key="r_odd_bet")
        if odd_bet > 0: current_bets["Odd"] = odd_bet; total_current_bet += odd_bet
        even_bet = ext_cols[1].number_input("Bet on Even:", min_value=0, value=st.session_state.last_roulette_bets.get("Even", 0), step=1, key="r_even_bet")
        if even_bet > 0: current_bets["Even"] = even_bet; total_current_bet += even_bet

        low_bet = ext_cols[2].number_input("Bet on Low (1-18):", min_value=0, value=st.session_state.last_roulette_bets.get("Low", 0), step=1, key="r_low_bet")
        if low_bet > 0: current_bets["Low"] = low_bet; total_current_bet += low_bet
        high_bet = ext_cols[2].number_input("Bet on High (19-36):", min_value=0, value=st.session_state.last_roulette_bets.get("High", 0), step=1, key="r_high_bet")
        if high_bet > 0: current_bets["High"] = high_bet; total_current_bet += high_bet

        st.markdown(f"**Total Bet: {total_current_bet}**")

        if st.button("Spin Wheel", key="spin_roulette"):
            if total_current_bet <= 0:
                st.warning("Please place at least one bet.")
            elif total_current_bet > st.session_state.player_balance:
                st.error(f"Total bet ({total_current_bet}) cannot exceed your balance ({st.session_state.player_balance}).")
            else:
                st.session_state.player_balance -= total_current_bet
                st.session_state.roulette_bets = current_bets
                st.session_state.last_roulette_bets = current_bets

                winning_number = random.randint(0, 36)
                winning_color = get_color(winning_number)
                winning_odd_even = get_odd_even(winning_number)
                winning_low_high = get_low_high(winning_number)

                st.session_state.roulette_result = f"**{winning_number} {winning_color}**"

                total_winnings = 0
                winning_messages = []
                net_outcome = -total_current_bet # Başlangıçta kaybetti varsay

                for bet_type, bet_amount in st.session_state.roulette_bets.items():
                    win = False
                    payout_ratio = 0

                    if bet_type.startswith("number_"):
                        bet_num = int(bet_type.split("_")[1])
                        if bet_num == winning_number: win = True; payout_ratio = 35
                    elif bet_type == "Red" and winning_color == "Red": win = True; payout_ratio = 1
                    elif bet_type == "Black" and winning_color == "Black": win = True; payout_ratio = 1
                    elif bet_type == "Odd" and winning_odd_even == "Odd": win = True; payout_ratio = 1
                    elif bet_type == "Even" and winning_odd_even == "Even": win = True; payout_ratio = 1
                    elif bet_type == "Low" and winning_low_high == "Low": win = True; payout_ratio = 1
                    elif bet_type == "High" and winning_low_high == "High": win = True; payout_ratio = 1

                    if win:
                        winnings = bet_amount * payout_ratio
                        total_winnings += winnings + bet_amount # Bahsi geri iade et
                        net_outcome += winnings + bet_amount # Net kazancı güncelle
                        winning_messages.append(f"Win on {bet_type.replace('_', ' ').title()}: +{winnings}!")

                if total_winnings > 0:
                    st.session_state.player_balance += total_winnings
                    st.session_state.roulette_message = "🎉 **Winning Bets:**\n" + "\n".join(winning_messages)
                    st.balloons()
                else:
                    st.session_state.roulette_message = "😕 No winning bets this round."

                add_history("rl", total_current_bet, net_outcome, st.session_state.player_balance)
                st.rerun()

    if st.session_state.roulette_result:
        st.subheader(f"Wheel Result: {st.session_state.roulette_result}")
        st.write(st.session_state.roulette_message)
        st.button("Place New Bets", on_click=lambda: st.session_state.update({"roulette_result":"", "roulette_message":"", "roulette_bets":{}}))

    display_history("rl")


# --- SEKME 5: SLOTS ---
with tab_slots:
    st.header("🎰 Simple Slots")
    st.markdown("Spin the reels and try to match the symbols on the middle line!")

    # Slot Sembolleri ve Olasılıkları
    symbols = ["🍒", "🍋", "🍊", "🍉", "⭐", "💎", "❼"]
    weights = [   25,   20,   18,   15,   10,    7,   5]

    # Payout Oranları
    payouts = {
        "🍒": {2: 2, 3: 5},
        "🍋": {3: 10},
        "🍊": {3: 15},
        "🍉": {3: 20},
        "⭐": {3: 50},
        "💎": {3: 75},
        "❼": {3: 100}
    }

    # State başlatma
    if "slot_state" not in st.session_state:
        st.session_state.slot_state = "ready"
        st.session_state.slot_reels = ["❓", "❓", "❓"]
        st.session_state.slot_message = "Place your bet and spin!"
        st.session_state.slot_history = []
        st.session_state.last_slot_bet = 5

    def reset_slots_state(reset_balance=False):
        st.session_state.slot_state = "ready"
        st.session_state.slot_reels = ["❓", "❓", "❓"]
        st.session_state.slot_message = "Place your bet and spin!"
        st.session_state.slot_history = []
        st.session_state.last_slot_bet = 5
        if reset_balance: st.session_state.player_balance = 1000
    globals()["sl_reset_func"] = reset_slots_state

    def spin_reels():
        return random.choices(symbols, weights=weights, k=3)

    def check_win(reels, bet):
        middle_symbol = reels[1]

        # Üçlü eşleşme
        if reels[0] == middle_symbol == reels[2]:
            symbol = middle_symbol
            if symbol in payouts and 3 in payouts[symbol]:
                multiplier = payouts[symbol][3]
                winnings = bet * multiplier
                return winnings, f"🎉 JACKPOT! Three {symbol}! Win {winnings} ({multiplier}x)!"

        # İkili kiraz (sadece ortada başlarsa)
        if (reels[0] == "🍒" == reels[1]) or (reels[1] == "🍒" == reels[2]):
             if "🍒" in payouts and 2 in payouts["🍒"]:
                 multiplier = payouts["🍒"][2]
                 winnings = bet * multiplier
                 return winnings, f"🍒 Two Cherries! Win {winnings} ({multiplier}x)!"

        return 0, "😕 No win this time."

    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset features from the sidebar.")
    else:
        slot_bet = st.number_input("Bet Amount per Spin:", min_value=1, max_value=st.session_state.player_balance,
                                     value=st.session_state.last_slot_bet, step=1, key="slot_bet")

        if st.button("Spin Reels!", key="spin_slots", disabled=(st.session_state.slot_state == "spinning")):
            st.session_state.slot_state = "spinning"
            st.session_state.player_balance -= slot_bet
            st.session_state.last_slot_bet = slot_bet
            st.session_state.slot_message = "Spinning..."
            st.rerun()

    # Çarkları göster
    reel_cols = st.columns(3)
    for i, symbol in enumerate(st.session_state.slot_reels):
         with reel_cols[i].container(border=True):
              st.markdown(f"<h1 style='text-align: center; font-size: 4em;'>{symbol}</h1>", unsafe_allow_html=True)

    # Sonuçları işle
    if st.session_state.slot_state == "spinning":
        st.session_state.slot_reels = spin_reels()
        winnings, message = check_win(st.session_state.slot_reels, st.session_state.last_slot_bet)

        if winnings > 0:
            st.session_state.player_balance += winnings + st.session_state.last_slot_bet
            add_history("slot", st.session_state.last_slot_bet, winnings, st.session_state.player_balance)
            st.balloons()
        else:
             add_history("slot", st.session_state.last_slot_bet, -st.session_state.last_slot_bet, st.session_state.player_balance)

        st.session_state.slot_message = message
        st.session_state.slot_state = "result"
        st.rerun()

    if st.session_state.slot_state == "result":
        st.markdown(f"**{st.session_state.slot_message}**")
        st.session_state.slot_state = "ready"

    display_history("slot")

# --- SEKME 6: VIDEO POKER ---
with tab_vpoker:
    st.header(" Jacks or Better Video Poker")
    st.markdown("Get a pair of Jacks or better to win!")

    # State başlatma
    if "vp_state" not in st.session_state:
        st.session_state.vp_state = "betting"
        st.session_state.vp_deck = []
        st.session_state.vp_hand = []
        st.session_state.vp_message = ""
        st.session_state.vp_history = []
        st.session_state.vp_current_bet = 1

    # Payout Tablosu
    vp_payouts = {
        "Royal Flush": 800, "Straight Flush": 50, "Four of a Kind": 25,
        "Full House": 9, "Flush": 6, "Straight": 4,
        "Three of a Kind": 3, "Two Pair": 2, "Jacks or Better": 1,
    }
    st.dataframe(vp_payouts.items(), column_config={"0": "Hand", "1": "Payout (for 1 credit bet)"})

    # Poker Eli Kontrol Fonksiyonları
    def check_vp_hand(hand):
        ranks = sorted([card['rank'] for card in hand], key=lambda r: vp_rank_map.get(r, 0))
        suits = [card['suit'] for card in hand]
        rank_counts = Counter(ranks)
        is_flush = len(set(suits)) == 1
        numerical_ranks = sorted(list(set(vp_rank_map.get(r, 0) for r in ranks)))
        is_straight = len(numerical_ranks) == 5 and (numerical_ranks[-1] - numerical_ranks[0] == 4)
        if numerical_ranks == [2, 3, 4, 5, 14]: is_straight = True

        if is_straight and is_flush and numerical_ranks[-1] == 14 and numerical_ranks[0] == 10: return "Royal Flush"
        if is_straight and is_flush: return "Straight Flush"
        if 4 in rank_counts.values(): return "Four of a Kind"
        if sorted(rank_counts.values()) == [2, 3]: return "Full House"
        if is_flush: return "Flush"
        if is_straight: return "Straight"
        if 3 in rank_counts.values(): return "Three of a Kind"
        if list(rank_counts.values()).count(2) == 2: return "Two Pair"
        for rank, count in rank_counts.items():
            if count == 2 and rank in ['J', 'Q', 'K', 'A']:
                return "Jacks or Better"
        return "Nothing"

    vp_rank_map = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13, 'A':14}

    def reset_vpoker_state(reset_balance=False):
        st.session_state.vp_state = "betting"
        st.session_state.vp_deck = []
        st.session_state.vp_hand = []
        st.session_state.vp_message = ""
        st.session_state.vp_history = []
        # st.session_state.vp_current_bet = 1 # Keep last bet
        if reset_balance: st.session_state.player_balance = 1000
    globals()["vp_reset_func"] = reset_vpoker_state

    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset features from the sidebar.")
    else:
        # Bahis Aşaması
        if st.session_state.vp_state == "betting":
            vp_bet = st.number_input("Bet Amount (1-5 credits):", min_value=1, max_value=min(5, st.session_state.player_balance),
                                     value=st.session_state.vp_current_bet, step=1, key="vp_bet")

            if st.button("Deal Hand", key="vp_deal"):
                st.session_state.player_balance -= vp_bet
                st.session_state.vp_current_bet = vp_bet
                st.session_state.vp_deck = create_deck()[:52]
                st.session_state.vp_hand = [{'card': st.session_state.vp_deck.pop(), 'held': False} for _ in range(5)]
                st.session_state.vp_state = "dealt"
                st.session_state.vp_message = "Select cards to hold and click Draw."
                st.rerun()

        # Kart Seçme ve Çekme Aşaması
        elif st.session_state.vp_state in ["dealt", "holding"]:
            st.subheader("Your Hand:")
            cols = st.columns(5)
            for i, card_info in enumerate(st.session_state.vp_hand):
                card = card_info['card']
                is_held = st.session_state.vp_hand[i]['held']
                button_type = "primary" if is_held else "secondary"
                color = "red" if card['suit'] in ['♥', '♦'] else "black"
                card_html = f"<div style='border:1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px; text-align: center; background-color: white; color: {color};'> <span style='font-size: 1.5em; font-weight: bold;'>{card['rank']}</span><br><span style='font-size: 1.5em;'>{card['suit']}</span> </div>"
                cols[i].markdown(card_html, unsafe_allow_html=True)
                if cols[i].button("Hold" if not is_held else "Discard", key=f"vp_card_{i}", type=button_type, use_container_width=True):
                    st.session_state.vp_hand[i]['held'] = not is_held
                    st.session_state.vp_state = "holding"
                    st.rerun()

            st.caption(st.session_state.vp_message)

            if st.button("Draw", key="vp_draw", disabled=(st.session_state.vp_state != "holding")):
                for i in range(5):
                    if not st.session_state.vp_hand[i]['held']:
                        if st.session_state.vp_deck:
                             st.session_state.vp_hand[i]['card'] = st.session_state.vp_deck.pop()
                             st.session_state.vp_hand[i]['held'] = False
                        else:
                            st.error("Deck is empty!")

                final_hand_cards = [info['card'] for info in st.session_state.vp_hand]
                hand_rank = check_vp_hand(final_hand_cards)
                payout_multiplier = vp_payouts.get(hand_rank, 0)

                if payout_multiplier > 0:
                    winnings = st.session_state.vp_current_bet * payout_multiplier
                    st.session_state.player_balance += winnings + st.session_state.vp_current_bet
                    st.session_state.vp_message = f"🎉 {hand_rank}! You win {winnings}!"
                    add_history("vp", st.session_state.vp_current_bet, winnings, st.session_state.player_balance)
                    st.balloons()
                else:
                    st.session_state.vp_message = f"😕 {hand_rank}. No win."
                    add_history("vp", st.session_state.vp_current_bet, -st.session_state.vp_current_bet, st.session_state.player_balance)

                st.session_state.vp_state = "result"
                st.rerun()

        # Sonuç Gösterme Aşaması
        elif st.session_state.vp_state == "result":
            st.subheader("Final Hand:")
            cols = st.columns(5)
            for i, card_info in enumerate(st.session_state.vp_hand):
                card = card_info['card']
                color = "red" if card['suit'] in ['♥', '♦'] else "black"
                card_html = f"<div style='border:1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px; text-align: center; background-color: white; color: {color};'> <span style='font-size: 1.5em; font-weight: bold;'>{card['rank']}</span><br><span style='font-size: 1.5em;'>{card['suit']}</span> </div>"
                cols[i].markdown(card_html, unsafe_allow_html=True)

            st.header(st.session_state.vp_message)
            if st.button("Deal New Hand", key="vp_new_deal"):
                reset_vpoker_state()
                st.rerun()

    display_history("vp")


# --- SEKME 7: MUSIC PLAYER ---
with tab_music:
    st.header("🎶 Music Player")
    st.markdown("How about a nice blues session? Maybe it will relax you.")

    youtube_url = "https://www.youtube.com/watch?v=1eNSWZ4x2ZU&list=PLoPLEt1InO1x_fhNUCZW2HgRI3uTUn5NY"

    st.video(youtube_url)
    st.caption("Music provided via YouTube embed.")

# --- SEKME 8: SETTINGS ---
with tab_settings:
    st.header("⚙️ Settings")

    st.subheader("Chatbot Settings")
    simlish_on = st.toggle("Enable Simlish Mode 👽", value=st.session_state.simlish_mode, key="settings_simlish")
    if simlish_on != st.session_state.simlish_mode:
        st.session_state.simlish_mode = simlish_on
        st.rerun()

    st.subheader("Blackjack Settings")
    st.session_state.bj_deck_count = st.selectbox(
        "Number of Decks",
        [4, 6, 8],
        index=[4, 6, 8].index(st.session_state.get("bj_deck_count", 6))
    )
    # st.caption("Note: The game logic now uses this setting.") # Eğer create_deck fonksiyonunu güncellediyseniz bu notu kullanabilirsiniz.

    st.subheader("General Settings")
    st.markdown("Use the 'Reset Interactive Features' button in the sidebar to reset game states and the shared balance.")

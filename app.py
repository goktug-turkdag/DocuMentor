import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import os
import tempfile
import random # Casino oyunları için
import time # Cashout mesajı için

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
            # ... (Dataset yükleme kodunun geri kalanı)...
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
    # ... (Dosya işleme kodunun geri kalanı)...
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
Sul sul! You are a Sim from The Sims game... 
(Simlish prompt'unuzun geri kalanı) ...
Context: {context}
Chat History: {chat_history}
Question: {question}
Simlish Answer:
"""

# --- 2. WEB ARAYÜZÜ (Sekmeli Yapı) ---
st.title("DocuMentor & Mini Casino 📄🃏🪙🎡") # Başlığı güncelledim

# --- YENİ: Sekmeler (Chat, Blackjack, Coin Flip, Roulette) ---
tab_chat, tab_blackjack, tab_coinflip, tab_roulette = st.tabs(["💬 Chatbot", "🃏 Blackjack", "🪙 Coin Flip", "🎡 Roulette"])

# --- Sidebar ---
with st.sidebar:
    st.header("About DocuMentor")
    st.markdown(
        "An intelligent Q&A chatbot and mini-casino app built with RAG, "
        "Google Gemini, Streamlit, and Python by Göktuğ Türkdağ."
    )
    st.markdown("---")
    st.subheader("Developed by Göktuğ Türkdağ")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/goktugturkdag)")
    st.markdown("🐙 [GitHub](https://github.com/goktug-turkdag)")
    st.markdown("---")
    
    # Düzeltme: Clear Chat History sadece sohbeti temizler
    if st.button("Clear Chat History 🧹"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.file_retriever = None
        st.session_state.processed_files = []
        st.success("Chat history and uploaded files cleared!")
        st.rerun()

    # YENİ: Oyunları Sıfırlama Butonu
    if st.button("Reset Casino Games 💰"):
        # Blackjack state
        st.session_state.game_state = "betting" 
        st.session_state.player_balance = 1000 
        # Coin Flip state (varsa)
        st.session_state.coin_flip_result = ""
        st.session_state.coin_flip_message = ""
        # Roulette state (varsa)
        st.session_state.roulette_bets = {}
        st.session_state.roulette_result = ""
        st.session_state.roulette_message = ""
        st.success("Casino balance and game states reset!")
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
# Sohbet için
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.file_retriever = None
    st.session_state.processed_files = []
    
    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    I'm trained on **Dolly 15k**, but you can also **upload your own documents** in the sidebar to chat with them!
    Check out the other tabs for some mini-casino games!
    
    - Try asking me: "Who is Göktuğ Türkdağ?"
    - Or toggle **Simlish Mode** 👽
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Blackjack için
if "game_state" not in st.session_state:
    st.session_state.game_state = "betting" 
    st.session_state.deck = []
    st.session_state.player_hand = []
    st.session_state.dealer_hand = []
    st.session_state.game_message = ""
    st.session_state.side_bet_message = ""
    st.session_state.player_balance = 1000 
    st.session_state.current_bet = 0
    st.session_state.bet_21_3 = 0
    st.session_state.bet_perfect_pairs = 0
    st.session_state.bet_lucky_seven = 0
    st.session_state.bet_bust = 0

# Coin Flip için
if "coin_flip_result" not in st.session_state:
    st.session_state.coin_flip_result = ""
    st.session_state.coin_flip_message = ""

# Roulette için
if "roulette_bets" not in st.session_state:
    st.session_state.roulette_bets = {}
    st.session_state.roulette_result = ""
    st.session_state.roulette_message = ""

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

    # Prompt'u seç
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

        # Easter Egg kontrolü
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
            # Normal RAG cevabı
            spinner_text = "Searching for the answer... (Chumcha!)" if st.session_state.simlish_mode else "Searching for the answer..."
            
            with st.chat_message("assistant", avatar=avatars["assistant"]):
                response_container = st.empty()

                # Stream generator fonksiyonu
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


# --- SEKME 2: BLACKJACK (TÜM YAN BAHİSLER + 5 CARD CHARLIE) ---
with tab_blackjack:
    st.header("🃏 Blackjack")
    st.markdown("Place your bet, and optional side bets, to beat the dealer!")

    with st.expander("Show/Hide Basic Strategy & Side Bet Info"):
        st.markdown("""
        **Basic Blackjack Strategy:** ... (Strateji bilgisi buraya gelecek) ...
        
        **Side Bet Payouts:**
        - **Perfect Pairs:** Mixed (6:1), Colored (12:1), Perfect (25:1)
        - **21+3:** Flush (5:1), Straight (10:1), Trips (30:1), Str Flush (40:1)
        - **Lucky 7s:** One 7 (3:1), Two 7s (50:1), Three 7s (100:1)
        - **Bust It!:** Dealer busts with 3 cards (2:1), 4 cards (3:1), 5+ cards (5:1)
        
        **Special Wins:**
        - **5-Card Charlie:** You draw 5 cards without busting - you automatically win 1:1 on your main bet (unless dealer has Blackjack).
        """)

    # --- Blackjack Oyun Fonksiyonları ---
    
    def reset_blackjack_state():
        st.session_state.game_state = "betting" 
        st.session_state.deck = []
        st.session_state.player_hand = []
        st.session_state.dealer_hand = []
        st.session_state.game_message = ""
        st.session_state.side_bet_message = ""
        # Bakiye sıfırlanmaz st.session_state.player_balance = 1000 
        st.session_state.current_bet = 0
        st.session_state.bet_21_3 = 0
        st.session_state.bet_perfect_pairs = 0
        st.session_state.bet_lucky_seven = 0
        st.session_state.bet_bust = 0

    def create_deck():
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'rank': rank, 'suit': suit} for suit in suits for rank in ranks] * 6
        random.shuffle(deck)
        return deck

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

    def display_hand(hand, title):
        st.subheader(title)
        cols = st.columns(len(hand) if hand else 1)
        for i, card in enumerate(hand):
            with cols[i].container(border=True):
                st.markdown(f"<h3 style='text-align: center; margin: 0;'>{card['rank']}{card['suit']}</h3>", unsafe_allow_html=True)
        score = calculate_score(hand)
        st.markdown(f"**Score: {score}**")
        return score

    def display_dealer_hand_hidden(hand):
        st.subheader("Dealer's Hand")
        cols = st.columns(2)
        with cols[0].container(border=True):
            st.markdown(f"<h3 style='text-align: center; margin: 0;'>{hand[0]['rank']}{hand[0]['suit']}</h3>", unsafe_allow_html=True)
        with cols[1].container(border=True):
            st.markdown(f"<h3 style='text-align: center; margin: 0;'>❔</h3>", unsafe_allow_html=True)
        st.markdown("**Score: ?**")

    # --- Yan Bahis Kontrol Fonksiyonları ---
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

    # YENİ: Gelişmiş Bust It! Kontrolü
    def check_bust_it(dealer_hand, dealer_score, bet):
        if bet == 0: return 0, ""
        if dealer_score > 21:
            num_cards = len(dealer_hand)
            if num_cards >= 5: payout = 5
            elif num_cards == 4: payout = 3
            elif num_cards == 3: payout = 2
            else: payout = 0 # 2 kartla bust olmaz
            
            if payout > 0:
                winnings = bet * payout
                return winnings + bet, f"**Bust It! Bet Win: +{winnings}!** (Dealer busted with {num_cards} cards, {payout}:1)"
        
        return 0, "Bust It! bet lost."

    # --- Dinamik Cashout Teklifi Hesaplayıcı ---
    def get_cashout_offer_heuristic(player_hand, dealer_up_card, bet):
        # ... (Cashout hesaplama kodunuz burada - değişiklik yok) ...
        p_score = calculate_score(player_hand)
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

    # --- Oyun Durumu Yönetimi ---
    if "game_state" not in st.session_state:
        reset_blackjack_state() 

    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")
    
    # --- Bahis Arayüzü ---
    if st.session_state.game_state == "betting":
        st.session_state.side_bet_message = "" 
        
        if st.session_state.player_balance <= 0:
            st.error("You are out of money! Game over.")
            if st.button("Start Over with 1000?"):
                reset_blackjack_state() # Reset only blackjack state, keep balance logic
                st.session_state.player_balance = 1000 # Reset balance explicitly
                st.rerun()
        else:
            with st.form(key="bet_form"):
                st.subheader("Place Your Bets")
                bet_amount = st.number_input(
                    "Main Bet:", min_value=10, max_value=st.session_state.player_balance, value=50, step=10
                )
                
                st.markdown("---")
                st.markdown("**Side Bets (Optional)**")
                
                # Max bahisleri kalan bakiye ile sınırla
                max_side_bet = min(100, st.session_state.player_balance)
                bet_21_3_amount = st.number_input("21+3 Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_pp_amount = st.number_input("Perfect Pairs Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_lucky_seven_amount = st.number_input("Lucky 7s Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                bet_bust_amount = st.number_input("Bust It! Bet:", min_value=0, max_value=max_side_bet, value=0, step=5)
                
                deal_button = st.form_submit_button("Deal")

            if st.button("Cash Out & Reset Game"):
                st.success(f"You cashed out with {st.session_state.player_balance}! Game is resetting.")
                time.sleep(1.5)
                reset_blackjack_state() # Reset only blackjack state, keep balance logic
                st.session_state.player_balance = 1000 # Reset balance explicitly
                st.rerun()

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
                    # PP ve 21+3'ü hemen kontrol et
                    if st.session_state.bet_perfect_pairs > 0:
                        pp_payout = check_perfect_pairs(st.session_state.player_hand)
                        if pp_payout > 0:
                            winnings = st.session_state.bet_perfect_pairs * pp_payout
                            st.session_state.player_balance += winnings + st.session_state.bet_perfect_pairs
                            side_messages.append(f"**Perfect Pairs Win: +{winnings}!** ({pp_payout}:1)")
                    
                    if st.session_state.bet_21_3 > 0:
                        p3_payout = check_21_plus_3(st.session_state.player_hand, st.session_state.dealer_hand[0])
                        if p3_payout > 0:
                            winnings = st.session_state.bet_21_3 * p3_payout
                            st.session_state.player_balance += winnings + st.session_state.bet_21_3
                            side_messages.append(f"**21+3 Win: +{winnings}!** ({p3_payout}:1)")
                    
                    if not side_messages and (st.session_state.bet_21_3 > 0 or st.session_state.bet_perfect_pairs > 0):
                        st.session_state.side_bet_message = "21+3 / Perfect Pairs bets lost."
                    else:
                        st.session_state.side_bet_message = " \n".join(side_messages)
                    
                    # Ana Oyun Blackjack Kontrolü
                    player_score = calculate_score(st.session_state.player_hand)
                    dealer_score = calculate_score(st.session_state.dealer_hand) # Dealer BJ kontrolü için
                    
                    if player_score == 21 and dealer_score != 21: # Oyuncu BJ, Krupiye değil
                        st.session_state.game_state = "game_over"
                        st.session_state.game_message = "Blackjack! 🎉 You win!"
                        st.session_state.player_balance += int(st.session_state.current_bet * 2.5) 
                        
                        # Lucky 7s kontrolü
                        l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                        if l7_winnings > 0:
                            st.session_state.player_balance += l7_winnings
                            st.session_state.side_bet_message += "\n" + l7_msg
                            
                    elif player_score == 21 and dealer_score == 21: # İkisi de BJ
                        st.session_state.game_state = "game_over"
                        st.session_state.game_message = "Push! Both have Blackjack. 😐"
                        st.session_state.player_balance += st.session_state.current_bet # Ana bahsi iade
                        # Yan bahisler zaten ödendi/kaybedildi
                    elif dealer_score == 21: # Sadece krupiye BJ
                         st.session_state.game_state = "game_over"
                         st.session_state.game_message = "Dealer has Blackjack. 😕 You lose."
                         # Ana bahis kaybedildi
                         # Yan bahisler zaten ödendi/kaybedildi
                    else:
                        st.session_state.game_state = "player_turn"
                        st.session_state.game_message = "Your turn! Hit or Stand?"
                    st.rerun()

    # --- Oyun Akışı ---
    if st.session_state.game_state in ["player_turn", "dealer_turn", "game_over"]:
        
        if st.session_state.side_bet_message:
            st.info(st.session_state.side_bet_message)
        
        if st.session_state.game_state == "player_turn":
            display_dealer_hand_hidden(st.session_state.dealer_hand)
        else:
            display_hand(st.session_state.dealer_hand, "Dealer's Hand")
            
        st.markdown("---")
        
        player_score = display_hand(st.session_state.player_hand, "Your Hand")

        if st.session_state.game_state == "game_over":
            st.header(st.session_state.game_message)
            if st.button("Play Again?", key="play_again"):
                # Ana oyun durumunu sıfırla, bakiyeyi koru
                reset_blackjack_state() # Reset only non-balance game state
                st.rerun()
        
        if st.session_state.game_state == "player_turn":
            
            cashout_offer = get_cashout_offer_heuristic(
                st.session_state.player_hand, 
                st.session_state.dealer_hand[0], 
                st.session_state.current_bet
            )
            
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Hit (Kart Çek)", key="hit"):
                st.session_state.player_hand.append(st.session_state.deck.pop())
                player_score = calculate_score(st.session_state.player_hand)
                num_player_cards = len(st.session_state.player_hand)

                # YENİ: 5-Card Charlie kontrolü
                if num_player_cards == 5 and player_score <= 21:
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = "5-Card Charlie! 🎉 You win!"
                    st.session_state.player_balance += st.session_state.current_bet * 2 # Ana bahsi 1:1 öde
                elif player_score > 21:
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = "Bust! 💥 You lose."
                    
                    # Oyuncu patladığında Lucky 7s'i kontrol et
                    l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                    if l7_winnings > 0:
                        st.session_state.player_balance += l7_winnings
                        st.session_state.side_bet_message += "\n" + l7_msg
                        
                st.rerun()

            if col2.button("Stand (Dur)", key="stand"):
                st.session_state.game_state = "dealer_turn"
                
                # Oyuncu durduğunda Lucky 7s'i kontrol et
                l7_winnings, l7_msg = check_lucky_sevens(st.session_state.player_hand, st.session_state.bet_lucky_seven)
                if l7_winnings > 0:
                    st.session_state.player_balance += l7_winnings
                    st.session_state.side_bet_message += "\n" + l7_msg
                    
                st.rerun()
            
            if col3.button(f"Cash Out for 💰 {cashout_offer}", key="cashout"):
                st.session_state.player_balance += cashout_offer
                st.session_state.game_state = "game_over"
                st.session_state.game_message = f"You cashed out for {cashout_offer}! (Original bet was {st.session_state.current_bet})"
                st.rerun()
                
    if st.session_state.game_state == "dealer_turn":
        dealer_score = calculate_score(st.session_state.dealer_hand)
        
        while dealer_score < 17:
            st.session_state.dealer_hand.append(st.session_state.deck.pop())
            dealer_score = calculate_score(st.session_state.dealer_hand)
        
        player_score = calculate_score(st.session_state.player_hand)
        bet = st.session_state.current_bet
        
        # Bust It! bahsini kontrol et
        bust_winnings, bust_msg = check_bust_it(st.session_state.dealer_hand, dealer_score, st.session_state.bet_bust)
        if bust_winnings > 0:
            st.session_state.player_balance += bust_winnings
            st.session_state.side_bet_message += "\n" + bust_msg
        elif st.session_state.bet_bust > 0: # Eğer bahis yapıldı ama kazanmadıysa
             st.session_state.side_bet_message += "\nBust It! bet lost."
        
        # Ana bahsi öde ve mesajı oluştur
        if dealer_score > 21:
            st.session_state.game_message = "Dealer busts! 🎉 You win!"
            st.session_state.player_balance += bet * 2 
        elif dealer_score > player_score:
            st.session_state.game_message = "Dealer wins. 😕"
        elif player_score > dealer_score:
            st.session_state.game_message = "🎉 You win!"
            st.session_state.player_balance += bet * 2
        else:
            st.session_state.game_message = "It's a tie! (Push) 😐"
            st.session_state.player_balance += bet 
            
        st.session_state.game_state = "game_over"
        st.rerun()

# --- YENİ ÖZELLİK: SEKME 3 - COIN FLIP ---
with tab_coinflip:
    st.header("🪙 Coin Flip")
    st.markdown("A simple Heads or Tails betting game.")

    # Coin Flip State
    if "coin_flip_result" not in st.session_state:
        st.session_state.coin_flip_result = ""
        st.session_state.coin_flip_message = ""

    # Bakiye göster
    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset games from the sidebar.")
    else:
        with st.form("coin_flip_form"):
            cf_bet = st.number_input("Bet Amount:", min_value=1, max_value=st.session_state.player_balance, value=10, step=1)
            cf_choice = st.radio("Choose:", ("Heads", "Tails"), horizontal=True)
            flip_button = st.form_submit_button("Flip Coin")

        if flip_button:
            st.session_state.player_balance -= cf_bet
            result = random.choice(["Heads", "Tails"])
            st.session_state.coin_flip_result = result
            
            if result == cf_choice:
                winnings = cf_bet * 2
                st.session_state.player_balance += winnings
                st.session_state.coin_flip_message = f"🎉 It's {result}! You win {winnings}!"
                st.balloons()
            else:
                st.session_state.coin_flip_message = f"😕 It's {result}. You lost."
            
            st.rerun()

    # Sonucu göster
    if st.session_state.coin_flip_result:
        st.subheader(f"Result: {st.session_state.coin_flip_result}")
        st.write(st.session_state.coin_flip_message)

# --- YENİ ÖZELLİK: SEKME 4 - ROULETTE ---
with tab_roulette:
    st.header("🎡 Roulette (European)")
    st.markdown("Place your bets on the table and spin the wheel!")

    # Roulette State
    if "roulette_bets" not in st.session_state:
        st.session_state.roulette_bets = {} # {'bet_type_number': amount}
        st.session_state.roulette_result = ""
        st.session_state.roulette_message = ""

    # Rulet Çarkı Tanımları
    numbers = list(range(37)) # 0 to 36
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
        return "Low (1-18)" if 1 <= number <= 18 else "High (19-36)"

    # Bakiye göster
    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")

    if st.session_state.player_balance <= 0:
        st.error("You are out of money! Reset games from the sidebar.")
    else:
        st.subheader("Place Your Bets:")
        
        # Bahisleri geçici bir dict'te topla
        current_bets = {} 
        total_current_bet = 0

        # Sayı Bahsi
        num_cols = st.columns(4)
        selected_number = num_cols[0].selectbox("Number (0-36):", ["-"] + numbers)
        if selected_number != "-":
            number_bet = num_cols[1].number_input("Bet on Number (35:1):", min_value=0, value=0, step=1, key="num_bet")
            if number_bet > 0:
                current_bets[f"number_{selected_number}"] = number_bet
                total_current_bet += number_bet

        # Dış Bahisler (Renk, Tek/Çift, Küçük/Büyük)
        st.markdown("**Outside Bets (1:1 Payout)**")
        ext_cols = st.columns(3)
        
        red_bet = ext_cols[0].number_input("Bet on Red:", min_value=0, value=0, step=1, key="red_bet")
        if red_bet > 0: current_bets["Red"] = red_bet; total_current_bet += red_bet
        
        black_bet = ext_cols[0].number_input("Bet on Black:", min_value=0, value=0, step=1, key="black_bet")
        if black_bet > 0: current_bets["Black"] = black_bet; total_current_bet += black_bet
        
        odd_bet = ext_cols[1].number_input("Bet on Odd:", min_value=0, value=0, step=1, key="odd_bet")
        if odd_bet > 0: current_bets["Odd"] = odd_bet; total_current_bet += odd_bet
        
        even_bet = ext_cols[1].number_input("Bet on Even:", min_value=0, value=0, step=1, key="even_bet")
        if even_bet > 0: current_bets["Even"] = even_bet; total_current_bet += even_bet
        
        low_bet = ext_cols[2].number_input("Bet on Low (1-18):", min_value=0, value=0, step=1, key="low_bet")
        if low_bet > 0: current_bets["Low"] = low_bet; total_current_bet += low_bet
        
        high_bet = ext_cols[2].number_input("Bet on High (19-36):", min_value=0, value=0, step=1, key="high_bet")
        if high_bet > 0: current_bets["High"] = high_bet; total_current_bet += high_bet

        st.markdown(f"**Total Bet: {total_current_bet}**")

        if st.button("Spin Wheel", key="spin_roulette"):
            if total_current_bet <= 0:
                st.warning("Please place at least one bet.")
            elif total_current_bet > st.session_state.player_balance:
                st.error(f"Total bet ({total_current_bet}) cannot exceed your balance ({st.session_state.player_balance}).")
            else:
                # Bahsi düş
                st.session_state.player_balance -= total_current_bet
                st.session_state.roulette_bets = current_bets # Bahisleri kaydet
                
                # Çarkı çevir
                winning_number = random.randint(0, 36)
                winning_color = get_color(winning_number)
                winning_odd_even = get_odd_even(winning_number)
                winning_low_high = get_low_high(winning_number)
                
                st.session_state.roulette_result = f"**{winning_number} {winning_color}**"
                
                # Kazançları hesapla
                total_winnings = 0
                winning_messages = []

                for bet_type, bet_amount in st.session_state.roulette_bets.items():
                    win = False
                    payout_ratio = 0
                    
                    if bet_type.startswith("number_"):
                        bet_num = int(bet_type.split("_")[1])
                        if bet_num == winning_number:
                            win = True
                            payout_ratio = 35
                    elif bet_type == "Red" and winning_color == "Red": win = True; payout_ratio = 1
                    elif bet_type == "Black" and winning_color == "Black": win = True; payout_ratio = 1
                    elif bet_type == "Odd" and winning_odd_even == "Odd": win = True; payout_ratio = 1
                    elif bet_type == "Even" and winning_odd_even == "Even": win = True; payout_ratio = 1
                    elif bet_type == "Low" and winning_low_high == "Low (1-18)": win = True; payout_ratio = 1
                    elif bet_type == "High" and winning_low_high == "High (19-36)": win = True; payout_ratio = 1

                    if win:
                        winnings = bet_amount * payout_ratio
                        total_winnings += winnings + bet_amount # Bahsi geri iade et
                        winning_messages.append(f"Win on {bet_type.replace('_', ' ')}: +{winnings}!")
                
                if total_winnings > 0:
                    st.session_state.player_balance += total_winnings
                    st.session_state.roulette_message = "🎉 **Winning Bets:**\n" + "\n".join(winning_messages)
                    st.balloons()
                else:
                    st.session_state.roulette_message = "😕 No winning bets this round."
                    
                st.rerun() # Sonucu göster

    # Rulet sonucunu göster
    if st.session_state.roulette_result:
        st.subheader(f"Wheel Result: {st.session_state.roulette_result}")
        st.write(st.session_state.roulette_message)
        st.button("Place New Bets", on_click=lambda: st.session_state.update({"roulette_result":"", "roulette_message":"", "roulette_bets":{}}))

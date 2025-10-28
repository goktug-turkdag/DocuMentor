import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import os
import tempfile
import random # Blackjack için
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
st.title("DocuMentor 📄")

tab_chat, tab_blackjack = st.tabs(["💬 Chatbot", "🃏 Blackjack"])

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
        # Sadece sohbetle ilgili session state'leri temizle
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.file_retriever = None
        st.session_state.processed_files = []
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
# Sohbet için
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.file_retriever = None
    st.session_state.processed_files = []
    
    welcome_message = f"""
    Hi! I'm **DocuMentor**, an intelligent RAG chatbot developed by **Göktuğ Türkdağ**.
    I'm trained on **Dolly 15k**, but you can also **upload your own documents** in the sidebar to chat with them!
    
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

# --- SEKME 1: CHATBOT (TAM KOD) ---
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


# --- SEKME 2: BLACKJACK (DİNAMİK CASHOUT İLE) ---
with tab_blackjack:
    st.header("🃏 Blackjack")
    st.markdown("Place your bet, and optional side bets, to beat the dealer!")

    with st.expander("Show/Hide Basic Strategy Guide"):
        st.markdown("""
        **Basic Blackjack Strategy** helps you make the best decision to minimize the house edge.
        - **Hit:** Take another card.
        - **Stand:** Keep your current hand.
        
        **General Rules:**
        - **Always Stand** if your hand is 17 or higher.
        - **Always Hit** if your hand is 11 or lower.
        
        **Your Hand vs. Dealer's Up Card:**
        | Your Hand | Dealer's Up Card 2-6 | Dealer's Up Card 7-A |
        | :--- | :--- | :--- |
        | **12-16** (Stiff) | Stand (Dealer might bust) | Hit (Dealer likely has 17+) |
        | **A, 2-6** (Soft 13-17) | Hit | Hit |
        | **A, 7** (Soft 18) | Stand | Hit (Stand vs 2,7,8) |
        """)

    # --- Blackjack Oyun Fonksiyonları ---
    
    def reset_game_state():
        """Oyun durumunu ve bakiyeyi sıfırlar."""
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

    def create_deck():
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'rank': rank, 'suit': suit} for suit in suits for rank in ranks]
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
            cols[i].container(border=True).markdown(f"## {card['rank']}{card['suit']}")
        score = calculate_score(hand)
        st.markdown(f"**Score: {score}**")
        return score

    def display_dealer_hand_hidden(hand):
        st.subheader("Dealer's Hand")
        cols = st.columns(2)
        cols[0].container(border=True).markdown(f"## {hand[0]['rank']}{hand[0]['suit']}")
        cols[1].container(border=True).markdown("## ❔")
        st.markdown("**Score: ?**")

    # --- Yan Bahis Kontrol Fonksiyonları ---
    def check_perfect_pairs(hand):
        card1, card2 = hand[0], hand[1]
        if card1['rank'] == card2['rank']:
            if card1['suit'] == card2['suit']: return 25  # Perfect Pair (25:1)
            card1_color_is_red = card1['suit'] in ['♥', '♦']
            card2_color_is_red = card2['suit'] in ['♥', '♦']
            if card1_color_is_red == card2_color_is_red: return 12  # Colored Pair (12:1)
            return 6   # Mixed Pair (6:1)
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
        if is_straight and is_flush: return 40 # Straight Flush (40:1)
        if is_trips: return 30 # Three of a Kind (30:1)
        if is_straight: return 10 # Straight (10:1)
        if is_flush: return 5 # Flush (5:1)
        return 0 

    # --- YENİ ÖZELLİK: Dinamik Cashout Teklifi Hesaplayıcı ---
    def get_cashout_offer_heuristic(player_hand, dealer_up_card, bet):
        """
        Oyuncunun kazanma olasılığına dayalı yaklaşık (heuristic) bir cashout teklifi hesaplar.
        """
        p_score = calculate_score(player_hand)
        d_val = get_card_value(dealer_up_card['rank'])
        
        # 1. Kazanma olasılığını (p_win) 0.0 ile 1.0 arasında tahmin et
        p_win = 0.48 # Temel ev avantajı

        # Oyuncunun güçlü elleri
        if p_score == 20: p_win = 0.85
        elif p_score == 19: p_win = 0.75
        elif p_score == 18: p_win = 0.65
        # Oyuncunun "patlama" riski olan (stiff) elleri (12-16)
        elif p_score in [12, 13, 14, 15, 16]:
            if d_val in [2, 3, 4, 5, 6]: p_win = 0.40 # Krupiye patlayabilir
            elif d_val in [7, 8, 9]: p_win = 0.25 # Krupiye güçlü
            elif d_val in [10, 11]: p_win = 0.20 # Krupiye çok güçlü
        # Oyuncunun "Hit" elleri (9-11)
        elif p_score == 11:
            if d_val in [10, 11]: p_win = 0.45
            else: p_win = 0.60
        elif p_score == 10:
            if d_val in [10, 11]: p_win = 0.40
            else: p_win = 0.55
        elif p_score == 9:
            if d_val in [10, 11]: p_win = 0.30 # Kullanıcının örneğine yakın (dezavantaj)
            else: p_win = 0.40
        # Diğer (yumuşak As'lı) eller
        elif p_score == 17 and (get_card_value('A') in [get_card_value(c['rank']) for c in player_hand]):
             p_win = 0.45 # Yumuşak 17

        # 2. Teklifi hesapla
        # p_win > 0.5 ise (Avantaj): Bahsin (bet) üzerine, avantajın (p_win - 0.5) 1.5 katını ekle
        # p_win < 0.5 ise (Dezavantaj): Bahisten, dezavantajın (0.5 - p_win) 3 katını çıkar
        # (Bu çarpanlar, kullanıcının 9v10 -> 40 teklif örneğine uyması için ayarlanmıştır)
        
        if p_win >= 0.5:
            multiplier = 1.0 + (p_win - 0.5) * 1.5
        else:
            multiplier = 1.0 - (0.5 - p_win) * 3.0
            
        offer = int(bet * multiplier)
        
        # Teklifin negatif veya bahisten çok yüksek olmamasını sağla
        if offer < 0: offer = 0
        if offer > (bet * 1.9): offer = int(bet * 1.9) # Max 1.9x teklif
        
        return offer

    # --- Oyun Durumu Yönetimi ---
    st.metric(label="Your Balance", value=f"💰 {st.session_state.player_balance}")
    
    if st.session_state.game_state == "betting":
        st.session_state.side_bet_message = "" 
        
        if st.session_state.player_balance <= 0:
            st.error("You are out of money! Game over.")
            if st.button("Start Over with 1000?"):
                reset_game_state()
                st.rerun()
        else:
            with st.form(key="bet_form"):
                st.subheader("Place Your Bets")
                bet_amount = st.number_input(
                    "Main Bet:", min_value=10, max_value=st.session_state.player_balance, value=50, step=10
                )
                
                st.markdown("---")
                st.markdown("**Side Bets (Optional)**")
                bet_21_3_amount = st.number_input(
                    "21+3 Bet (5:1 to 40:1):", min_value=0, max_value=min(100, st.session_state.player_balance - bet_amount), value=0, step=5
                )
                bet_pp_amount = st.number_input(
                    "Perfect Pairs Bet (6:1 to 25:1):", min_value=0, max_value=min(100, st.session_state.player_balance - bet_amount - bet_21_3_amount), value=0, step=5
                )
                
                deal_button = st.form_submit_button("Deal")

            if st.button("Cash Out & Reset Game"):
                st.success(f"You cashed out with {st.session_state.player_balance}! Game is resetting.")
                time.sleep(1.5)
                reset_game_state()
                st.rerun()

            if deal_button:
                total_bet = bet_amount + bet_21_3_amount + bet_pp_amount
                if total_bet > st.session_state.player_balance:
                    st.error("Total bet cannot exceed your balance.")
                else:
                    st.session_state.player_balance -= total_bet
                    st.session_state.current_bet = bet_amount
                    st.session_state.bet_21_3 = bet_21_3_amount
                    st.session_state.bet_perfect_pairs = bet_pp_amount
                    
                    st.session_state.deck = create_deck()
                    st.session_state.player_hand = [st.session_state.deck.pop(), st.session_state.deck.pop()]
                    st.session_state.dealer_hand = [st.session_state.deck.pop(), st.session_state.deck.pop()]
                    
                    side_messages = []
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
                        st.session_state.side_bet_message = "No side bet wins."
                    else:
                        st.session_state.side_bet_message = " \n".join(side_messages)
                    
                    player_score = calculate_score(st.session_state.player_hand)
                    if player_score == 21:
                        st.session_state.game_state = "game_over"
                        st.session_state.game_message = "Blackjack! 🎉 You win!"
                        st.session_state.player_balance += int(st.session_state.current_bet * 2.5) 
                    else:
                        st.session_state.game_state = "player_turn"
                        st.session_state.game_message = "Your turn! Hit or Stand?"
                    st.rerun()

    # --- Oyun Akışı (Dinamik Cashout Eklendi) ---
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
                st.session_state.game_state = "betting"
                st.session_state.player_hand = []
                st.session_state.dealer_hand = []
                st.session_state.game_message = ""
                st.session_state.current_bet = 0
                st.session_state.bet_21_3 = 0
                st.session_state.bet_perfect_pairs = 0
                st.session_state.side_bet_message = ""
                st.rerun()
        
        if st.session_state.game_state == "player_turn":
            
            # --- YENİ: DİNAMİK CASHOUT TEKLİFİ ---
            # Oyuncunun eline ve krupiyenin açık kartına göre teklifi hesapla
            cashout_offer = get_cashout_offer_heuristic(
                st.session_state.player_hand, 
                st.session_state.dealer_hand[0], 
                st.session_state.current_bet
            )
            
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Hit (Kart Çek)", key="hit"):
                st.session_state.player_hand.append(st.session_state.deck.pop())
                player_score = calculate_score(st.session_state.player_hand)
                if player_score > 21:
                    st.session_state.game_state = "game_over"
                    st.session_state.game_message = "Bust! 💥 You lose."
                st.rerun()

            if col2.button("Stand (Dur)", key="stand"):
                st.session_state.game_state = "dealer_turn"
                st.rerun()
            
            # Dinamik Cashout Butonu
            if col3.button(f"Cash Out for 💰 {cashout_offer}", key="cashout"):
                st.session_state.player_balance += cashout_offer
                st.session_state.game_state = "game_over"
                st.session_state.game_message = f"You cashed out for {cashout_offer}! (Original bet was {st.session_state.current_bet})"
                st.rerun()
                
    # Dağıtıcının sırası
    if st.session_state.game_state == "dealer_turn":
        dealer_score = calculate_score(st.session_state.dealer_hand)
        
        while dealer_score < 17:
            st.session_state.dealer_hand.append(st.session_state.deck.pop())
            dealer_score = calculate_score(st.session_state.dealer_hand)
        
        player_score = calculate_score(st.session_state.player_hand)
        bet = st.session_state.current_bet
        
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

# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
import random
import math
import tempfile
import io
from collections import Counter
from dotenv import load_dotenv

# --- LANGCHAIN & AI IMPORTS ---
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document, StrOutputParser
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
except ImportError:
    st.error("Kritik kütüphaneler eksik. Lütfen: pip install langchain-community langchain-google-genai chromadb sentence-transformers langchain-huggingface")
    st.stop()

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="DocuMentor Pro", layout="wide", page_icon="📄")
load_dotenv()

# API Key Kontrolü (Hata vermemesi için pass geçildi, .env dosyanızda olmalı)
if "GOOGLE_API_KEY" not in os.environ:
    # st.warning("GOOGLE_API_KEY bulunamadı. AI özellikleri devre dışı kalabilir.")
    pass 

PERSIST_DIRECTORY = "chroma_db_multilingual"

# --- 2. VERİ SETLERİ VE SABİTLER ---

# Genişletilmiş Yasaklı Kelime Listesi (Güvenlik Bariyeri)
BANNED_KEYWORDS = [
    "kill", "murder", "bomb", "terror", "attack", "assault", "rape", "abuse", "torture", "violence", 
    "drug", "cocaine", "heroin", "meth", "lsd", "theft", "steal", "fraud", "scam", "hack", "phish", 
    "racist", "sexist", "homophobic", "nazi", "supremacy", "discrimination", "slur", "hate speech",
    "porn", "nude", "naked", "erotic", "sex", "incest", "pedophile", "hentai", "xxx",
    "suicide", "self-harm", "die", "death", "overdose", "cut myself", "intihar",
    "fake news", "hoax", "conspiracy", "chemtrail", "flat earth",
    "how to make a bomb", "how to hack", "credit card", "social security", "password"
]

POLITICAL_KEYWORDS = [
    "recep", "tayyip", "erdoğan", "akp", "chp", "mhp", "iyi parti", "hdp", "dem parti",
    "siyaset", "seçim", "hükümet", "bakan", "başkan", "meclis", "tbmm", "oy kullan",
    "trump", "biden", "putin", "zelensky", "russia", "ukraine", "israel", "palestine", "hamas", "gaza",
    "kılıçdaroğlu", "imamoğlu", "yavaş", "özgür özel", "bahçeli", "akşener", "demirtaş", "fetö", "pkk"
]

ACHIEVEMENT_LIST = {
    # General
    "gen_welcome": {"name": "Hoşgeldin Paketi", "desc": "Oyunu ilk kez açtın.", "icon": "👋"},
    "gen_balance_10k": {"name": "Nostradamus", "desc": "Bakiyeni 10,000'in üzerine çıkar.", "icon": "🔮"},
    "gen_balance_100": {"name": "Son Pişmanlık", "desc": "Bakiyen 100'ün altına düştü.", "icon": "🎻"},
    "gen_win_1000": {"name": "I'M NOT LEAVING!", "desc": "Tek bir bahisten 1000+ kazan.", "icon": "🐺"},
    "gen_played_all": {"name": "Mekanın Sahibi", "desc": "Tüm oyunları en az bir kez oyna.", "icon": "👑"},
    # Blackjack
    "bj_win_1": {"name": "Acemi Şansı", "desc": "İlk Blackjack elini kazan.", "icon": "🃏"},
    "bj_win_25": {"name": "Kasa Katili", "desc": "Blackjack'te 25 el kazan.", "icon": "🦈"},
    "bj_blackjack": {"name": "Natural 21", "desc": "İlk iki kartta Blackjack yap.", "icon": "✨"},
    "bj_charlie": {"name": "Beşibiryerde", "desc": "5-Card Charlie yap.", "icon": "✋"},
    "bj_split_win": {"name": "Dublör", "desc": "Split yaptıktan sonra kazan.", "icon": "👯"},
    "bj_side_bet": {"name": "Yan Gelir", "desc": "Bir yan bahis kazan.", "icon": "💸"},
    # Coin Flip
    "cf_win_10": {"name": "Yazı Tura", "desc": "10 kez Yazı Tura kazan.", "icon": "🪙"},
    "cf_played_25": {"name": "Flipping Out", "desc": "25 kez Yazı Tura oyna.", "icon": "🔄"},
    # Roulette
    "rl_win_0": {"name": "Yeşil Yol", "desc": "0'a (Yeşil) bahis koy ve kazan.", "icon": "📿"},
    "rl_win_black": {"name": "Always Bet on Black", "desc": "Siyaha bahis koy ve kazan.", "icon": "⚫"},
    "rl_played_25": {"name": "Dönme Dolap", "desc": "Rulette 25 spin oyna.", "icon": "🎡"},
    # Slots
    "slot_jackpot_7": {"name": "Midas Dokunuşu", "desc": "Jackpot (777) yakala.", "icon": "💎"},
    "slot_win_cherry": {"name": "Kiraz Mevsimi", "desc": "İki 🍒 ile kazan.", "icon": "🍒"},
    "slot_played_100": {"name": "Kolu Çürüttün", "desc": "100 spin at.", "icon": "🦾"},
    # Video Poker
    "vp_win_royal": {"name": "Ezel", "desc": "Royal Flush yakala.", "icon": "👑"},
    "vp_win_aces": {"name": "Kare As", "desc": "Dört As yakala.", "icon": "♠️"},
    "vp_played_25": {"name": "Poker Face", "desc": "25 el oyna.", "icon": "😐"},
    # Sisyphus
    "sc_cashout_20x": {"name": "Fly Me to the Moon", "desc": "20x üzeri cashout.", "icon": "🚀"},
    "sc_cashout_50x": {"name": "Zirve", "desc": "50x üzeri cashout.", "icon": "🏔️"},
    "sc_crash_early": {"name": "Absürd", "desc": "1.05x altı patla.", "icon": "💥"},
    "sc_cashout_1_01x": {"name": "Risk Budur", "desc": "Tam 1.01x'te cashout yap.", "icon": "🐔"},
    # Cognitive
    "iq_level_5": {"name": "Fil Hafızası", "desc": "Hafıza testinde 5. seviyeye ulaş.", "icon": "🐘"},
}

# --- 3. HELPER FUNCTIONS (AI & DATA PROCESSING) ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

@st.cache_resource
def load_creative_llm():
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

@st.cache_data(max_entries=1)
def process_uploaded_files(uploaded_files_data):
    if not uploaded_files_data: return None
    all_documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name, file_content in uploaded_files_data.items():
            temp_path = os.path.join(temp_dir, file_name)
            with open(temp_path, "wb") as f: f.write(file_content)
            try:
                if file_name.endswith(".pdf"): loader = PyPDFLoader(temp_path)
                elif file_name.endswith(".docx"): loader = Docx2txtLoader(temp_path)
                elif file_name.endswith(".txt"): loader = TextLoader(temp_path, encoding="utf-8")
                else: continue
                docs = loader.load()
                for doc in docs: doc.metadata["source"] = file_name
                all_documents.extend(docs)
            except Exception as e: st.error(f"Dosya işleme hatası ({file_name}): {e}")
    
    if not all_documents: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(all_documents)

# --- 4. STATE MANAGEMENT ---

def init_state():
    if "player_balance" not in st.session_state: st.session_state.player_balance = 1000
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Ben DocuMentor. Doküman analiz edebilir, sohbet edebilir veya oyun oynayabiliriz."}]
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "simlish_mode" not in st.session_state: st.session_state.simlish_mode = False
    if "bj_deck_count" not in st.session_state: st.session_state.bj_deck_count = 6
    if "achievement_queue" not in st.session_state: st.session_state.achievement_queue = []
    
    if "achievements" not in st.session_state:
        st.session_state.achievements = {k: {"name": v["name"], "desc": v["desc"], "icon": v["icon"], "unlocked": False} for k, v in ACHIEVEMENT_LIST.items()}
        # Welcome başarımı
        if not st.session_state.achievements["gen_welcome"]["unlocked"]:
            st.session_state.achievements["gen_welcome"]["unlocked"] = True
            st.session_state.achievement_queue.append("gen_welcome")
        
    if "player_stats" not in st.session_state:
        st.session_state.player_stats = {
            "start_balance": 1000, "total_bets": 0, "total_won_amount": 0, "total_lost_amount": 0,
            "bj": {"played": 0, "won": 0, "lost": 0, "push": 0},
            "cf": {"played": 0, "won": 0, "lost": 0},
            "rl": {"played": 0, "won": 0, "lost": 0},
            "slot": {"played": 0, "won": 0, "lost": 0},
            "vp": {"played": 0, "won": 0, "lost": 0},
            "sc": {"played": 0, "won": 0, "lost": 0},
            "iq": {"played": 0, "max_level": 0},
            "biggest_win": 0, "biggest_loss": 0,
        }

init_state()

# --- 5. LOGIC & HELPER FUNCTIONS ---

def unlock_achievement(ach_id):
    if ach_id in st.session_state.achievements and not st.session_state.achievements[ach_id]["unlocked"]:
        st.session_state.achievements[ach_id]["unlocked"] = True
        st.session_state.achievement_queue.append(ach_id)

def check_stat_achievements():
    stats = st.session_state.player_stats
    bal = st.session_state.player_balance
    if bal >= 10000: unlock_achievement("gen_balance_10k")
    if bal < 100: unlock_achievement("gen_balance_100")
    if stats["bj"]["won"] >= 25: unlock_achievement("bj_win_25")
    if stats["bj"]["won"] >= 1: unlock_achievement("bj_win_1")
    if stats["cf"]["won"] >= 10: unlock_achievement("cf_win_10")
    if stats["rl"]["played"] >= 25: unlock_achievement("rl_played_25")
    if stats["slot"]["played"] >= 100: unlock_achievement("slot_played_100")
    if stats["vp"]["played"] >= 25: unlock_achievement("vp_played_25")
    if stats["iq"]["max_level"] >= 5: unlock_achievement("iq_level_5")
    
    # Tüm oyunları oynama kontrolü
    if all(stats[g]["played"] > 0 for g in ["bj", "cf", "rl", "slot", "vp", "sc"]):
        unlock_achievement("gen_played_all")

def add_history(game_key, bet, outcome, balance):
    # Geçmiş listesi yönetimi
    history_key = f"{game_key}_history"
    if history_key not in st.session_state: st.session_state[history_key] = []
    st.session_state[history_key].insert(0, {"bet": bet, "outcome": outcome, "balance": balance})
    st.session_state[history_key] = st.session_state[history_key][:5]

    # İstatistik Güncelleme
    try:
        stats = st.session_state.player_stats
        is_side_bet = False
        
        # Blackjack yan bahislerini ana oyun sayısından ayırmak için basit mantık
        if game_key == "bj":
            # Eğer bahis ana bahis değilse, yan bahistir
            if bet != st.session_state.get("bj_main_bet", -1) and bet > 0:
                is_side_bet = True
            
            # Sadece ana bahisleri oyun sayısı olarak say
            if not is_side_bet: 
                stats[game_key]["played"] += 1
        else:
            stats[game_key]["played"] += 1

        stats["total_bets"] += bet

        if outcome > 0:
            stats["total_won_amount"] += outcome
            stats["biggest_win"] = max(stats["biggest_win"], outcome)
            if not is_side_bet: stats[game_key]["won"] += 1
            if outcome >= 1000: unlock_achievement("gen_win_1000")
        elif outcome < 0:
            stats["total_lost_amount"] += abs(outcome)
            stats["biggest_loss"] = max(stats["biggest_loss"], abs(outcome))
            if not is_side_bet: stats[game_key]["lost"] += 1
        elif game_key == "bj" and not is_side_bet:
            stats["bj"]["push"] += 1

        st.session_state.player_stats = stats
        check_stat_achievements()

    except Exception as e:
        print(f"Stats error: {e}")

def display_history(game_key):
    history_key = f"{game_key}_history"
    if history_key in st.session_state and st.session_state[history_key]:
        with st.expander("Son Oyun Geçmişi"):
            for entry in st.session_state[history_key]:
                sign = "+" if entry["outcome"] > 0 else ""
                val = entry['outcome'] if entry['outcome'] != 0 else 'Push'
                st.markdown(f"**Bahis:** {entry['bet']} | **Sonuç:** {sign}{val} | **Bakiye:** {entry['balance']}")

# --- 6. UI TABS & STRUCTURE ---
st.title("DocuMentor 📄")

# Bildirim Sırası (Toast)
while st.session_state.achievement_queue:
    ach_id = st.session_state.achievement_queue.pop(0)
    ach = st.session_state.achievements[ach_id]
    st.toast(f"🏆 Başarım Açıldı: {ach['name']}", icon=ach['icon'])
    time.sleep(0.5)

# Tabs
tabs = st.tabs([
    "💬 Chat", "🃏 Blackjack", "🪙 Coin Flip", "🎡 Roulette", "🎰 Slots", 
    "🃏 Poker", "📊 Stats", "🏆 Achievements", "⛰️ Sisyphus", "🧠 IQ Test", 
    "🎶 Music", "🎨 Creative", "⚙️ Settings"
])

(tab_chat, tab_bj, tab_cf, tab_rl, tab_slot, tab_vp, tab_stats, tab_ach, tab_crash, tab_iq, tab_music, tab_creative, tab_settings) = tabs

# --- SIDEBAR ---
with st.sidebar:
    st.header("DocuMentor Dashboard")
    st.markdown(f"**Bakiye:** 💰 {st.session_state.player_balance}")
    
    if st.button("Her Şeyi Sıfırla 🔄"):
        st.session_state.player_balance = 1000
        # Reset stats
        st.session_state.player_stats = {
            "start_balance": 1000, "total_bets": 0, "total_won_amount": 0, "total_lost_amount": 0,
            "bj": {"played": 0, "won": 0, "lost": 0, "push": 0},
            "cf": {"played": 0, "won": 0, "lost": 0},
            "rl": {"played": 0, "won": 0, "lost": 0},
            "slot": {"played": 0, "won": 0, "lost": 0},
            "vp": {"played": 0, "won": 0, "lost": 0},
            "sc": {"played": 0, "won": 0, "lost": 0},
            "iq": {"played": 0, "max_level": 0},
            "biggest_win": 0, "biggest_loss": 0,
        }
        # Reset game states
        for key in list(st.session_state.keys()):
            if key.endswith("_state"):
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Geliştirici:** Göktuğ Türkdağ")
    st.markdown("🔗 [LinkedIn](https://linkedin.com/in/goktugturkdag)")
    st.markdown("🐙 [GitHub](https://github.com/goktug-turkdag)")

# --- TAB 1: CHATBOT (RAG + SAFETY) ---
with tab_chat:
    st.markdown("### Dokümanlarla Sohbet & Asistan")
    
    uploaded_files = st.file_uploader("PDF/DOCX/TXT Yükle", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        files_data = {f.name: f.getvalue() for f in uploaded_files}
        # Sadece yeni dosya varsa işle
        if "processed_file_names" not in st.session_state or set(files_data.keys()) != set(st.session_state.get("processed_file_names", [])):
            with st.spinner("Dokümanlar işleniyor ve vektör veritabanı oluşturuluyor..."):
                docs = process_uploaded_files(files_data)
                if docs:
                    st.session_state.processed_docs = docs
                    st.session_state.processed_file_names = list(files_data.keys())
                    st.success(f"{len(docs)} parça işlendi ve hafızaya alındı!")

    # Mesajları Göster
    for msg in st.session_state.messages:
        avatar = "👽" if msg["role"] == "assistant" and st.session_state.simlish_mode else None
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    # Kullanıcı Girdisi
    if prompt := st.chat_input("Bir soru sor..."):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").markdown(prompt)

        # Güvenlik Kontrolü
        lower_prompt = prompt.lower()
        is_safe = True
        warning_msg = ""

        if any(bad in lower_prompt for bad in BANNED_KEYWORDS):
            is_safe = False
            warning_msg = "Bu içerik güvenli değil veya politikalarımıza aykırı."
        elif any(pol in lower_prompt for pol in POLITICAL_KEYWORDS):
            is_safe = False
            warning_msg = "Siyasi konular hakkında yorum yapmıyorum."

        # Easter Egg: Geliştirici Hakkında
        if "göktuğ" in lower_prompt or "goktug" in lower_prompt:
            is_safe = False # LLM'e gitme, direkt cevapla
            warning_msg = "Göktuğ Türkdağ, bu uygulamanın geliştiricisidir. Ankara Üniversitesi İşletme öğrencisidir."

        with st.chat_message("assistant", avatar="👽" if st.session_state.simlish_mode else None):
            with st.spinner("Düşünüyor..."):
                response_text = ""
                
                if not is_safe:
                    response_text = warning_msg
                else:
                    try:
                        if "processed_docs" in st.session_state:
                             # RAG Flow
                             llm = load_llm()
                             embeddings = get_embeddings()
                             vectorstore = Chroma.from_documents(st.session_state.processed_docs, embeddings)
                             retriever = vectorstore.as_retriever()
                             chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
                             res = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                             response_text = res["answer"]
                        else:
                             # Normal Chat
                             llm = load_llm()
                             response_text = llm.invoke(prompt).content
                    except Exception as e:
                        response_text = f"Bir hata oluştu (API Key veya Bağlantı): {e}"

                if st.session_state.simlish_mode:
                    response_text = f"Sul sul! {response_text} Dag dag!"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                if is_safe and "processed_docs" in st.session_state:
                    st.session_state.chat_history.append((prompt, response_text))

# --- TAB 2: BLACKJACK (FULL LOGIC: Split, Double, Side Bets, AI Coach) ---
with tab_bj:
    st.header("🃏 Blackjack Pro")
    
    # BJ Init
    if "bj_state" not in st.session_state:
        st.session_state.bj_state = "betting"
        st.session_state.bj_hands = [] 
        st.session_state.bj_dealer = []
        st.session_state.bj_deck = []
        st.session_state.current_hand_idx = 0
        st.session_state.bj_coach_msg = ""

    # Kart Değerleri ve Yardımcılar
    def get_card_val(card):
        r = card[:-1]
        if r in ['J','Q','K']: return 10
        if r == 'A': return 11
        return int(r)

    def calc_bj_score(hand):
        score = sum([get_card_val(c) for c in hand])
        aces = sum([1 for c in hand if c.startswith('A')])
        while score > 21 and aces > 0:
            score -= 10
            aces -= 1
        return score

    def check_perfect_pairs(card1, card2):
        r1, s1 = card1[:-1], card1[-1]
        r2, s2 = card2[:-1], card2[-1]
        if r1 != r2: return 0
        if s1 == s2: return 25 # Perfect Pair
        if (s1 in "♥♦") == (s2 in "♥♦"): return 12 # Coloured Pair
        return 6 # Mixed Pair

    # AI Coach Logic
    def get_ai_advice(hand, dealer_card):
        score = calc_bj_score(hand)
        d_val = get_card_val(dealer_card)
        advice = ""
        if score >= 17: advice = "Stand. Elin güçlü."
        elif score <= 11: advice = "Hit. Kaybedecek bir şeyin yok."
        elif score == 12 and 4 <= d_val <= 6: advice = "Stand. Dealer batabilir."
        else: advice = "Matematiksel olarak Hit mantıklı olabilir ama riskli."
        return f"🤖 Coach: {advice}"

    # --- Betting Stage ---
    if st.session_state.bj_state == "betting":
        c1, c2, c3 = st.columns(3)
        bet = c1.number_input("Ana Bahis", 10, st.session_state.player_balance, 50, key="bj_main_bet_in")
        sb_pp = c2.number_input("Perfect Pairs (Yan)", 0, st.session_state.player_balance, 0, key="bj_pp_in")
        sb_213 = c3.number_input("21+3 (Yan)", 0, st.session_state.player_balance, 0, key="bj_213_in")
        
        if st.button("Kartları Dağıt", key="bj_deal_btn"):
            total_bet = bet + sb_pp + sb_213
            if st.session_state.player_balance < total_bet:
                st.error("Yetersiz bakiye.")
            else:
                st.session_state.player_balance -= total_bet
                st.session_state.bj_main_bet = bet 
                
                # Deste oluştur
                deck = [f"{r}{s}" for r in ['2','3','4','5','6','7','8','9','10','J','Q','K','A'] for s in "♠♥♣♦"] * st.session_state.bj_deck_count
                random.shuffle(deck)
                st.session_state.bj_deck = deck
                
                p_cards = [deck.pop(), deck.pop()]
                d_cards = [deck.pop(), deck.pop()]
                
                st.session_state.bj_hands = [{'cards': p_cards, 'bet': bet, 'status': 'active'}]
                st.session_state.bj_dealer = d_cards
                st.session_state.current_hand_idx = 0
                st.session_state.bj_state = "playing"
                st.session_state.bj_coach_msg = get_ai_advice(p_cards, d_cards[0])

                # --- Side Bets Resolution ---
                # Perfect Pairs
                if sb_pp > 0:
                    mult = check_perfect_pairs(p_cards[0], p_cards[1])
                    if mult > 0:
                        win = sb_pp * mult
                        st.session_state.player_balance += (sb_pp + win)
                        st.toast(f"Perfect Pairs Kazandı! +{win}", icon="💰")
                        add_history("bj", sb_pp, win, st.session_state.player_balance)
                        unlock_achievement("bj_side_bet")
                    else:
                        add_history("bj", sb_pp, -sb_pp, st.session_state.player_balance)
                
                # 21+3 (Basitleştirilmiş: Sadece Flush/Straight/Trips bakıyoruz)
                if sb_213 > 0:
                    ranks = sorted([get_card_val(c) for c in p_cards + [d_cards[0]]])
                    suits = [c[-1] for c in p_cards + [d_cards[0]]]
                    is_flush = len(set(suits)) == 1
                    is_trips = len(set([c[:-1] for c in p_cards + [d_cards[0]]])) == 1
                    win_213 = 0
                    if is_flush: win_213 = sb_213 * 9
                    elif is_trips: win_213 = sb_213 * 30
                    
                    if win_213 > 0:
                        st.session_state.player_balance += (sb_213 + win_213)
                        st.toast(f"21+3 Kazandı! +{win_213}", icon="💰")
                        add_history("bj", sb_213, win_213, st.session_state.player_balance)
                        unlock_achievement("bj_side_bet")
                    else:
                        add_history("bj", sb_213, -sb_213, st.session_state.player_balance)

                # Natural Blackjack Check
                if calc_bj_score(p_cards) == 21:
                    st.session_state.bj_hands[0]['status'] = 'blackjack'
                    st.session_state.bj_state = "dealer_turn" 
                
                st.rerun()

    # --- Playing Stage ---
    elif st.session_state.bj_state == "playing":
        d_card = st.session_state.bj_dealer[0]
        
        # Dealer Görsel
        col_d1, col_d2 = st.columns([1,3])
        with col_d1:
            st.markdown(f"""<div style='border:2px solid red; padding:10px; border-radius:5px; text-align:center'>
                        <h3>Dealer</h3><h1>{d_card}</h1></div>""", unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"<div style='padding:20px; font-size:24px'>🂠 Gizli Kart</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Player Hands Loop
        active_idx = st.session_state.current_hand_idx
        
        cols = st.columns(len(st.session_state.bj_hands))
        for idx, hand_data in enumerate(st.session_state.bj_hands):
            with cols[idx]:
                score = calc_bj_score(hand_data['cards'])
                status = hand_data['status']
                border_color = "green" if idx == active_idx else "gray"
                st.markdown(f"""<div style='border:2px solid {border_color}; padding:10px; border-radius:5px'>
                            <h4>El {idx+1} ({status})</h4>
                            <h2>{hand_data['cards']}</h2>
                            <p>Skor: {score} | Bahis: {hand_data['bet']}</p>
                            </div>""", unsafe_allow_html=True)

        # AI Coach Mesajı
        if st.session_state.bj_coach_msg:
            st.info(st.session_state.bj_coach_msg)

        # Action Buttons
        if active_idx < len(st.session_state.bj_hands):
            curr_hand = st.session_state.bj_hands[active_idx]
            curr_cards = curr_hand['cards']
            
            if curr_hand['status'] != 'active':
                st.session_state.current_hand_idx += 1
                st.rerun()
            
            c1, c2, c3, c4 = st.columns(4)
            
            # HIT
            if c1.button("Hit 👊"):
                curr_cards.append(st.session_state.bj_deck.pop())
                st.session_state.bj_coach_msg = get_ai_advice(curr_cards, d_card)
                score = calc_bj_score(curr_cards)
                
                # 5-Card Charlie Check
                if len(curr_cards) == 5 and score <= 21:
                    curr_hand['status'] = 'charlie'
                    st.session_state.current_hand_idx += 1
                    unlock_achievement("bj_charlie")
                elif score > 21:
                    curr_hand['status'] = 'bust'
                    st.session_state.current_hand_idx += 1
                st.rerun()
            
            # STAND
            if c2.button("Stand ✋"):
                curr_hand['status'] = 'stand'
                st.session_state.current_hand_idx += 1
                st.rerun()
            
            # DOUBLE
            if c3.button("Double 💰"):
                if st.session_state.player_balance >= curr_hand['bet']:
                    st.session_state.player_balance -= curr_hand['bet']
                    curr_hand['bet'] *= 2
                    curr_cards.append(st.session_state.bj_deck.pop())
                    if calc_bj_score(curr_cards) > 21:
                        curr_hand['status'] = 'bust'
                    else:
                        curr_hand['status'] = 'stand'
                    st.session_state.current_hand_idx += 1
                    st.rerun()
                else:
                    st.error("Bakiye Yetersiz")

            # SPLIT
            val1 = get_card_val(curr_cards[0])
            val2 = get_card_val(curr_cards[1])
            if c4.button("Split ↔️") and len(curr_cards) == 2 and val1 == val2:
                if st.session_state.player_balance >= curr_hand['bet']:
                    st.session_state.player_balance -= curr_hand['bet']
                    split_card = curr_cards.pop()
                    new_hand = {'cards': [split_card, st.session_state.bj_deck.pop()], 'bet': curr_hand['bet'], 'status': 'active'}
                    curr_cards.append(st.session_state.bj_deck.pop())
                    st.session_state.bj_hands.append(new_hand)
                    unlock_achievement("bj_split_win")
                    st.rerun()
                else:
                    st.error("Bakiye Yetersiz")
        else:
            st.session_state.bj_state = "dealer_turn"
            st.rerun()

    # --- Dealer Turn & Resolve ---
    elif st.session_state.bj_state == "dealer_turn":
        # Dealer oynar
        while calc_bj_score(st.session_state.bj_dealer) < 17:
            time.sleep(0.5)
            st.session_state.bj_dealer.append(st.session_state.bj_deck.pop())
        
        d_score = calc_bj_score(st.session_state.bj_dealer)
        st.markdown(f"### Dealer Final: {st.session_state.bj_dealer} (Skor: {d_score})")
        
        for i, hand in enumerate(st.session_state.bj_hands):
            p_score = calc_bj_score(hand['cards'])
            bet = hand['bet']
            status = hand['status']
            
            outcome = 0
            msg = f"**El {i+1}:** "
            
            if status == 'blackjack':
                if d_score == 21 and len(st.session_state.bj_dealer)==2:
                    msg += "Push (İkisi de BJ). Bahis iade."
                    st.session_state.player_balance += bet
                else:
                    msg += "BLACKJACK! 🎉 (2.5x Kazanç)"
                    win = int(bet * 2.5)
                    outcome = win - bet
                    st.session_state.player_balance += win
                    unlock_achievement("bj_blackjack")
            elif status == 'charlie':
                msg += "5-Card Charlie! 🎉 (Otomatik Kazanç)"
                win = bet * 2
                outcome = bet
                st.session_state.player_balance += win
            elif status == 'bust':
                msg += "Bust! 💥 (Kaybettin)"
                outcome = -bet
            else:
                if d_score > 21:
                    msg += "Dealer Bust! 🎉 (Kazandın)"
                    outcome = bet
                    st.session_state.player_balance += (bet * 2)
                elif p_score > d_score:
                    msg += "Kazandın! 🎉"
                    outcome = bet
                    st.session_state.player_balance += (bet * 2)
                elif p_score == d_score:
                    msg += "Push (Beraberlik). Bahis iade."
                    st.session_state.player_balance += bet
                else:
                    msg += "Dealer Kazandı. ❌"
                    outcome = -bet
            
            if outcome > 0: st.success(msg)
            elif outcome < 0: st.error(msg)
            else: st.warning(msg)
            
            add_history("bj", bet, outcome, st.session_state.player_balance)
            
        if st.button("Yeni El Dağıt"):
            st.session_state.bj_state = "betting"
            st.rerun()
            
    display_history("bj")

# --- TAB 3: COIN FLIP ---
with tab_cf:
    st.header("🪙 Yazı Tura Simülasyonu")
    
    col1, col2 = st.columns(2)
    with col1:
        bet = st.number_input("Bahis", 10, st.session_state.player_balance, 10, key="cf_bet")
        choice = st.radio("Seçimin:", ["Yazı", "Tura"], horizontal=True)
    
    with col2:
        if st.button("Parayı Havaya At", use_container_width=True):
            if bet > st.session_state.player_balance:
                st.error("Bakiye yetersiz!")
            else:
                st.session_state.player_balance -= bet
                
                # Animasyon efekti
                placeholder = st.empty()
                for _ in range(5):
                    placeholder.markdown(f"<h1 style='text-align:center'>🪙 ...</h1>", unsafe_allow_html=True)
                    time.sleep(0.1)
                    placeholder.markdown(f"<h1 style='text-align:center'>💫 ...</h1>", unsafe_allow_html=True)
                    time.sleep(0.1)
                
                res = random.choice(["Yazı", "Tura"])
                placeholder.markdown(f"<h1 style='text-align:center; color:gold'>{res}</h1>", unsafe_allow_html=True)
                
                if res == choice:
                    st.success(f"Kazandın! +{bet}")
                    st.session_state.player_balance += bet * 2
                    add_history("cf", bet, bet, st.session_state.player_balance)
                    st.balloons()
                else:
                    st.error("Kaybettin.")
                    add_history("cf", bet, -bet, st.session_state.player_balance)
    
    display_history("cf")

# --- TAB 4: ROULETTE (BOARD LAYOUT) ---
with tab_rl:
    st.header("🎡 Avrupa Ruleti")
    st.info("Bahislerini masaya yatır ve çarkı çevir!")
    
    # Bahis State
    if "rl_bets" not in st.session_state: st.session_state.rl_bets = {}
    
    # Bahis Giriş Alanı (Görsel Masa)
    bet_cols = st.columns(3)
    current_bet_total = 0
    
    with bet_cols[0]:
        st.markdown("**Renk & Sayı**")
        bet_red = st.number_input("Kırmızı (1:1)", 0, st.session_state.player_balance, 0, key="rl_red")
        bet_black = st.number_input("Siyah (1:1)", 0, st.session_state.player_balance, 0, key="rl_black")
        bet_green = st.number_input("Yeşil/0 (35:1)", 0, st.session_state.player_balance, 0, key="rl_green")
    
    with bet_cols[1]:
        st.markdown("**Tek/Çift**")
        bet_odd = st.number_input("Tek (1:1)", 0, st.session_state.player_balance, 0, key="rl_odd")
        bet_even = st.number_input("Çift (1:1)", 0, st.session_state.player_balance, 0, key="rl_even")
        bet_num = st.number_input("Özel Sayı Bahsi (35:1)", 0, st.session_state.player_balance, 0, key="rl_spec_bet")
        spec_num = st.number_input("Hangi Sayı? (1-36)", 1, 36, 1, key="rl_spec_num")

    with bet_cols[2]:
        st.markdown("**Aralık**")
        bet_low = st.number_input("1-18 (1:1)", 0, st.session_state.player_balance, 0, key="rl_low")
        bet_high = st.number_input("19-36 (1:1)", 0, st.session_state.player_balance, 0, key="rl_high")

    total_bet = bet_red + bet_black + bet_green + bet_odd + bet_even + bet_num + bet_low + bet_high
    st.markdown(f"### Toplam Bahis: {total_bet}")

    if st.button("ÇARKI ÇEVİR 🎡", use_container_width=True):
        if total_bet == 0:
            st.warning("En az bir bahis yapmalısın.")
        elif total_bet > st.session_state.player_balance:
            st.error("Bakiye yetersiz.")
        else:
            st.session_state.player_balance -= total_bet
            
            # Spin Logic
            with st.spinner("Çark dönüyor..."):
                time.sleep(1.5)
            
            res_num = random.randint(0, 36)
            res_color = "Yeşil" if res_num == 0 else "Kırmızı" if res_num in [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36] else "Siyah"
            
            st.metric("SONUÇ", f"{res_num} ({res_color})")
            
            total_win = 0
            
            # Payout Checks
            if bet_red > 0 and res_color == "Kırmızı": total_win += bet_red * 2
            if bet_black > 0 and res_color == "Siyah": total_win += bet_black * 2; unlock_achievement("rl_win_black")
            if bet_green > 0 and res_num == 0: total_win += bet_green * 36; unlock_achievement("rl_win_0")
            if bet_odd > 0 and res_num % 2 != 0 and res_num != 0: total_win += bet_odd * 2
            if bet_even > 0 and res_num % 2 == 0 and res_num != 0: total_win += bet_even * 2
            if bet_low > 0 and 1 <= res_num <= 18: total_win += bet_low * 2
            if bet_high > 0 and 19 <= res_num <= 36: total_win += bet_high * 2
            if bet_num > 0 and res_num == spec_num: total_win += bet_num * 36
            
            net_outcome = total_win - total_bet
            
            if total_win > 0:
                st.success(f"Tebrikler! Toplam Kazanç: {total_win} (Net: {net_outcome})")
                st.session_state.player_balance += total_win
                st.balloons()
            else:
                st.error("Hiçbir bahsin tutmadı.")
            
            add_history("rl", total_bet, net_outcome, st.session_state.player_balance)
            
    display_history("rl")

# --- TAB 5: SLOTS (AUTOSPIN) ---
with tab_slot:
    st.header("🎰 Slot Makinesi (Vegas Style)")
    
    col_set1, col_set2 = st.columns(2)
    bet = col_set1.number_input("Spin Başına Bahis", 5, st.session_state.player_balance, 10, key="sl_bet")
    auto_count = col_set2.selectbox("Otomatik Spin Sayısı", [1, 5, 10, 20])
    
    btn_col1, btn_col2 = st.columns(2)
    spin_btn = btn_col1.button("Tek Spin 🎲")
    auto_btn = btn_col2.button(f"Otomatik Başlat ({auto_count}x) 🔄")

    def run_spin(current_bet):
        syms = ["🍒", "🍋", "🍊", "🍉", "⭐", "💎", "❼"]
        weights = [25, 20, 15, 10, 8, 5, 2] # 7 gelme ihtimali düşük
        r1 = random.choices(syms, weights=weights, k=1)[0]
        r2 = random.choices(syms, weights=weights, k=1)[0]
        r3 = random.choices(syms, weights=weights, k=1)[0]
        return r1, r2, r3

    if spin_btn or auto_btn:
        loops = auto_count if auto_btn else 1
        placeholders = st.columns(3)
        msg_ph = st.empty()
        
        for i in range(loops):
            if st.session_state.player_balance < bet:
                st.warning("Bakiye tükendi!")
                break
                
            st.session_state.player_balance -= bet
            
            # Spin Animation
            if not auto_btn:
                for _ in range(5):
                    for ph in placeholders:
                        ph.markdown(f"<h1 style='text-align:center'>{random.choice(['🍒','🍋','❼'])}</h1>", unsafe_allow_html=True)
                    time.sleep(0.1)
            
            r1, r2, r3 = run_spin(bet)
            
            # Show Results
            placeholders[0].markdown(f"<h1 style='text-align:center'>{r1}</h1>", unsafe_allow_html=True)
            placeholders[1].markdown(f"<h1 style='text-align:center'>{r2}</h1>", unsafe_allow_html=True)
            placeholders[2].markdown(f"<h1 style='text-align:center'>{r3}</h1>", unsafe_allow_html=True)
            
            winnings = 0
            # Win Logic
            if r1 == r2 == r3:
                if r1 == "❼": winnings = bet * 100; unlock_achievement("slot_jackpot_7")
                elif r1 == "💎": winnings = bet * 50
                elif r1 == "⭐": winnings = bet * 25
                else: winnings = bet * 10
                if not auto_btn: st.balloons()
            elif r1 == r2 or r2 == r3 or r1 == r3: # Any pair
                if "🍒" in [r1, r2, r3]: # Cherry pair bonus
                    winnings = bet * 3
                    unlock_achievement("slot_win_cherry")
                else:
                    winnings = int(bet * 1.5)
            
            if winnings > 0:
                st.session_state.player_balance += winnings
                add_history("slot", bet, winnings - bet, st.session_state.player_balance)
                msg_ph.success(f"Spin {i+1}: Kazandın! +{winnings}")
            else:
                add_history("slot", bet, -bet, st.session_state.player_balance)
                msg_ph.info(f"Spin {i+1}: Kayıp.")
            
            if auto_btn: time.sleep(0.5)
            
    display_history("slot")

# --- TAB 6: VIDEO POKER ---
with tab_vp:
    st.header("🃏 Video Poker (Jacks or Better)")
    st.caption("J ve üzeri perler (Jacks or Better) kazandırır.")
    
    if "vp_stage" not in st.session_state: st.session_state.vp_stage = "deal"
    if "vp_hand" not in st.session_state: st.session_state.vp_hand = []
    
    vp_paytable = {
        "Royal Flush": 800, "Straight Flush": 50, "Four of a Kind": 25, "Full House": 9,
        "Flush": 6, "Straight": 4, "Three of a Kind": 3, "Two Pair": 2, "Jacks or Better": 1
    }
    
    if st.session_state.vp_stage == "deal":
        bet = st.number_input("Credit Bet", 1, 5, 1, key="vp_bet_in")
        if st.button("DAĞIT"):
            if bet > st.session_state.player_balance:
                st.error("Bakiye Yetersiz")
            else:
                st.session_state.player_balance -= bet
                st.session_state.vp_bet_val = bet
                deck = [r+s for r in '23456789TJQKA' for s in '♠♥♣♦']
                random.shuffle(deck)
                st.session_state.vp_deck = deck
                st.session_state.vp_hand = [deck.pop() for _ in range(5)]
                st.session_state.vp_stage = "draw"
                st.rerun()
            
    elif st.session_state.vp_stage == "draw":
        st.subheader("Kartları Tutmak için Seçin:")
        
        # Kart Görselleştirme
        cols = st.columns(5)
        holds = []
        for i, card in enumerate(st.session_state.vp_hand):
            color = "red" if card[-1] in "♥♦" else "black"
            if cols[i].checkbox(f"{card}", key=f"hold_{i}"):
                holds.append(i)
                cols[i].markdown(f"<div style='color:{color}; border:2px solid gold; padding:10px; text-align:center'><h1>{card}</h1>HELD</div>", unsafe_allow_html=True)
            else:
                cols[i].markdown(f"<div style='color:{color}; border:1px solid gray; padding:10px; text-align:center'><h1>{card}</h1></div>", unsafe_allow_html=True)
        
        if st.button("ÇEK (DRAW)"):
            # Kart değişimi
            new_hand = []
            for i in range(5):
                if i in holds:
                    new_hand.append(st.session_state.vp_hand[i])
                else:
                    new_hand.append(st.session_state.vp_deck.pop())
            st.session_state.vp_hand = new_hand
            
            # EL HESAPLAMA (Evaluation Logic)
            ranks = [c[:-1] for c in new_hand]
            suits = [c[-1] for c in new_hand]
            
            # Rank dönüşümü
            r_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
            r_nums = sorted([r_map[r] for r in ranks])
            counts = Counter(r_nums)
            
            is_flush = len(set(suits)) == 1
            is_straight = (r_nums[-1] - r_nums[0] == 4 and len(set(r_nums)) == 5) or r_nums == [2,3,4,5,14] # Wheel A-5
            
            hand_name = "High Card"
            mult = 0
            
            if is_flush and is_straight:
                if r_nums[-1] == 14 and r_nums[0] == 10: hand_name="Royal Flush"; mult=800; unlock_achievement("vp_win_royal")
                else: hand_name="Straight Flush"; mult=50
            elif 4 in counts.values(): 
                hand_name="Four of a Kind"; mult=25
                if 14 in [k for k,v in counts.items() if v==4]: unlock_achievement("vp_win_aces")
            elif 3 in counts.values() and 2 in counts.values(): hand_name="Full House"; mult=9
            elif is_flush: hand_name="Flush"; mult=6
            elif is_straight: hand_name="Straight"; mult=4
            elif 3 in counts.values(): hand_name="Three of a Kind"; mult=3
            elif list(counts.values()).count(2) == 2: hand_name="Two Pair"; mult=2
            elif 2 in counts.values():
                pair_rank = [k for k,v in counts.items() if v==2][0]
                if pair_rank >= 11: hand_name="Jacks or Better"; mult=1
            
            # Payout
            win = st.session_state.vp_bet_val * mult
            st.markdown("---")
            st.markdown(f"### Son El: {' '.join(new_hand)}")
            if win > 0:
                st.success(f"{hand_name}! Kazanç: {win}")
                st.session_state.player_balance += win
                add_history("vp", st.session_state.vp_bet_val, win, st.session_state.player_balance)
            else:
                st.error(f"{hand_name}. Kaybettin.")
                add_history("vp", st.session_state.vp_bet_val, -st.session_state.vp_bet_val, st.session_state.player_balance)
            
            if st.button("Yeni El"):
                st.session_state.vp_stage = "deal"
                st.rerun()
                
    display_history("vp")

# --- TAB 7 & 8: STATS & ACHIEVEMENTS ---
with tab_stats:
    st.markdown("### 📊 Detaylı İstatistikler")
    stats = st.session_state.player_stats
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Bahis", stats["total_bets"])
    col2.metric("En Büyük Kazanç", stats["biggest_win"])
    col3.metric("En Büyük Kayıp", stats["biggest_loss"])
    
    st.json(stats)

with tab_ach:
    st.header("🏆 Başarımlar")
    cols = st.columns(4)
    idx = 0
    for k, v in st.session_state.achievements.items():
        with cols[idx % 4]:
            if v["unlocked"]:
                st.success(f"**{v['icon']} {v['name']}**\n\n{v['desc']}")
            else:
                st.info(f"**🔒 {v['name']}**\n\n???")
        idx += 1

# --- TAB 9: SISYPHUS (CRASH GAME) ---
with tab_crash:
    st.header("⛰️ Sisyphus' Climb")
    st.caption("Camus'nün Absürt Kahramanı: Kaya ne kadar yükseğe çıkacak?")
    
    if "sc_state" not in st.session_state: st.session_state.sc_state = "betting"
    
    # CSS Animasyonu
    def render_sisyphus(multiplier, crashed=False):
        color = "red" if crashed else "green"
        pos = min(90, (multiplier - 1) * 10) # Basit pozisyon hesabı
        st.markdown(f"""
        <div style="width:100%; height:200px; background:linear-gradient(to top, #444, #111); position:relative; border-bottom:5px solid #654321;">
            <div style="position:absolute; bottom:{pos}%; left:{pos}%; transition:all 0.1s;">
                <span style="font-size:40px">🪨🏃</span>
            </div>
            <div style="position:absolute; top:10px; right:10px; color:{color}; font-size:50px; font-weight:bold">
                {multiplier:.2f}x
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.sc_state == "betting":
        bet = st.number_input("Bahis", 10, st.session_state.player_balance, key="sc_bet")
        if st.button("Tırmanışa Başla"):
            if bet > st.session_state.player_balance:
                st.error("Bakiye Yetersiz")
            else:
                st.session_state.player_balance -= bet
                st.session_state.sc_bet_val = bet
                # Crash noktası (Pareto distribution benzeri)
                st.session_state.sc_crash = max(1.0, random.paretovariate(2.0))
                if st.session_state.sc_crash > 50: st.session_state.sc_crash = 50
                st.session_state.sc_curr = 1.0
                st.session_state.sc_state = "climbing"
                st.rerun()
            render_sisyphus(1.0)
            
    elif st.session_state.sc_state == "climbing":
        # Simülasyon Döngüsü (Streamlit'te while loop ile UI güncelleme)
        placeholder = st.empty()
        
        # Basit artış, crash'e kadar
        time.sleep(0.1) # Hız kontrolü
        st.session_state.sc_curr += 0.01 + (st.session_state.sc_curr * 0.02)
        
        with placeholder.container():
            render_sisyphus(st.session_state.sc_curr)
        
        if st.session_state.sc_curr >= st.session_state.sc_crash:
            st.error(f"💥 KAYA YUVARLANDI! {st.session_state.sc_crash:.2f}x")
            add_history("sc", st.session_state.sc_bet_val, -st.session_state.sc_bet_val, st.session_state.player_balance)
            if st.session_state.sc_crash < 1.05: unlock_achievement("sc_crash_early")
            st.session_state.sc_state = "betting"
            if st.button("Tekrar Dene"): st.rerun()
        
        else:
            # Cashout Butonu
            if st.button("💰 CASHOUT 💰"):
                win = int(st.session_state.sc_bet_val * st.session_state.sc_curr)
                st.session_state.player_balance += win
                st.success(f"Başardın! {st.session_state.sc_curr:.2f}x oranında {win} kazandın.")
                add_history("sc", st.session_state.sc_bet_val, win - st.session_state.sc_bet_val, st.session_state.player_balance)
                
                if st.session_state.sc_curr > 20: unlock_achievement("sc_cashout_20x")
                if st.session_state.sc_curr > 50: unlock_achievement("sc_cashout_50x")
                if 1.00 < st.session_state.sc_curr < 1.02: unlock_achievement("sc_cashout_1_01x")
                
                st.session_state.sc_state = "betting"
                st.balloons()
                if st.button("Yeni Oyun"): st.rerun()
            else:
                st.rerun()

# --- TAB 10: COGNITIVE TEST (MEMORY) ---
with tab_iq:
    st.header("🧠 Sayı Hafıza Testi (Şempanze Testi)")
    st.caption("Ekranda beliren sayıyı ezberle, kaybolunca yaz.")
    
    if "iq_state" not in st.session_state:
        st.session_state.iq_state = "start"
        st.session_state.iq_level = 1
        st.session_state.iq_num = ""

    if st.session_state.iq_state == "start":
        if st.button("Testi Başlat"):
            st.session_state.iq_state = "show"
            st.session_state.iq_level = 1
            st.rerun()

    elif st.session_state.iq_state == "show":
        length = st.session_state.iq_level + 2 # Lvl 1 = 3 hane
        num = "".join([str(random.randint(0,9)) for _ in range(length)])
        st.session_state.iq_num = num
        
        ph = st.empty()
        ph.markdown(f"<h1 style='text-align:center; font-size:80px'>{num}</h1>", unsafe_allow_html=True)
        
        # Timer Bar
        prog = st.progress(100)
        wait_time = max(1.0, 3.0 - (st.session_state.iq_level * 0.2)) # Zorlaştıkça süre azalır
        step = wait_time / 20
        for i in range(20):
            time.sleep(step)
            prog.progress(100 - (i*5))
        
        ph.empty()
        prog.empty()
        st.session_state.iq_state = "input"
        st.rerun()

    elif st.session_state.iq_state == "input":
        st.subheader(f"Seviye {st.session_state.iq_level}")
        with st.form("iq_form"):
            guess = st.text_input("Gördüğün sayı neydi?", autocomplete="off")
            if st.form_submit_button("Kontrol Et"):
                if guess == st.session_state.iq_num:
                    st.success("Doğru! Sonraki seviyeye geçiliyor...")
                    st.session_state.iq_level += 1
                    # Stats update
                    st.session_state.player_stats["iq"]["max_level"] = max(st.session_state.player_stats["iq"]["max_level"], st.session_state.iq_level)
                    check_stat_achievements()
                    
                    time.sleep(1)
                    st.session_state.iq_state = "show"
                    st.rerun()
                else:
                    st.error(f"Yanlış! Doğru cevap: {st.session_state.iq_num}")
                    st.info(f"Oyun Bitti. Ulaştığın Seviye: {st.session_state.iq_level}")
                    st.session_state.iq_state = "start"
                    if st.button("Tekrar Dene"): st.rerun()

# --- TAB 11-13: EXTRAS (MUSIC, CREATIVE, SETTINGS) ---
with tab_music:
    st.header("🎶 Müzik Player")
    m_opt = st.radio("Modunu Seç:", ["Blues Focus", "Guilty Pleasure"])
    if m_opt == "Blues Focus":
        st.video("https://www.youtube.com/watch?v=1eNSWZ4x2ZU")
    else:
        st.video("https://www.youtube.com/watch?v=yQ9lXHfv9Yg")

with tab_creative:
    st.header("🎨 Creative Corner")
    st.caption("AI ile şiir, hikaye veya tweet oluştur.")
    
    ctype = st.selectbox("Tür:", ["Şiir", "Kısa Hikaye", "Haiku", "Tweet"])
    prompt_c = st.text_input("Konu:")
    
    if st.button("Oluştur") and prompt_c:
        with st.spinner("Yaratıcılık yükleniyor..."):
            try:
                llm = load_creative_llm()
                # Yasaklı kelime kontrolü
                if any(bad in prompt_c.lower() for bad in BANNED_KEYWORDS):
                    st.error("Bu konu hakkında içerik üretemem.")
                else:
                    template = f"Sen yaratıcı bir yazarsın. Konu: {prompt_c}. Tür: {ctype}. Lütfen yaratıcı ve ilgi çekici bir metin yaz."
                    res = llm.invoke(template).content
                    st.success("Sonuç:")
                    st.write(res)
            except Exception as e:
                st.error(f"Hata: {e}")

with tab_settings:
    st.header("⚙️ Ayarlar")
    st.session_state.simlish_mode = st.toggle("Simlish Modu (Chatbot dili)", st.session_state.simlish_mode)
    st.session_state.bj_deck_count = st.slider("Blackjack Deste Sayısı", 1, 8, 6)
    
    if st.button("Önbelleği Temizle (Cache Clear)"):
        st.cache_resource.clear()
        st.success("Önbellek temizlendi.")

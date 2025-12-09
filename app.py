# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
import random
import tempfile
import math
from collections import Counter
from dotenv import load_dotenv

# --- LANGCHAIN & AI IMPORTS ---
try:
    # CHROMA YERİNE FAISS KULLANIYORUZ (HATA ÇIKARMAZ)
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
except ImportError as e:
    st.error(f"Kütüphane Hatası: {e}. Lütfen requirements.txt dosyanızı güncelleyin.")
    st.stop()

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="DocuMentor Pro", layout="wide", page_icon="📄")
load_dotenv()

# API Key Kontrolü (Hata vermemesi için pass geçildi, .env dosyanızda olmalı)
if "GOOGLE_API_KEY" not in os.environ:
    pass 

# --- 2. BAŞARIM LİSTESİ ---
ACHIEVEMENT_LIST = {
    "gen_welcome": {"name": "Hoşgeldin Paketi", "desc": "Oyunu ilk kez açtın.", "icon": "👋"},
    "gen_balance_10k": {"name": "Nostradamus", "desc": "Bakiyeni 10,000'in üzerine çıkar.", "icon": "🔮"},
    "gen_balance_100": {"name": "Son Pişmanlık", "desc": "Bakiyen 100'ün altına düştü.", "icon": "🎻"},
    "gen_win_1000": {"name": "I'M NOT LEAVING!", "desc": "Tek bahisten 1000+ kazan.", "icon": "🐺"},
    "gen_played_all": {"name": "Mekanın Sahibi", "desc": "Tüm oyunları en az bir kez oyna.", "icon": "👑"},
    "bj_blackjack": {"name": "Natural 21", "desc": "İlk iki kartta Blackjack yap.", "icon": "✨"},
    "bj_win_25": {"name": "Kasa Katili", "desc": "Blackjack'te 25 el kazan.", "icon": "🦈"},
    "bj_split_win": {"name": "Dublör", "desc": "Split yaptıktan sonra kazan.", "icon": "👯"},
    "cf_win_10": {"name": "Yazı Tura", "desc": "10 kez Yazı Tura kazan.", "icon": "🪙"},
    "rl_win_0": {"name": "Yeşil Yol", "desc": "Rulette 0'a (Yeşil) bas ve kazan.", "icon": "📿"},
    "slot_jackpot_7": {"name": "Midas", "desc": "Slotlarda 7-7-7 yakala.", "icon": "💎"},
    "sc_cashout_20x": {"name": "Fly Me to the Moon", "desc": "Sisyphus'ta 20x üzeri cashout.", "icon": "🚀"},
    "iq_level_5": {"name": "Fil Hafızası", "desc": "Hafıza testinde 5. seviyeye ulaş.", "icon": "🐘"},
    "bj_side_bet": {"name": "Yan Gelir", "desc": "Bir yan bahis kazan.", "icon": "💸"},
    "sc_crash_early": {"name": "Absürd", "desc": "1.05x altı patla.", "icon": "💥"},
    "sc_cashout_1_01x": {"name": "Risk Budur", "desc": "Tam 1.01x'te cashout yap.", "icon": "🐔"},
    "vp_win_royal": {"name": "Ezel", "desc": "Royal Flush yakala.", "icon": "👑"},
    "vp_win_aces": {"name": "Kare As", "desc": "Dört As yakala.", "icon": "♠️"},
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
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Ben DocuMentor. Doküman yükleyebilir veya oyun oynayabiliriz."}]
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

# --- TAB 1: CHATBOT (RAG - FAISS Version) ---
with tab_chat:
    st.markdown("### Dokümanlarla Sohbet & Asistan")
    
    uploaded_files = st.file_uploader("PDF/DOCX/TXT Yükle", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        files_data = {f.name: f.getvalue() for f in uploaded_files}
        # Sadece yeni dosya varsa işle
        if "processed_file_names" not in st.session_state or set(files_data.keys()) != set(st.session_state.get("processed_file_names", [])):
            with st.spinner("Dokümanlar işleniyor (FAISS Index)..."):
                docs = process_uploaded_files(files_data)
                if docs:
                    st.session_state.processed_docs = docs
                    st.session_state.processed_file_names = list(files_data.keys())
                    st.success(f"{len(docs)} parça hafızaya alındı!")

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
        
        # Basit güvenlik listesi
        BANNED = ["kill", "suicide", "bomb", "hack", "terror", "porn", "sex", "racist"]
        if any(bad in lower_prompt for bad in BANNED):
            is_safe = False
            response_text = "Bu içerik güvenli değil."

        # Easter Egg
        elif "göktuğ" in lower_prompt or "goktug" in lower_prompt:
            is_safe = False
            response_text = "Göktuğ Türkdağ, bu uygulamanın geliştiricisidir."

        with st.chat_message("assistant", avatar="👽" if st.session_state.simlish_mode else None):
            with st.spinner("Düşünüyor..."):
                if is_safe:
                    try:
                        llm = load_llm()
                        if "processed_docs" in st.session_state:
                             # FAISS RAG Flow
                             embeddings = get_embeddings()
                             # Create Vectorstore on the fly (In-Memory)
                             vectorstore = FAISS.from_documents(st.session_state.processed_docs, embeddings)
                             retriever = vectorstore.as_retriever()
                             chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
                             res = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                             response_text = res["answer"]
                        else:
                             # Normal Chat
                             response_text = llm.invoke(prompt).content
                    except Exception as e:
                        response_text = f"Bir hata oluştu: {e}"

                if st.session_state.simlish_mode:
                    response_text = f"Sul sul! {response_text} Dag dag!"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                if is_safe and "processed_docs" in st.session_state:
                    st.session_state.chat_history.append((prompt, response_text))

# --- TAB 2: BLACKJACK (FULL) ---
with tab_bj:
    st.header("🃏 Blackjack Pro")
    
    if "bj_state" not in st.session_state:
        st.session_state.bj_state = "betting"
        st.session_state.bj_hands = [] 
        st.session_state.bj_dealer = []
        st.session_state.bj_deck = []
        st.session_state.current_hand_idx = 0
        st.session_state.bj_coach_msg = ""

    def get_card_val(card):
        r = card[:-1]
        if r in ['J','Q','K']: return 10
        if r == 'A': return 11
        return int(r)

    def calc_bj_score(hand):
        score = sum([get_card_val(c) for c in hand])
        aces = sum([1 for c in hand if c.startswith('A')])
        while score > 21 and aces > 0: score -= 10; aces -= 1
        return score

    def check_perfect_pairs(card1, card2):
        r1, s1 = card1[:-1], card1[-1]
        r2, s2 = card2[:-1], card2[-1]
        if r1 != r2: return 0
        if s1 == s2: return 25 
        if (s1 in "♥♦") == (s2 in "♥♦"): return 12 
        return 6 

    def get_ai_advice(hand, dealer_card):
        score = calc_bj_score(hand)
        d_val = get_card_val(dealer_card)
        if score >= 17: return "🤖 Coach: Stand."
        elif score <= 11: return "🤖 Coach: Hit."
        elif score == 12 and 4 <= d_val <= 6: return "🤖 Coach: Stand (Dealer zayıf)."
        return "🤖 Coach: Dikkatli oyna."

    if st.session_state.bj_state == "betting":
        c1, c2, c3 = st.columns(3)
        bet = c1.number_input("Ana Bahis", 10, st.session_state.player_balance, 50, key="bj_main_bet_in")
        sb_pp = c2.number_input("Perfect Pairs (Yan)", 0, st.session_state.player_balance, 0, key="bj_pp_in")
        sb_213 = c3.number_input("21+3 (Yan)", 0, st.session_state.player_balance, 0, key="bj_213_in")
        
        if st.button("Dağıt", key="bj_deal_btn"):
            total_bet = bet + sb_pp + sb_213
            if st.session_state.player_balance < total_bet:
                st.error("Yetersiz bakiye.")
            else:
                st.session_state.player_balance -= total_bet
                st.session_state.bj_main_bet = bet 
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

                if sb_pp > 0:
                    mult = check_perfect_pairs(p_cards[0], p_cards[1])
                    if mult > 0:
                        win = sb_pp * mult
                        st.session_state.player_balance += (sb_pp + win)
                        st.toast(f"Perfect Pairs Kazandı! +{win}")
                        add_history("bj", sb_pp, win, st.session_state.player_balance)
                        unlock_achievement("bj_side_bet")
                    else:
                        add_history("bj", sb_pp, -sb_pp, st.session_state.player_balance)
                
                if sb_213 > 0:
                    add_history("bj", sb_213, -sb_213, st.session_state.player_balance) # Basitleştirilmiş kayıp varsayımı

                if calc_bj_score(p_cards) == 21:
                    st.session_state.bj_hands[0]['status'] = 'blackjack'
                    st.session_state.bj_state = "dealer_turn" 
                st.rerun()

    elif st.session_state.bj_state == "playing":
        d_card = st.session_state.bj_dealer[0]
        st.markdown(f"**Dealer:** {d_card}, 🂠")
        
        active_idx = st.session_state.current_hand_idx
        for idx, hand_data in enumerate(st.session_state.bj_hands):
            score = calc_bj_score(hand_data['cards'])
            status = hand_data['status']
            b_color = "green" if idx == active_idx else "gray"
            st.markdown(f"<div style='border:2px solid {b_color}; padding:5px'>El {idx+1}: {hand_data['cards']} ({score})</div>", unsafe_allow_html=True)

        if st.session_state.bj_coach_msg: st.info(st.session_state.bj_coach_msg)

        if active_idx < len(st.session_state.bj_hands):
            curr_hand = st.session_state.bj_hands[active_idx]
            curr_cards = curr_hand['cards']
            
            if curr_hand['status'] != 'active':
                st.session_state.current_hand_idx += 1
                st.rerun()
            
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Hit"):
                curr_cards.append(st.session_state.bj_deck.pop())
                st.session_state.bj_coach_msg = get_ai_advice(curr_cards, d_card)
                score = calc_bj_score(curr_cards)
                if len(curr_cards) == 5 and score <= 21:
                    curr_hand['status'] = 'charlie'; st.session_state.current_hand_idx += 1; unlock_achievement("bj_charlie")
                elif score > 21:
                    curr_hand['status'] = 'bust'; st.session_state.current_hand_idx += 1
                st.rerun()
            if c2.button("Stand"):
                curr_hand['status'] = 'stand'; st.session_state.current_hand_idx += 1; st.rerun()
            if c3.button("Double"):
                if st.session_state.player_balance >= curr_hand['bet']:
                    st.session_state.player_balance -= curr_hand['bet']
                    curr_hand['bet'] *= 2
                    curr_cards.append(st.session_state.bj_deck.pop())
                    if calc_bj_score(curr_cards) > 21: curr_hand['status'] = 'bust'
                    else: curr_hand['status'] = 'stand'
                    st.session_state.current_hand_idx += 1; st.rerun()
            
            val1 = get_card_val(curr_cards[0])
            val2 = get_card_val(curr_cards[1])
            if c4.button("Split") and len(curr_cards) == 2 and val1 == val2:
                if st.session_state.player_balance >= curr_hand['bet']:
                    st.session_state.player_balance -= curr_hand['bet']
                    split_card = curr_cards.pop()
                    new_hand = {'cards': [split_card, st.session_state.bj_deck.pop()], 'bet': curr_hand['bet'], 'status': 'active'}
                    curr_cards.append(st.session_state.bj_deck.pop())
                    st.session_state.bj_hands.append(new_hand)
                    unlock_achievement("bj_split_win")
                    st.rerun()
        else:
            st.session_state.bj_state = "dealer_turn"
            st.rerun()

    elif st.session_state.bj_state == "dealer_turn":
        while calc_bj_score(st.session_state.bj_dealer) < 17:
            st.session_state.bj_dealer.append(st.session_state.bj_deck.pop())
        
        d_score = calc_bj_score(st.session_state.bj_dealer)
        st.subheader(f"Dealer Final: {st.session_state.bj_dealer} ({d_score})")
        
        for i, hand in enumerate(st.session_state.bj_hands):
            p_score = calc_bj_score(hand['cards'])
            bet = hand['bet']
            status = hand['status']
            outcome = 0
            
            if status == 'blackjack':
                if d_score == 21 and len(st.session_state.bj_dealer)==2: st.session_state.player_balance += bet
                else: win = int(bet * 2.5); outcome = win - bet; st.session_state.player_balance += win; unlock_achievement("bj_blackjack")
            elif status == 'charlie': win = bet * 2; outcome = bet; st.session_state.player_balance += win
            elif status == 'bust': outcome = -bet
            else:
                if d_score > 21: outcome = bet; st.session_state.player_balance += (bet * 2)
                elif p_score > d_score: outcome = bet; st.session_state.player_balance += (bet * 2)
                elif p_score == d_score: st.session_state.player_balance += bet
                else: outcome = -bet
            
            if outcome > 0: st.success(f"El {i+1}: Kazandın! +{outcome}")
            elif outcome < 0: st.error(f"El {i+1}: Kaybettin.")
            else: st.warning(f"El {i+1}: Push.")
            add_history("bj", bet, outcome, st.session_state.player_balance)
            
        if st.button("Yeni El"):
            st.session_state.bj_state = "betting"
            st.rerun()
            
    display_history("bj")

# --- TAB 3: COIN FLIP ---
with tab_cf:
    st.header("🪙 Yazı Tura")
    c1, c2 = st.columns(2)
    bet = c1.number_input("Bahis", 10, st.session_state.player_balance, 10, key="cf_bet")
    choice = c1.radio("Seçim", ["Yazı", "Tura"])
    if c2.button("At"):
        if bet > st.session_state.player_balance: st.error("Yetersiz Bakiye")
        else:
            st.session_state.player_balance -= bet
            res = random.choice(["Yazı", "Tura"])
            ph = st.empty()
            ph.markdown(f"<h1 style='text-align:center; font-size:60px'>🪙</h1>", unsafe_allow_html=True)
            time.sleep(0.5)
            ph.markdown(f"<h1 style='text-align:center; font-size:60px'>{res}</h1>", unsafe_allow_html=True)
            if res == choice:
                st.success("Kazandın!")
                st.session_state.player_balance += bet * 2
                add_history("cf", bet, bet, st.session_state.player_balance)
                st.balloons()
            else:
                st.error("Kaybettin.")
                add_history("cf", bet, -bet, st.session_state.player_balance)
    display_history("cf")

# --- TAB 4: ROULETTE ---
with tab_rl:
    st.header("🎡 Rulet")
    b_col = st.columns(3)
    bet_red = b_col[0].number_input("Kırmızı", 0, st.session_state.player_balance, key="r_red")
    bet_black = b_col[1].number_input("Siyah", 0, st.session_state.player_balance, key="r_blk")
    bet_green = b_col[2].number_input("Yeşil (0)", 0, st.session_state.player_balance, key="r_grn")
    
    total = bet_red + bet_black + bet_green
    if st.button("Çevir"):
        if total > st.session_state.player_balance: st.error("Yetersiz")
        elif total == 0: st.warning("Bahis yap")
        else:
            st.session_state.player_balance -= total
            num = random.randint(0, 36)
            color = "Yeşil" if num == 0 else "Kırmızı" if num in [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36] else "Siyah"
            st.metric("Sonuç", f"{num} ({color})")
            win = 0
            if bet_red > 0 and color=="Kırmızı": win += bet_red*2
            if bet_black > 0 and color=="Siyah": win += bet_black*2; unlock_achievement("rl_win_black")
            if bet_green > 0 and num==0: win += bet_green*36; unlock_achievement("rl_win_0")
            
            if win > 0: st.success(f"Kazanç: {win}"); st.session_state.player_balance += win; st.balloons()
            else: st.error("Kayıp")
            add_history("rl", total, win-total, st.session_state.player_balance)
    display_history("rl")

# --- TAB 5: SLOTS ---
with tab_slot:
    st.header("🎰 Slot")
    bet = st.number_input("Bahis", 5, st.session_state.player_balance, key="sl_bet")
    if st.button("Spin"):
        if st.session_state.player_balance < bet: st.error("Yetersiz")
        else:
            st.session_state.player_balance -= bet
            syms = ["🍒", "🍋", "🍊", "💎", "❼"]
            r1, r2, r3 = [random.choice(syms) for _ in range(3)]
            cols = st.columns(3)
            cols[0].markdown(f"# {r1}")
            cols[1].markdown(f"# {r2}")
            cols[2].markdown(f"# {r3}")
            
            win = 0
            if r1==r2==r3:
                if r1=="❼": win=bet*100; unlock_achievement("slot_jackpot_7")
                elif r1=="💎": win=bet*50
                else: win=bet*10
                st.balloons()
            elif r1==r2 or r2==r3 or r1==r3: win=int(bet*1.5)
            
            if win>0: st.success(f"Kazandın: {win}"); st.session_state.player_balance+=win
            else: st.error("Kaybettin")
            add_history("slot", bet, win-bet, st.session_state.player_balance)
    display_history("slot")

# --- TAB 6: POKER ---
with tab_vp:
    st.header("🃏 Video Poker")
    if "vp_stage" not in st.session_state: st.session_state.vp_stage="deal"
    
    if st.session_state.vp_stage=="deal":
        bet=st.number_input("Bahis", 1, 50, 5, key="vp_bet")
        if st.button("Dağıt"):
            st.session_state.player_balance -= bet
            st.session_state.vp_bet_val = bet
            d = [r+s for r in '23456789TJQKA' for s in '♠♥♣♦']; random.shuffle(d)
            st.session_state.vp_deck = d
            st.session_state.vp_hand = [d.pop() for _ in range(5)]
            st.session_state.vp_stage = "draw"
            st.rerun()
    elif st.session_state.vp_stage=="draw":
        cols=st.columns(5); holds=[]
        for i,c in enumerate(st.session_state.vp_hand):
            if cols[i].checkbox(c, key=f"h{i}"): holds.append(i)
        if st.button("Değiştir"):
            nh=[st.session_state.vp_hand[i] if i in holds else st.session_state.vp_deck.pop() for i in range(5)]
            st.write(f"Son El: {nh}")
            ranks=[c[0] for c in nh]; cnt=Counter(ranks)
            win=0; hand="Loss"
            if any(v==4 for v in cnt.values()): win=25; hand="Four of a Kind"; unlock_achievement("vp_win_aces") if 'A' in ranks else None
            elif 3 in cnt.values() and 2 in cnt.values(): win=9; hand="Full House"
            elif any(v==3 for v in cnt.values()): win=3; hand="3 of a Kind"
            elif list(cnt.values()).count(2)==2: win=2; hand="Two Pair"
            elif any(cnt[r]==2 for r in 'JQKA'): win=1; hand="Jacks+"
            
            pay=st.session_state.vp_bet_val*win
            if pay>0: st.success(f"{hand}: {pay}"); st.session_state.player_balance+=pay
            else: st.error(hand)
            add_history("vp", st.session_state.vp_bet_val, pay, st.session_state.player_balance)
            st.session_state.vp_stage="deal"
    display_history("vp")

# --- TAB 7 & 8: STATS & ACHIEVEMENTS ---
with tab_stats: st.json(st.session_state.player_stats)
with tab_ach:
    cols=st.columns(4); i=0
    for k,v in st.session_state.achievements.items():
        with cols[i%4]:
            if v['unlocked']: st.success(f"{v['icon']} {v['name']}")
            else: st.info("🔒 ???")
        i+=1

# --- TAB 9: SISYPHUS ---
with tab_crash:
    st.header("⛰️ Sisyphus Crash")
    if "sc_state" not in st.session_state: st.session_state.sc_state="betting"
    
    if st.session_state.sc_state=="betting":
        bet=st.number_input("Bahis", 10, st.session_state.player_balance, key="sc_bet")
        if st.button("Tırman"):
            st.session_state.player_balance -= bet
            st.session_state.sc_bet_val = bet
            st.session_state.sc_crash = max(1.0, random.paretovariate(1.5))
            if st.session_state.sc_crash > 20: unlock_achievement("sc_cashout_20x")
            st.session_state.sc_curr = 1.0
            st.session_state.sc_state = "climbing"
            st.rerun()
    elif st.session_state.sc_state=="climbing":
        st.metric("Çarpan", f"{st.session_state.sc_curr:.2f}x")
        time.sleep(0.1); st.session_state.sc_curr += 0.05
        if st.session_state.sc_curr >= st.session_state.sc_crash:
            st.error(f"Patladı! {st.session_state.sc_crash:.2f}x"); st.session_state.sc_state="betting"
            add_history("sc", st.session_state.sc_bet_val, -st.session_state.sc_bet_val, st.session_state.player_balance)
            if st.session_state.sc_crash < 1.05: unlock_achievement("sc_crash_early")
        elif st.button("Çekil"):
            win=int(st.session_state.sc_bet_val*st.session_state.sc_curr)
            st.session_state.player_balance += win
            st.success(f"Kazandın: {win}"); st.session_state.sc_state="betting"
            add_history("sc", st.session_state.sc_bet_val, win-st.session_state.sc_bet_val, st.session_state.player_balance)
            if st.session_state.sc_curr > 1.0 and st.session_state.sc_curr < 1.02: unlock_achievement("sc_cashout_1_01x")
            st.rerun()
        else: st.rerun()

# --- TAB 10: IQ TEST ---
with tab_iq:
    st.header("🧠 Hafıza Testi")
    if "iq_state" not in st.session_state: st.session_state.iq_state="start"; st.session_state.iq_lvl=1
    
    if st.session_state.iq_state=="start":
        if st.button("Başla"): st.session_state.iq_state="show"; st.rerun()
    elif st.session_state.iq_state=="show":
        num="".join([str(random.randint(0,9)) for _ in range(st.session_state.iq_lvl+2)])
        st.session_state.iq_num=num
        st.markdown(f"<h1 style='text-align:center; font-size:80px'>{num}</h1>", unsafe_allow_html=True)
        bar=st.progress(100)
        for i in range(20): time.sleep(0.1); bar.progress(100-i*5)
        st.session_state.iq_state="input"; st.rerun()
    elif st.session_state.iq_state=="input":
        with st.form("iq"):
            g=st.text_input("Sayı?")
            if st.form_submit_button("Tamam"):
                if g==st.session_state.iq_num:
                    st.success("Doğru!"); st.session_state.iq_lvl+=1; time.sleep(1)
                    st.session_state.player_stats["iq"]["max_level"] = max(st.session_state.player_stats["iq"]["max_level"], st.session_state.iq_lvl)
                    check_stat_achievements()
                    st.session_state.iq_state="show"; st.rerun()
                else:
                    st.error(f"Yanlış. Cevap: {st.session_state.iq_num}"); st.session_state.iq_state="start"; st.session_state.iq_lvl=1

# --- EXTRAS ---
with tab_music: st.video("https://www.youtube.com/watch?v=1eNSWZ4x2ZU")
with tab_creative:
    p=st.text_input("Konu:")
    if st.button("Yaz") and p:
        try: st.write(load_creative_llm().invoke(f"Write creative text about {p}").content)
        except: st.error("Hata")
with tab_settings: st.session_state.simlish_mode=st.toggle("Simlish")

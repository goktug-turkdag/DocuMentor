````markdown
# DocuMentor 📄 - Advanced AI Chatbot & Interactive Features

**DocuMentor** is a multi-functional Streamlit web application developed by Göktuğ Türkdağ. It started as an intelligent Q&A chatbot leveraging Retrieval-Augmented Generation (RAG) but has evolved into a comprehensive platform showcasing advanced AI integration, complex state management, and interactive features, including several games of chance and creative text generation.

This project serves as a portfolio piece demonstrating proficiency in building sophisticated, interactive AI applications using Python and modern AI frameworks.

---

## ✨ Features

The application is organized into several tabs, each offering distinct functionalities:

### 💬 Chatbot
* **Intelligent Q&A:** Answers questions based on a knowledge base using a RAG architecture.
* **Multi-Document Chat:** Users can upload multiple `.pdf`, `.docx`, or `.txt` files via the sidebar and chat specifically with their content.
* **Default Knowledge Base:** Utilizes the Databricks Dolly 15k dataset when no user files are uploaded.
* **Conversational Memory:** Remembers the context of the conversation for follow-up questions (`ConversationalRetrievalChain`).
* **Streaming Responses:** Answers are streamed token-by-token for a smoother user experience (`st.write_stream`).
* **Multilingual Understanding:** Can understand questions in various languages thanks to multilingual embeddings.
* **Source Highlighting:** Shows snippets of the source documents used to generate the answer.
* **Simlish Mode 👽:** An optional fun mode (toggleable in Settings) where the chatbot responds by imitating Simlish, demonstrating prompt engineering.
* **Developer FAQ Easter Egg:** The chatbot can answer specific questions about its developer, Göktuğ Türkdağ (skills, experience, contact), in multiple languages.
* **Content Moderation:** Includes an expanded safety barrier to detect and refuse to answer inappropriate, unethical, harmful, or sensitive political questions, providing helpful redirections where appropriate (e.g., for self-harm related queries).

### 🃏 Blackjack
* **Classic Game:** Full implementation of Blackjack rules.
* **Multiple Decks:** Uses a configurable number of decks (4, 6, or 8 via Settings).
* **Betting System:** Players bet using a shared balance.
* **Side Bets:** Includes popular side bets:
    * Perfect Pairs (Mixed, Colored, Perfect)
    * 21+3 (Flush, Straight, Trips, Straight Flush)
    * Lucky 7s (Pays on number of 7s)
    * Bust It! (Pays based on the number of cards the dealer busts with)
* **Advanced Rules:** Implements **Double Down** and **Insurance**.
* **Special Wins:** Includes the **5-Card Charlie** rule.
* **Dynamic Cash Out:** Offers a cash-out value based on a heuristic evaluation of the player's hand vs. the dealer's upcard before the player acts.
* **Visual Card Display:** Uses styled text and emojis to represent cards.
* **Basic Strategy Guide:** An expandable section explains optimal play.
* **History Tracking:** Records the outcome of the last 5 hands.

### 🪙 Coin Flip
* **Simple Betting:** Classic Heads or Tails game.
* **Balance Integration:** Uses the shared player balance.
* **Last Bet Memory:** Remembers the previous bet amount and choice.
* **History Tracking:** Records the outcome of the last 5 flips.
* **Simple Animation:** Includes a basic "Flipping..." visual effect.

### 🎡 Roulette
* **European Roulette:** Single zero wheel (0-36).
* **Basic Bets:** Supports betting on individual numbers (Straight Up) and outside bets (Red/Black, Odd/Even, Low/High).
* **Balance Integration:** Uses the shared player balance.
* **Last Bet Memory:** Remembers the previous bets placed.
* **History Tracking:** Records the outcome of the last 5 spins.
* **Visual Result:** Displays the winning number with a color emoji.
* **Simple Animation:** Includes a basic "Spinning..." visual effect.

### 🎰 Slots
* **Simple 3-Reel Slot:** Classic slot machine mechanic.
* **Weighted Symbols:** Uses different probabilities for symbols (`random.choices`).
* **Payout Table:** Clear payouts for 3-of-a-kind and cherry combinations on the middle line.
* **Balance Integration:** Uses the shared player balance.
* **Last Bet Memory:** Remembers the previous bet amount.
* **History Tracking:** Records the outcome of the last 5 spins.
* **Simple Animation:** Includes a basic reel "spinning" effect using `time.sleep`.

### 🃏 Video Poker (Jacks or Better)
* **Classic Game:** Standard Jacks or Better rules (Pair of Jacks or higher wins).
* **5-Card Draw:** Player receives 5 cards and chooses which ones to hold.
* **Payout Table:** Displays standard payout ratios for poker hands.
* **Balance Integration:** Uses the shared player balance (bet 1-5 credits).
* **History Tracking:** Records the outcome of the last 5 hands.
* **Visual Card Display:** Uses styled text and emojis with clear "Hold" indication.

### 📊 Stats
* **Overall Performance:** Tracks Starting Balance, Current Balance, Total Bets Placed, Net Profit/Loss, Biggest Win, Biggest Loss.
* **Performance by Game:** Shows Played, Won, Lost, Push (for BJ), and Win Rate (%) for each game.
* **Reset Option:** Allows resetting statistics independently.

### 🎶 Music Player
* **YouTube Embed:** Plays a selected YouTube playlist (currently a Blues playlist) using `st.video`.

### 🎨 Creative Corner
* **Text Generation:** Uses the creative Gemini model (`load_creative_llm`) to generate:
    * Short Poems
    * Story Ideas
    * Haikus
    * Tweets
* **User Prompts:** Takes a topic or theme from the user as input.

### ⚙️ Settings
* **Simlish Mode Toggle:** Enable/disable the chatbot's Simlish personality.
* **Blackjack Deck Count:** Select the number of decks used in the Blackjack game.
* **General Info:** Points to the sidebar reset button.

---

## 🛠️ Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Core Logic:** Python 3.11+
* **AI/LLM Framework:** [LangChain](https://python.langchain.com/)
* **LLM:** Google Gemini Pro (via `langchain-google-genai`)
* **Embeddings:** HuggingFace Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2` via `langchain-huggingface`)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (via `langchain-community`)
* **Data Handling:** `datasets` (for Dolly 15k), `PyPDFLoader`, `Docx2txtLoader`, `TextLoader` (via `langchain-community`), `tempfile`
* **Environment Variables:** `python-dotenv`
* **Utilities:** `random`, `time`, `collections.Counter`, `math`

---

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/goktug-turkdag/DocuMentor.git](https://github.com/goktug-turkdag/DocuMentor.git) # Replace with your actual repo URL if different
    cd DocuMentor
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Make sure your `requirements.txt` file includes all necessary packages:
    ```txt
    streamlit
    python-dotenv
    datasets
    langchain
    langchain-core
    langchain-community
    langchain-huggingface
    langchain-google-genai
    google-generativeai
    sentence-transformers
    chromadb
    pypdf
    docx2txt
    # Add any other specific versions if needed
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    In the root directory of the project, create a file named `.env` and add your Google API key:
    ```ini
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```
    *(You can get an API key from Google AI Studio.)*

---

## ▶️ Running the Application

Once the setup is complete, run the Streamlit app from your terminal:

```bash
streamlit run app.py
````

The application should open automatically in your web browser.

-----

## usage

  * Navigate between the different features (Chatbot, Games, Music, Settings) using the tabs at the top.
  * **Chatbot:** Ask questions in the input bar. Use the sidebar to upload your own documents to chat with them instead of the default knowledge base.
  * **Games:** Follow the on-screen instructions to place bets and play. Use the sidebar button to reset game states and balance.
  * **Settings:** Toggle Simlish mode or adjust Blackjack deck settings.

-----

## 💻 Codebase

This entire application, including the AI logic, multiple game implementations, and UI, is contained within a single `app.py` file exceeding 2000 lines. While this demonstrates the ability to manage a large script, in a production scenario, breaking the code into multiple modules (e.g., `chatbot.py`, `blackjack.py`, `utils.py`) would be recommended for better maintainability.

-----

## 👨‍💻 Developer

Developed by **Göktuğ Türkdağ**.

  * **LinkedIn:** [linkedin.com/in/goktugturkdag](https://www.google.com/search?q=https://www.linkedin.com/in/goktugturkdag)
  * **GitHub:** [github.com/goktug-turkdag](https://www.google.com/search?q=https://github.com/goktug-turkdag)
  * **Book a Meeting:** [cal.com/goktugturkdag](https://www.google.com/search?q=https://cal.com/goktugturkdag)

-----

Enjoy using DocuMentor\!

```

```

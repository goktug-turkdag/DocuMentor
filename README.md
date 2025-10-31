-----

# DocuMentor 📄

**[🚀 View the Live Application Here\!](https://www.google.com/search?q=https://documentor1.streamlit.app/)**

-----

**DocuMentor** is a sophisticated, multi-functional Streamlit application that combines an advanced Retrieval-Augmented Generation (RAG) chatbot with a suite of interactive games and creative tools. This project is designed to showcase complex AI integrations, advanced state management, and a feature-rich user interface.

## 🚀 Core Features

The application is organized into multiple tabs, accessible via a main navigation bar:

### 1\. 💬 Intelligent RAG Chatbot

  * **Multi-Document Q\&A:** Allows users to upload multiple `.pdf`, `.docx`, and `.txt` files simultaneously and ask questions based on their content.
  * **LLM Engine:** Powered by **Google's Gemini Pro** model (via LangChain) to understand context and generate answers.
  * **Vector Store:** Chunks and embeds uploaded documents into a **ChromaDB** vector store for efficient retrieval.
  * **Embeddings:** Uses the `paraphrase-multilingual-MiniLM-L12-v2` model (from HuggingFace) for multilingual text vectorization.
  * **Default Knowledge Base:** When no files are uploaded, the chatbot defaults to a knowledge base built from the `databricks-dolly-15k` dataset.
  * **Conversational Memory:** Uses `ConversationalRetrievalChain` to maintain the context of the conversation.
  * **Streaming Responses:** Answers are streamed token-by-token for a dynamic, "live" user experience.
  * **Content Moderation:** Features an extensive safety barrier to filter potentially harmful, unethical, or political queries, providing guided responses.
  * **Easter Eggs:**
      * **Simlish Mode:** A toggle in the Settings tab that makes the chatbot respond in the playful, nonsensical language from "The Sims."
      * **Developer FAQ:** Provides hard-coded, detailed answers to questions about the developer (Göktuğ Türkdağ), his skills, experience, and contact info.

### 2\. 🎲 Interactive Games (Advanced State Management)

All games run on pure Python logic, share a single `player_balance`, and are managed entirely within Streamlit's `session_state`.

  * **🃏 Blackjack:**

      * **Full-Featured Gameplay:** A complete Blackjack experience against a dealer.
      * **Side Bets:** Includes four optional side bets: Perfect Pairs (up to 25:1), 21+3 (up to 40:1), Lucky 7s (up to 100:1), and Bust It\! (up to 5:1).
      * **Core Features:**
          * **Split:** Fully functional hand-splitting on any two cards of the same value (e.g., 8-8, 10-J, Q-K).
          * **Double Down:** Allows the player to double their bet on any initial two-card hand (e.g., 8-3, 5-5, A-2).
          * **Insurance:** Offered when the dealer shows an Ace.
          * **5-Card Charlie:** Automatically wins if the player draws 5 cards without busting.
      * **AI Coach:** Uses Google Gemini Pro to analyze the player's (Hit, Stand, Double, Split) moves against basic strategy and provides actionable feedback at the end of the hand.

  * **🎡 Roulette:**

      * **European Style:** Single-zero roulette wheel.
      * **Betting:** Supports inside (straight-up number) and all outside (Red/Black, Odd/Even, Low/High) bets.

  * **🎰 Slot Machine:**

      * **Classic 3-Reel:** A simple, single-payline slot machine.
      * **Symbols & Payouts:** Features 7 symbols (🍒, 🍋, 🍊, 🍉, ⭐, 💎, ❼) with payouts for 2 Cherries (2x) or any 3-of-a-kind (up to 100x).

  * **🃏 Video Poker:**

      * **Jacks or Better:** Classic 5-card draw poker where a pair of Jacks or better wins.
      * **Hold Mechanic:** Full card-holding functionality with standard poker hand payouts (up to 800x for a Royal Flush).

  * **🪙 Coin Flip:**

      * A simple 1:1 Heads or Tails betting game.

### 3\. 📊 Player Stats

  * **Overall Performance:** Tracks Starting Balance, Current Balance, Total Wagered, Net Profit/Loss, Biggest Win, and Biggest Loss.
  * **Per-Game Breakdown:** A detailed dataframe showing stats for each game (Played, Won, Lost, Push, Win Rate %).

### 4\. 🎨 Creative Corner

  * **Generative AI Playground:** Uses a high-temperature instance of Gemini Pro to generate creative text based on user prompts.
  * **Generation Types:** Short Poem, Story Idea, Haiku (5-7-5 syllables), and Tweet (max 280 chars).

### 5\. 🎶 Music Player

  * **Embedded Media:** An embedded YouTube player that can play music while browsing other tabs (playback may stop on some browsers/mobile).
  * **User Choice:** A radio button allows the user to switch between:
      * **Blues 🎶** (A relaxing blues playlist)
      * **Guilty pleasures? 🤫** (manifest - Arıyo)

-----

## 🛠️ Technology Stack

  * **Core Language:** Python
  * **Web Framework:** Streamlit
  * **AI & LLM:** Google Gemini Pro
  * **LLM Orchestration:** LangChain
  * **Embeddings:** HuggingFace Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
  * **Vector Store:** ChromaDB
  * **Data Handling:** `datasets` (HuggingFace), `pypdf`, `docx2txt`

-----

## 🏁 Setup & Installation

To run this project on your local machine, follow these steps.

### 1\. Prerequisites

  * Python 3.9 or newer.
  * A valid [Google API Key](https://aistudio.google.com/app/apikey) for Google Gemini.

### 2\. Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/your-username/docuMentor.git
    cd docuMentor
    ```

2.  Create and activate a virtual environment:

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the required Python packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### 3\. Configuration

1.  Create a file named `.env` in the root directory of the project.

2.  Add your Google API Key to this file in the following format:

    ```env
    GOOGLE_API_KEY="AIzaSy...your_google_api_key_here"
    ```

### 4\. Running the Application

Launch the Streamlit app from your terminal:

```bash
streamlit run app.py
```

The application will now be running and accessible in your web browser, typically at `http://localhost:8501`.

-----

## 📦 Sample `requirements.txt`

This project relies on the following major libraries:

```
streamlit
python-dotenv
datasets
langchain
langchain-community
langchain-google-genai
langchain-huggingface
chromadb
sentence-transformers
pypdf
docx2txt
tiktoken
```

-----

## 👤 Developer

This project was developed by **Göktuğ Türkdağ**.

  * **LinkedIn:** [linkedin.com/in/goktugturkdag](https://www.linkedin.com/in/goktugturkdag)
  * **GitHub:** [github.com/goktug-turkdag](https://github.com/goktug-turkdag)
  * **Book a Meeting:** [cal.com/goktugturkdag](https://cal.com/goktugturkdag)

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

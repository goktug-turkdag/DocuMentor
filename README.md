# DocuMentor: A RAG-Powered Q&A Chatbot for Technical Documents

**DocuMentor** is an intelligent, web-based chatbot designed to navigate and answer questions from dense technical knowledge bases. Developed as a final project for the Akbank GenAI Bootcamp, this application showcases the power of Retrieval-Augmented Generation (RAG) in making complex information accessible and understandable.

## The Problem: Information Overload in Technical Fields

Professionals in fields like business administration, public finance, and tax law often face the challenge of navigating vast and complex datasets. Finding a specific piece of information within thousands of pages of technical documentation can be a daunting task. A simple Ctrl+F search is often insufficient as it lacks contextual understanding. The core problem is the inability to ask a direct question and receive a precise, relevant answer—or to be quickly informed if the information is outside the document's scope.

## The Solution: Intelligent, Context-Aware Retrieval

**DocuMentor** addresses this challenge by acting as an intelligent search assistant. Instead of just matching keywords, it leverages a sophisticated RAG architecture to *understand* the user's query and retrieve the most relevant information from its knowledge base. It provides users with:

* **Precise Answers:** Delivers direct answers based on the context provided in the dataset.

* **Semantic Understanding:** Goes beyond keyword matching to find conceptually related information.

* **Scope Awareness:** Can infer when a question cannot be answered by the available data.

## Knowledge Base: The Databricks Dolly 15k Dataset

**Dataset:** [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Description:** This project is powered by the Dolly 15k dataset, a high-quality collection of \~15,000 instruction/response pairs crowdsourced from Databricks employees. It covers diverse categories such as information extraction, brainstorming, and summarization.

**Application:** The "context" paragraphs from this dataset, many of which are sourced from Wikipedia, serve as the foundational knowledge base for DocuMentor. The chatbot relies exclusively on this context to formulate its answers.

## System Architecture and Technology Stack

DocuMentor is built on a RAG (Retrieval-Augmented Generation) pipeline orchestrated with the LangChain framework. The system's workflow is as follows:

1. **Ingestion:** The "context" texts from the databricks-dolly-15k dataset are loaded into the system.

2. **Embedding:** Each text document is converted into a numerical vector representation using HuggingFaceEmbeddingswith thesentence-transformers/all-MiniLM-L6-v2 model.

3. **Indexing:** These vectors are stored and indexed in a FAISS vector database, allowing for efficient, high-speed semantic searches.

4. **Retrieval:** When a user asks a question, it is also converted into a vector. The FAISS database is then queried to find the most semantically similar text chunks (the context)

5. **Generation:** The retrieved context and the original user query are passed to the Google Gemini (gemini-pro-latest) model. The model then generates a human-like, contextually accurate answer.

**Technology Stack:**

* **Web Framework:** Streamlit

* **RAG Orchestration:** LangChain

* **Generation Model:** Google Gemini (gemini-pro-latest)

* **Embedding Model:** Hugging Face (sentence-transformers/all-MiniLM-L6-v2)

* **Vector Database:** FAISS (CPU)

## Getting Started

To run this project on your local machine, please follow the steps below.

1. **Clone the Repository:**

`   ````

   git clone https://github.com/goktug-turkdag/DocuMentor.git

   cd DocuMentor

`   ````

2. **Set Up and Activate the Virtual Environment:**

`   ````

   python -m venv venv

   venv\Scripts\activate  # On Windows

   # source venv/bin/activate  # On macOS/Linux

`   ````

3. **Install Dependencies:**

`   ````

   pip install --upgrade -r requirements.txt

`   ````

4. **Configure the API Key:**


  * Create a file named.env in the root directory.


  * Add your Google API key to the file in the following format:GOOGLE_API_KEY="YOUR_API_KEY_HERE"``

5. **Run the Application:**

`   ````

   streamlit run app.py

`   ````

   *Note: The initial startup may take a few minutes as the application needs to download the dataset and models for the first time.*

## Live Demo

A live version of the DocuMentor application is available at the following link:
➡️ [**https://documentor1.streamlit.app**](https://documentor1.streamlit.app)

*Developed by **Göktuğ Türkdağ** as a part of the Akbank GenAI Bootcamp to merge a passion for AI with new skills.*


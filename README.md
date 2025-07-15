 🤖 RAG Chatbot with Gemini + LangChain

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain** and **Google's Gemini API**. It takes PDF documents as input, creates vector embeddings using HuggingFace, stores them in a FAISS vector database, and uses a Large Language Model to answer user queries based on the document content.

---

 📁 Project Structure
 rag-chatbot/
│
├── chatbot/
│ ├── init.py
│ ├── app.py # Main chatbot script
│ ├── loader.py # Loads and splits PDF files
│ ├── embedder.py # Embedding generation and vector store logic
│ ├── gemini_llm.py # Gemini API wrapper
│ └── rag_chain.py # Chain configuration for retrieval-based QA
│
├── data/ # Folder for PDF documents
├── vectorstore/ # Folder where FAISS index is saved
├── venv/ # Python virtual environment (excluded in Git)
├── .env # Environment variables (API key, paths, etc.)
├── .gitignore # Git ignore rules
└── requirements.txt # Python dependencies



  Setup Instructions

 1. Clone the Repository

git clone https://github.com/areeba42197/rag-chatbot.git
cd rag-chatbot

2. Install Dependencies

pip install -r requirements.txt

3. Add PDF Documents

4. Run the Chatbot
python -m chatbot.app








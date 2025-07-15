 🤖 RAG Chatbot with Gemini + LangChain

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain** and **Google's Gemini API**. It takes PDF documents as input, creates vector embeddings using HuggingFace, stores them in a FAISS vector database, and uses a Large Language Model to answer user queries based on the document content.

---

 📁 Project Structure
 rag-chatbot/
│
├── chatbot/
│ ├── init.py
│ ├── app.py 
│ ├── loader.py 
│ ├── embedder.py 
│ ├── gemini_llm.py
│ └── rag_chain.py 
│
├── data/ 
├── vectorstore/ 
├── venv/ 
├── .env # Environment variables 
├── .gitignore
└── requirements.txt



  **Setup Instructions**

 1. Clone the Repository

git clone https://github.com/areeba42197/rag-chatbot.git
cd rag-chatbot

2. Install Dependencies

pip install -r requirements.txt

3. Add PDF Documents

4. Run the Chatbot
python -m chatbot.app








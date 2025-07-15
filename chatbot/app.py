from dotenv import load_dotenv
import os
from chatbot.loader import load_and_split
from chatbot.embedder import create_and_store_embeddings, load_vectorstore
from chatbot.rag_chain import create_qa_chain

# Load environment variables from .env
load_dotenv()

DOC_PATH = os.getenv("DOC_PATH")
VDB_PATH = os.getenv("VECTORDB_PATH")

# Load or create vector store
if not os.path.exists(VDB_PATH):
    print("🔄 Creating vector store from documents in folder...")
    docs = load_and_split(DOC_PATH)
    vectorstore = create_and_store_embeddings(docs, VDB_PATH)
else:
    print("✅ Loading existing vector store...")
    vectorstore = load_vectorstore(VDB_PATH)

# Set up retrieval-based question answering chain
qa_chain = create_qa_chain(vectorstore)

# Start interactive CLI chatbot
print("\n💬 Chatbot Ready! Ask your questions (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = qa_chain.invoke({"query": user_input})
    print("Bot:", response["result"])



from dotenv import load_dotenv
import os
from chatbot.loader import load_and_split, create_and_store_embeddings, load_vectorstore
from chatbot.rag_chain import answer_with_rag_and_tools  # IMPORTANT: your rag.py function

load_dotenv()

DOC_PATH = os.getenv("DOC_PATH")
VDB_PATH = os.getenv("VECTORDB_PATH")

# Create or load the vector store
if not os.path.exists(VDB_PATH):
    print("🔄 Creating vector store from documents in folder...")
    docs = load_and_split(DOC_PATH)
    vectorstore = create_and_store_embeddings(docs, VDB_PATH)
else:
    print("✅ Loading existing vector store...")
    vectorstore = load_vectorstore(VDB_PATH)


print(f"\n💬 Chatbot Ready  Ask your questions (type 'exit' to quit)\n")

chat_history = []  # For multi-turn history tracking (optional)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    try:
        # Call your rag+tools orchestrator; pass history to retain context
        answer = answer_with_rag_and_tools(user_input, vectorstore, history=chat_history)
        print("Bot:", answer)
        print("-" * 60)

        # Update chat history: add user message and bot response
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})

        # Optional: keep history size manageable (last N messages)
        MAX_HISTORY = 10
        if len(chat_history) > MAX_HISTORY * 2:
            chat_history = chat_history[-MAX_HISTORY * 2 :]

    except Exception as e:
        print("❌ Error:", e)
        print("-" * 60)

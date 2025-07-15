from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

def create_and_store_embeddings(docs, save_path: str):
    """
    Create embeddings and save them in a FAISS vector store.
    :param docs: List of document chunks.
    :param save_path: Directory to save FAISS index.
    :return: FAISS vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(save_path)
    return vectorstore

def load_vectorstore(load_path: str):
    """
    Load an existing FAISS vector store from disk.
    :param load_path: Directory containing FAISS index.
    :return: FAISS vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
     folder_path=load_path,
     embeddings=embedding_model,
     allow_dangerous_deserialization=True  # ✅ Trust your own FAISS index
)



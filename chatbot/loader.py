from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_and_split(folder_path: str):
    """
    Load all PDF files in a folder and split them into text chunks.
    :param folder_path: Path to the folder containing PDF files.
    :return: List of all chunks from all PDFs.
    """
    all_chunks = []
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)

    return all_chunks

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






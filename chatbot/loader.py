from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
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



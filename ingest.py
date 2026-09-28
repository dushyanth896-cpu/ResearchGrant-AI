"""
Alternate/legacy ingestion pipeline: PDF grant documents -> FAISS.

NOTE: This is a separate pipeline from the one the app actually
uses. app.py and rag.py read grant_index.pkl, which is produced by
build_index.py from the JSON files in data/. This script instead
builds a FAISS vector store from PDF files in data/ using LangChain.

It's kept here in case you want to ingest grant announcements that
only exist as PDFs, but it isn't wired into app.py or rag.py, and
running it will not affect what the Streamlit app retrieves. If you
don't have PDF grant documents to ingest, you can safely ignore or
delete this file (and drop its dependencies from requirements.txt).
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data"
VECTOR_PATH = BASE_DIR / "vectorstore"


def build_vectorstore():

    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} document pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(str(VECTOR_PATH))

    print("Vector database created successfully.")


if __name__ == "__main__":
    build_vectorstore()

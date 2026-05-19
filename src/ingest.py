import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_DIR = "data/raw"
CHROMA_DB_DIR = "chroma_db"

def ingest_pdfs(directory: str) -> list[Document]:
    """
    Ingest PDF files from a directory into a list of documents.
    """
    documents = []
    for file in os.listdir(directory):
        if not file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(directory, file)
        pdf = fitz.open(pdf_path)
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": file, "page": page_num + 1}))
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk the documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    return text_splitter.split_documents(documents)

def build_vectorstore(chunks: list[Document]) -> Chroma:
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)

if __name__ == "__main__":
    print("Ingesting PDFs...")
    documents = ingest_pdfs(DATA_DIR)
    print(f"Ingested {len(documents)} documents")
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Chunked {len(chunks)} chunks")
    print("Building vector store...")
    vectorstore = build_vectorstore(chunks)
    print("Vectorstore built")
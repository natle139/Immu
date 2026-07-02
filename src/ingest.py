import fitz
import os
import re
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_DIR = "data/raw"
CHROMA_DB_DIR = "chroma_db"

# ponytail: filename-based tag instead of content classification, upgrade if filenames stop being a reliable signal
LEGISLATION_PREFIXES = ("F2026C00324", "C2026C00090")

# print-to-PDF header/footer cruft (timestamps, "Page N of M", raw URLs) that
# otherwise eats a big chunk of short fact-sheet chunks and dilutes their embedding
BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}\s*[ap]m$", re.IGNORECASE),
    re.compile(r"^Page \d+ of \d+$"),
    re.compile(r"^https?://\S+$"),
]

def clean_text(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not any(p.match(line.strip()) for p in BOILERPLATE_LINE_PATTERNS)]
    return "\n".join(kept)

def ingest_pdfs(directory: str) -> list[Document]:
    """
    Ingest PDF files from a directory into a list of documents.
    """
    documents = []
    for file in os.listdir(directory):
        if not file.endswith(".pdf"):
            continue
        doc_type = "legislation" if file.startswith(LEGISLATION_PREFIXES) else "fact_sheet"
        pdf_path = os.path.join(directory, file)
        pdf = fitz.open(pdf_path)
        for page_num, page in enumerate(pdf):
            text = clean_text(page.get_text())
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file, "page": page_num + 1, "doc_type": doc_type},
                ))
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk the documents into smaller chunks. Splits before Schedule 8 visa
    condition codes (e.g. "\n8101\n") first, so each condition's definition
    stays together instead of being merged with its neighbors.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=[r"\n(?=\d{4}\n)", r"\n\n", r"\n", r"\.", r" "],
        is_separator_regex=True,
    )
    return text_splitter.split_documents(documents)

def build_vectorstore(chunks: list[Document]) -> Chroma:
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)

if __name__ == "__main__":
    if os.path.exists(CHROMA_DB_DIR):
        # rebuilds from scratch each run; Chroma.from_documents appends, so a stale dir duplicates every chunk
        print(f"Removing existing {CHROMA_DB_DIR}...")
        shutil.rmtree(CHROMA_DB_DIR)
    print("Ingesting PDFs...")
    documents = ingest_pdfs(DATA_DIR)
    print(f"Ingested {len(documents)} documents")
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Chunked {len(chunks)} chunks")
    print("Building vector store...")
    vectorstore = build_vectorstore(chunks)
    print("Vectorstore built")
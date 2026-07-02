from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "chroma_db"
# ponytail: picked from observed score distribution (relevant matches ~0.6-0.9, irrelevant ~1.35+), not a formula
CONFIDENCE_THRESHOLD = 1.0

def build_retriever() -> Chroma:
    """
    Build a retriever from the Chroma vectorstore.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
    return vectorstore

def retrieve_documents(query: str, vectorstore, k: int=4) -> list[Document] | None:
    """
    Retrieve documents from the Chroma vectorstore. Prefers plain-English fact
    sheets over raw legislation (they vastly outnumber fact sheets and would
    otherwise dominate nearest-neighbor search); falls back to the full
    corpus, including legislation, only if the fact sheets have no confident match.
    """
    results = vectorstore.similarity_search_with_score(query, k=k, filter={"doc_type": "fact_sheet"})
    filtered_results = [(doc, score) for doc, score in results if score < CONFIDENCE_THRESHOLD]

    if not filtered_results:
        results = vectorstore.similarity_search_with_score(query, k=k)
        filtered_results = [(doc, score) for doc, score in results if score < CONFIDENCE_THRESHOLD]

    if not filtered_results:
        return None, [] # No relevant documents found

    documents = [doc for doc, _ in filtered_results]
    sources = list({f"{doc.metadata['source']} p.{doc.metadata['page']}" for doc in documents})
    return documents, sources
import re

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "chroma_db"
# ponytail: picked from observed score distribution (relevant matches ~0.6-0.9, irrelevant ~1.35+), not a formula
CONFIDENCE_THRESHOLD = 1.0

VISA_SUBCLASS_QUERY_PATTERN = re.compile(r"\bbridging\s+visa\s+([a-e])\b|\bbv([a-e])\b", re.IGNORECASE)

def detect_visa_subclass(query: str) -> str | None:
    match = VISA_SUBCLASS_QUERY_PATTERN.search(query)
    if not match:
        return None
    letter = (match.group(1) or match.group(2)).upper()
    return f"BV{letter}"

def build_retriever() -> Chroma:
    """
    Build a retriever from the Chroma vectorstore.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
    return vectorstore

def retrieve_documents(query: str, vectorstore, k: int=8) -> list[Document] | None:
    """
    Retrieve documents from the Chroma vectorstore. If the query names a
    specific subclass (e.g. "bridging visa B", "BVC"), search is narrowed to
    that subclass first -- every fact sheet has near-identical section
    headers ("Cost", "Travel", etc.), so pure semantic similarity alone can't
    reliably tell them apart.

    A narrow tier finding *something* doesn't mean it found the *best* match
    -- e.g. a query naming "Bridging Visa C" can still be best answered by a
    fact sheet that isn't tagged to any subclass (like the RAILS work-rights
    sheet). So at least two tiers are always searched and merged (subclass
    match first, then any fact sheet) rather than stopping at the first tier
    that clears the confidence threshold. Only escalates to legislation if
    the fact-sheet tiers come up empty.
    """
    subclass = detect_visa_subclass(query)
    if subclass:
        tiers = [
            {"$and": [{"doc_type": "fact_sheet"}, {"visa_subclass": subclass}]},
            {"doc_type": "fact_sheet"},
            {"visa_subclass": subclass},
            None,
        ]
    else:
        tiers = [{"doc_type": "fact_sheet"}, None]

    seen = set()
    merged = []
    for flt in tiers:
        for doc, score in vectorstore.similarity_search_with_score(query, k=k, filter=flt):
            if score >= CONFIDENCE_THRESHOLD:
                continue
            key = (doc.metadata["source"], doc.metadata["page"], doc.page_content)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))
        if len(merged) >= k:
            break

    if not merged:
        return None, [] # No relevant documents found

    documents = [doc for doc, _ in merged[:k]]
    sources = list({f"{doc.metadata['source']} p.{doc.metadata['page']}" for doc in documents})
    return documents, sources
import re

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "chroma_db"
# ponytail: picked from observed score distribution (relevant matches ~0.6-0.9, irrelevant ~1.35+), not a formula
CONFIDENCE_THRESHOLD = 1.0
MIN_CHUNK_CHARS = 60
TOP_K = 8

# only "visa" is required, so typos like "briding visa B" still match; group 2 catches lists ("visa A or B", "visas A, B and C")
VISA_LETTER_PATTERN = re.compile(r"\bvisas?\s+([a-e])\b((?:\s*(?:,|/|&|\bor\b|\band\b)\s*[a-e]\b)*)", re.IGNORECASE)
BV_CODE_PATTERN = re.compile(r"\bbv([a-e])\b", re.IGNORECASE)
SUBCLASS_NUMBERS = {"010": "BVA", "020": "BVB", "030": "BVC", "050": "BVE", "051": "BVE"}
SUBCLASS_NUMBER_PATTERN = re.compile(r"\b(010|020|030|050|051)\b")
# ponytail: any "br..." word before "visa" is taken as a misspelt "bridging"; widen if other typos show up in logs
BRIDGING_TYPO_PATTERN = re.compile(r"\bbr[a-z]*(?=\s+visas?\b)", re.IGNORECASE)
# "visa B" alone scores every BVB chunk past the threshold; the fact sheets all say "Bridging visa B"
MISSING_BRIDGING_PATTERN = re.compile(r"(?<!bridging )\b(?=visas?\s+[a-e]\b)", re.IGNORECASE)
# MiniLM can't rank 4-digit condition codes, so these get exact-match tiers
CONDITION_CODE_PATTERN = re.compile(r"\b8\d{3}\b")

def detect_visa_subclasses(query: str) -> list[str]:
    """Every bridging visa subclass the query names, in order of first mention."""
    found = []
    for m in VISA_LETTER_PATTERN.finditer(query):
        found.append(m.group(1))
        # lowercase "a" inside a list is usually the article ("visa B and a lawyer"), not BVA
        found += [c for c in re.findall(r"\b([a-e])\b", m.group(2), re.IGNORECASE) if c != "a"]
    found += BV_CODE_PATTERN.findall(query)
    codes = [f"BV{c.upper()}" for c in found]
    codes += [SUBCLASS_NUMBERS[n] for n in SUBCLASS_NUMBER_PATTERN.findall(query)]
    return list(dict.fromkeys(codes))

def build_retriever() -> Chroma:
    """
    Build a retriever from the Chroma vectorstore.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
    return vectorstore

def retrieve_documents(query: str, vectorstore, k: int=TOP_K) -> list[Document] | None:
    """
    Retrieve documents from the Chroma vectorstore. If the query names a
    specific subclass (e.g. "bridging visa B", "BVC"), search is narrowed to
    that subclass first -- every fact sheet has near-identical section
    headers ("Cost", "Travel", etc.), so pure semantic similarity alone can't
    reliably tell them apart.

    A narrow tier finding *something* doesn't mean it found the *best* match
    -- e.g. a query naming "Bridging Visa C" can still be best answered by a
    fact sheet that isn't tagged to any subclass (like the RAILS work-rights
    sheet). So each named subclass only gets a share of the k slots, and the
    any-fact-sheet tier always fills the rest. Sharing also means a question
    naming several subclasses ("visa A or B") sees every one of their fact
    sheets instead of the first filling all k. Only escalates to legislation
    if the fact-sheet tiers come up short.
    """
    # typos like "briding visa" push every score past the threshold, so embed the corrected spelling
    query = MISSING_BRIDGING_PATTERN.sub("bridging ", BRIDGING_TYPO_PATTERN.sub("bridging", query))
    subclasses = detect_visa_subclasses(query)
    # 1 visa -> 6 of 8 slots, 2 -> 3 each, 3+ -> 2 each; the rest go to the any-fact-sheet tier
    per_subclass = max(2, 3 * k // (4 * len(subclasses))) if subclasses else 0
    # (metadata filter, exact-text filter, slots); a condition code gets its Schedule 8 definition
    # ("8101\nThe holder must not...") plus fact sheets that mention it
    tiers = []
    for c in dict.fromkeys(CONDITION_CODE_PATTERN.findall(query)):
        tiers += [(None, {"$or": [{"$contains": f"{c}\nThe holder"}, {"$contains": f"{c}\n(1) The holder"}]}, 1),
                  ({"doc_type": "fact_sheet"}, {"$contains": c}, 2)]
    tiers += [({"$and": [{"doc_type": "fact_sheet"}, {"visa_subclass": s}]}, None, per_subclass) for s in subclasses]
    tiers += [({"doc_type": "fact_sheet"}, None, k), (None, None, k)]

    seen = set()
    merged = []
    for flt, contains, limit in tiers:
        added = 0
        for doc, score in vectorstore.similarity_search_with_score(query, k=k, filter=flt, where_document=contains):
            # an exact code match is relevant whatever the embedding thinks
            if (score >= CONFIDENCE_THRESHOLD and not contains) or added >= limit or len(merged) >= k:
                continue
            # ponytail: title-only chunks ("Bridging visa B") score high but carry no facts; drop at query time, filter in ingest if it spreads
            if len(doc.page_content.strip()) < MIN_CHUNK_CHARS:
                continue
            key = (doc.metadata["source"], doc.metadata["page"], doc.page_content)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))
            added += 1
        if len(merged) >= k:
            break

    if not merged:
        return None, [] # No relevant documents found

    documents = [doc for doc, _ in merged[:k]]
    sources = list({f"{doc.metadata['source']} p.{doc.metadata['page']}" for doc in documents})
    return documents, sources
if __name__ == "__main__":
    # self-check: python -m src.retriever
    assert detect_visa_subclasses("how much does briding visa B cost") == ["BVB"]
    assert detect_visa_subclasses("should i apply for bridging visa A or B") == ["BVA", "BVB"]
    assert detect_visa_subclasses("bridging visas A, B and C") == ["BVA", "BVB", "BVC"]
    assert detect_visa_subclasses("Compare BVA vs BVE") == ["BVA", "BVE"]
    assert detect_visa_subclasses("subclass 050 work rights") == ["BVE"]
    assert detect_visa_subclasses("on visa B and a lawyer said") == ["BVB"]
    assert detect_visa_subclasses("can I work while my visa is processed") == []
    assert BRIDGING_TYPO_PATTERN.sub("bridging", "how much does briding visa B cost") == "how much does bridging visa B cost"
    assert BRIDGING_TYPO_PATTERN.sub("bridging", "Bridgin visas A") == "bridging visas A"
    assert MISSING_BRIDGING_PATTERN.sub("bridging ", "how much does visa B cost") == "how much does bridging visa B cost"
    assert MISSING_BRIDGING_PATTERN.sub("bridging ", "Bridging Visa C fee") == "Bridging Visa C fee"
    assert CONDITION_CODE_PATTERN.findall("condition 8101 and 8503") == ["8101", "8503"]
    print("ok")

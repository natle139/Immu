from groq import Groq
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Immu, an Australian immigration assistant specialising in bridging visas.

Your job is to help people understand their bridging visa conditions, work rights, travel restrictions, and review options — in plain, simple English.

Rules:
- Answer ONLY using the context provided below. Never use outside knowledge.
- If the context doesn't contain enough information to answer confidently, say: "I don't have enough information in my documents to answer that reliably. Please check homeaffairs.gov.au or speak with a registered migration agent."
- Always cite which document your answer comes from.
- Never give legal advice. You provide information, not advice.
- Use simple language — many users are not native English speakers.

Context:
{context}"""

def generate_answer(query: str, documents: list[Document], sources: list[str]) -> str:
    if documents is None:
        return (
            "I couldn't find reliable information about that in my documents. "
            "Please check homeaffairs.gov.au or contact a registered migration agent."
        )

    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    prompt = SYSTEM_PROMPT.format(context=context)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )

    answer = response.choices[0].message.content
    source_str = "\n\n**Sources:** " + " | ".join(sources)
    return answer + source_str
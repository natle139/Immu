from groq import Groq
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# override when Groq retires a model (it retired llama-3.1-8b-instant)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = 0  # eval reproducibility; Groq may still vary slightly run to run

SYSTEM_PROMPT = """You are Immu, an Australian immigration assistant specialising in bridging visas.

Your job is to help people understand their bridging visa conditions, work rights, travel restrictions, and review options — in plain, simple English.

Rules:
- Answer ONLY using the context provided below. Never use outside knowledge.
- If the context doesn't contain enough information to answer confidently, say: "I don't have enough information in my documents to answer that reliably. Please check homeaffairs.gov.au or speak with a registered migration agent."
- If the context answers part of the question, answer that part and say which part you couldn't find.
- If the context gives different answers for different situations, explain each situation instead of refusing.
- Check which visa the question is about. If the context only covers a different visa, say you don't have information on the visa they asked about. Never present facts about one visa as the answer for another.
- If the question asks about several visas, cover each one the context has information on.
- Do not name or cite documents in your answer; the app lists the sources separately.
- Never give legal advice: don't tell the person what they should do. For "should I" questions, explain what the context says about each option so they can decide, instead of refusing.
- Use simple language — many users are not native English speakers.
- Format with short paragraphs and simple "- " bullet lists. No tables or headings.

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
        model=MODEL,
        reasoning_effort="low",  # reasoning tokens count against max_tokens
        temperature=TEMPERATURE,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content
FROM python:3.13-slim

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch: the default CUDA wheel adds GBs the free CPU Space can't use
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user . .
# builds chroma_db and caches the embedding model into the image, so startup needs no downloads
RUN python -m src.ingest

EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]

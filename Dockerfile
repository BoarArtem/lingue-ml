FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y build-essential git wget curl zstd && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN python -m nltk.downloader punkt wordnet omw-1.4 stopwords averaged_perceptron_tagger averaged_perceptron_tagger_eng
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download de_core_news_sm

COPY . .

RUN mkdir -p /models

EXPOSE 8000

CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull $OLLAMA_MODEL_NAME && python /app/storage/download_files.py && uvicorn app:app --host 0.0.0.0 --port 8000"]
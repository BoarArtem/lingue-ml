FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y build-essential git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 averaged_perceptron_tagger_eng
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download de_core_news_sm

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
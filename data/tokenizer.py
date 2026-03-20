import nltk
from fontTools.varLib.models import subList
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from pathlib import Path
import pandas as pd
import re
import pymorphy3
from razdel import tokenize
import spacy
import jieba
from data.preprocess import spam_classification_preprocess


morph = pymorphy3.MorphAnalyzer()
nlp_es = spacy.load("es_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_de = spacy.load("de_core_news_sm")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "english_corpus.txt"

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def sentence_preprocess_english(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]

    return lemmatized_words

def sentence_preprocess_russian(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    tokens = [token.text for token in tokenize(sentence)]

    lemmatized_words = [
        morph.parse(word)[0].normal_form
        for word in tokens
    ]

    return lemmatized_words

def sentence_preprocess_spanish(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    doc = nlp_es(sentence)

    lemmatized_words = [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return lemmatized_words

def sentence_preprocess_france(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    doc = nlp_fr(sentence)

    lemmatized_words = [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return lemmatized_words

def sentence_preprocess_german(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    doc = nlp_de(sentence)

    lemmatized_words = [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return lemmatized_words

def sentence_preprocess_chinese(sentence: str) -> list[str]:
    """
    :param sentence: Предложение которое надо очистить для задачи
    :return: Возвращает очищенное предложение
    """
    tokens = list(jieba.cut(sentence))

    return tokens

def vocabulary_expander_corpus():
    sentences = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.lower().strip()

            if not line:
                continue

            line = re.sub(r"[^a-z\s]", "", line)
            tokens = line.split()

            if len(tokens) > 2:
                sentences.append(tokens)

    return sentences

def spam_classification_tokenizer():
    data = spam_classification_preprocess("datasets/spam_Emails_data.csv")

    texts = data['text']

    # tokenization
    tokens_par_text = [word_tokenize(s) for s in texts]
    tokens = [word for sublist in tokens_par_text for word in sublist]

    return tokens

    # Исправить - TypeError: expected string or bytes-like object, got 'float'


# if __name__ == "__main__":
#     print(spam_classification_tokenizer())
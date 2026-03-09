import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

import pymorphy3
from razdel import tokenize

import spacy

import jieba

morph = pymorphy3.MorphAnalyzer()
nlp_es = spacy.load("es_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_de = spacy.load("de_core_news_sm")

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

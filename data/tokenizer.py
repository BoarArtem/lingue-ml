import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

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


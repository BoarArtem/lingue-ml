from gensim.models import KeyedVectors

wv = KeyedVectors.load("../inference/google_news.kv", mmap="r")

def vocabulary_expander(arr: list) -> list:
    """
    Модель будет смотреть на список пользователя и подбирать максимально те слова которые больше всего подходят для этого списка
    :param arr: Список со словами пользователя. Иначе - колода пользователя
    :return: Возвращает нам метод (positive=arr) - список пользователя по которому модель будет подбирать слова, (topn=10) - топ-10 слов которые имеют высший процент схожести
    """
    return wv.most_similar(
        positive=arr,
        topn=10
    )

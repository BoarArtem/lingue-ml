from gensim.models import KeyedVectors

class VocabularyExpander:
    def __init__(self, wv: KeyedVectors):
        self.wv = wv

    def expand(self, arr: list[str], topn: int) -> list[tuple[str, float]]:
        """
        Находит слова, наиболее близкие к заданному списку.
        :param arr: Список слов для анализа
        :param topn: Количество возвращаемых результатов
        :return: Список кортежей (слово, схожесть)
        """
        return self.wv.most_similar(positive=arr, topn=topn)

    @classmethod
    def load_model(cls, path: str = "../inference/google_news.kv"):
        wv = KeyedVectors.load(path, mmap='r')
        return cls(wv)
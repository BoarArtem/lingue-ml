from gensim.models import KeyedVectors


class VocabularyExpander:
    def __init__(self, wv: KeyedVectors):
        self.wv = wv

    def expand(self, arr: list[str], topn: int = 20) -> list[tuple[str, float]]:
        valid_words = [w for w in arr if w in self.wv]

        if not valid_words:
            return []

        results = self.wv.most_similar(valid_words, topn=topn * 3)

        cleaned = []
        for word, score in results:
            if (
                word.isalpha()
                and word.islower()
                and word not in arr
                and len(word) > 2
            ):
                cleaned.append((word, score))

        return cleaned[:topn]

    @classmethod
    def load_model(cls):
        wv = KeyedVectors.load("../inference/crawl_fasttext.kv")
        return cls(wv)


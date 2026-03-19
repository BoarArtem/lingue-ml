from pathlib import Path
from gensim.models import Word2Vec
from data.tokenizer import vocabulary_expander_corpus, vocabulary_expander_russian


class VocabularyExpander:

    def __init__(self, vector_size, window, min_count, workers):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def training(self):
        corpus = vocabulary_expander_corpus()
        corpus_russian = vocabulary_expander_russian()


        model = Word2Vec(
            sentences=corpus_russian,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )

        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "russian_word2vec.model"

        model.save(str(model_path))

        self.model = model
        return model

    def expand(self, words: list[str], topn: int = 10):
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.wv.most_similar(words, topn=topn * 3)


models = VocabularyExpander(
    vector_size=300,
    window=5,
    min_count=5,
    workers=4
)

print(models.training())
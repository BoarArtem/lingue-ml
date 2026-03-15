from pathlib import Path
from gensim.models import Word2Vec
from data.tokenizer import vocabulary_expander_corpus


class VocabularyExpander:

    def __init__(self, vector_size, window, min_count, workers):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def training(self):
        corpus = vocabulary_expander_corpus()

        model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )

        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "word2vec.model"

        model.save(str(model_path))

        self.model = model
        return model

    def expand(self, words: list[str], topn: int = 10):
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.wv.most_similar(words, topn=topn * 3)
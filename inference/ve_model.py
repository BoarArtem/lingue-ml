from gensim.models import Word2Vec

model_path = "russian_word2vec.model"

model = Word2Vec.load(model_path)

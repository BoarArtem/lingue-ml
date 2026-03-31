from gensim.models import Word2Vec

model_path = "word2vec.model"

model = Word2Vec.load(model_path)

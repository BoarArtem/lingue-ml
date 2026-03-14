from gensim.models import Word2Vec

model_path = "word2vec.model"

model = Word2Vec.load(model_path)

print(model.wv.most_similar(["pizza", "sushi", "coffee", "chocolate", "bread"], topn=10))
